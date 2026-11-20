"""Model backends and a resumable, concurrent request runner. Nothing here depends on the hardware.

Backends
--------
openai   Any OpenAI-compatible chat-completions endpoint: vLLM or SGLang on any GPU, Ollama, LM Studio, the llama.cpp
         server, Hugging Face TGI, or a hosted API with an OpenAI-compatible route.
         Model entry (in the config):
           base_url: http://localhost:8000/v1       the server's /v1 address
           model: <name the server knows the model by>
           api_key_env: OPENAI_API_KEY              environment variable holding the key ("EMPTY" if unset: local servers)
           api_key_file: <path>                     or a KEY=VALUE file outside the project holding it; never logged
           max_concurrency: 16                      requests in flight at once
           timeout_s: 900
           extra_body: {reasoning_effort: medium}   passed through to the server (e.g. a reasoning switch)
           token_param: max_completion_tokens       for APIs that reject max_tokens
           fixed_sampling: {temperature: 1.0, top_p: 1.0}   for models that accept only their defaults: none is sent,
                                                    and each row records these values instead of the requested ones
           retry_attempts: 4                        network errors and 429 rate limits are retried with back-off
mock     Deterministic, offline replies, so that the whole pipeline can be tested without a model (see mock.py).

Every request is written as one JSONL row with the exact messages and decoding settings, the raw reply, usage and
timing, so a run is reproducible and can be resumed: a request whose key already has a valid row is not sent again.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .io import JsonlAppender, is_ok, now_iso, read_jsonl
from .jsonparse import extract_json


@dataclass
class Request:
    key: str
    task: str
    messages: list
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: int | None = None
    meta: dict = field(default_factory=dict)
    extra_body: dict | None = None  # merged over the model's extra_body
    validator: Callable[[str], tuple[Any, str | None]] | None = None


def fill_prompt(template: str, **values: Any) -> str:
    """Replace {{name}} placeholders (single braces are left alone, so JSON examples in a prompt are safe)."""
    names = set(re.findall(r"\{\{(\w+)\}\}", template))
    missing = sorted(names - set(values))
    if missing:
        raise KeyError(f"prompt placeholders not filled: {missing}")
    out = template
    for k in names:
        v = values[k]
        out = out.replace("{{" + k + "}}", v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
    return out


def user_message(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ============================================================================ backends
class Backend:
    def __init__(self, mcfg: dict):
        self.cfg = mcfg
        self.model = mcfg.get("model", mcfg.get("name"))
        self.max_concurrency = int(mcfg.get("max_concurrency", 16))

    async def complete(self, req: Request) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    async def aclose(self):
        pass


def key_from_file(path: str, name: str) -> str | None:
    """The value of `name` in a KEY=VALUE file (optional `export`, quotes stripped), or None."""
    for line in Path(os.path.expanduser(path)).read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.strip().removeprefix("export ").strip() == name:
            return v.strip().strip("'\"")
    return None


class OpenAIBackend(Backend):
    def __init__(self, mcfg: dict):
        super().__init__(mcfg)
        from openai import AsyncOpenAI

        key_name = mcfg.get("api_key_env", "OPENAI_API_KEY")
        if mcfg.get("api_key_file"):
            api_key = key_from_file(mcfg["api_key_file"], key_name)
            if not api_key:
                raise SystemExit(f"{mcfg.get('name')}: no {key_name}= line in {mcfg['api_key_file']}")
        else:
            api_key = os.environ.get(key_name, "EMPTY")
        self.client = AsyncOpenAI(base_url=mcfg["base_url"], api_key=api_key,
                                  timeout=float(mcfg.get("timeout_s", 900)), max_retries=0)
        self.extra_body = mcfg.get("extra_body") or {}
        self.token_param = mcfg.get("token_param", "max_tokens")
        self.fixed_sampling = bool(mcfg.get("fixed_sampling"))

    async def complete(self, req: Request) -> dict:
        kwargs: dict[str, Any] = dict(model=self.model, messages=req.messages, **{self.token_param: req.max_tokens})
        if not self.fixed_sampling:
            kwargs.update(temperature=req.temperature, top_p=req.top_p)
        if req.seed is not None:
            kwargs["seed"] = req.seed
        extra = {**self.extra_body, **(req.extra_body or {})}
        if extra:
            kwargs["extra_body"] = extra
        resp = await self.client.chat.completions.create(**kwargs)
        ch = resp.choices[0]
        msg = ch.message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        usage = resp.usage.model_dump() if resp.usage else None
        return {"text": msg.content or "", "reasoning": reasoning, "finish_reason": ch.finish_reason,
                "usage": usage, "served_model": resp.model,
                "system_fingerprint": getattr(resp, "system_fingerprint", None)}

    async def aclose(self):
        await self.client.close()


class MockBackend(Backend):
    async def complete(self, req: Request) -> dict:
        from .mock import mock_reply

        await asyncio.sleep(0)
        text = mock_reply(req.task, req.messages, float(self.cfg.get("mock_yes", 0.6)),
                          float(self.cfg.get("mock_partial", 0.3)))
        user = req.messages[-1]["content"]
        return {"text": text, "reasoning": None, "finish_reason": "stop",
                "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4},
                "served_model": self.model}


def make_backend(mcfg: dict) -> Backend:
    kind = mcfg.get("backend", "openai")
    if kind == "openai":
        return OpenAIBackend(mcfg)
    if kind == "mock":
        return MockBackend(mcfg)
    raise ValueError(f"unknown backend {kind!r} (openai | mock)")


# ============================================================================ runner
Validator = Callable[[str], tuple[Any, str | None]]  # returns (parsed, error message or None)


def json_validator(schema_check: Callable[[dict], list[str]] | None = None) -> Validator:
    """Parse the last JSON object in a reply and run `schema_check` on it; its problems go back to the model."""
    def _v(text: str):
        try:
            obj = extract_json(text)
        except ValueError as e:
            return None, str(e)
        if schema_check:
            probs = schema_check(obj)
            if probs:
                return obj, "; ".join(probs)
        return obj, None

    return _v


def _retry_after(e: Exception) -> float | None:
    headers = getattr(getattr(e, "response", None), "headers", None)
    try:
        return float(headers.get("retry-after")) if headers and headers.get("retry-after") else None
    except (TypeError, ValueError):
        return None


async def _call_with_retries(backend: Backend, req: Request, attempts: int | None = None) -> dict:
    attempts = attempts or int(backend.cfg.get("retry_attempts", 4))
    delay = 2.0
    for i in range(attempts):
        try:
            return await backend.complete(req)
        except Exception as e:  # network or server errors
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= int(status) < 500 and int(status) not in (408, 409, 429):
                raise  # a bad request (e.g. context too long): retrying will not help
            if i == attempts - 1:
                raise
            print(f"[retry {i + 1}] {req.key}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            await asyncio.sleep(min(120.0, max(delay, _retry_after(e) or 0.0)) + random.random())
            delay = min(delay * 2, 120.0)
    raise RuntimeError("unreachable")


async def _one(backend: Backend, req: Request, parse_retries: int) -> dict:
    """One request. A reply that fails its validator is sent back with the problems, up to parse_retries times; a reply
    cut off by the token limit is retried with twice the budget instead."""
    t0 = time.time()
    res = await _call_with_retries(backend, req)
    parsed, perr, tries, first = None, None, 0, {}
    if req.validator:
        parsed, perr = req.validator(res["text"])
        if perr:  # the first reply and its problems are kept, e.g. to count invented quotes before correction
            first = {"first_text": res["text"], "first_error": perr}
        msgs, max_tok = list(req.messages), req.max_tokens
        while perr and tries < parse_retries:
            tries += 1
            if res.get("finish_reason") == "length":
                max_tok = min(max_tok * 2, 32768)
                retry_req = Request(**{**req.__dict__, "max_tokens": max_tok})
            else:
                msgs = msgs + [{"role": "assistant", "content": res["text"][:6000]},
                               {"role": "user", "content": f"Your reply could not be used: {perr}. "
                                                           "Reply again with ONLY the corrected JSON object, no other text."}]
                retry_req = Request(**{**req.__dict__, "messages": msgs, "max_tokens": max_tok})
            res = await _call_with_retries(backend, retry_req)
            parsed, perr = req.validator(res["text"])
    return {**res, "parsed": parsed, "parse_error": perr, "parse_retries": tries,
            "latency_s": round(time.time() - t0, 3), **first}


def _row(req: Request, mcfg: dict, res: dict | None, err: str | None) -> dict:
    request = {"messages": req.messages, "temperature": req.temperature, "top_p": req.top_p,
               "max_tokens": req.max_tokens, "seed": req.seed,
               "extra_body": {**(mcfg.get("extra_body") or {}), **(req.extra_body or {})} or None}
    fixed = mcfg.get("fixed_sampling")
    if fixed:
        request.update(temperature=fixed.get("temperature"), top_p=fixed.get("top_p"),
                       requested_temperature=req.temperature, requested_top_p=req.top_p,
                       sampling="provider default (not sent: the model accepts no other value)")
    if mcfg.get("token_param"):
        request["token_param"] = mcfg["token_param"]
    row = {"key": req.key, "task": req.task, "model_name": mcfg.get("name"), "model": mcfg.get("model"),
           "backend": mcfg.get("backend", "openai"), "base_url": mcfg.get("base_url"), "request": request,
           **req.meta, "timestamp": now_iso(), "error": err}
    if res:
        row.update({k: res.get(k) for k in ("text", "reasoning", "finish_reason", "usage", "served_model",
                                            "system_fingerprint", "parsed", "parse_error", "parse_retries", "latency_s")})
        row.update({k: res[k] for k in ("first_text", "first_error") if k in res})
    return row


async def _run_async(backend: Backend, mcfg: dict, reqs: list[Request], out: Path, parse_retries: int, desc: str):
    sem = asyncio.Semaphore(backend.max_concurrency)
    n_done, n_err, t0 = 0, 0, time.time()
    total = len(reqs)
    with JsonlAppender(out) as app:
        async def worker(r: Request):
            nonlocal n_done, n_err
            async with sem:
                try:
                    res = await _one(backend, r, parse_retries)
                    app.write(_row(r, mcfg, res, None))
                except Exception as e:
                    n_err += 1
                    app.write(_row(r, mcfg, None, f"{type(e).__name__}: {e}"))
                n_done += 1
                if n_done % max(10, total // 10) == 0 or n_done == total:
                    rate = n_done / max(1e-6, time.time() - t0)
                    print(f"  [{desc}] {n_done}/{total} ({rate:.2f}/s, errors={n_err})", file=sys.stderr)

        await asyncio.gather(*(worker(r) for r in reqs))
    await backend.aclose()


def run_requests(mcfg: dict, reqs: list[Request], out: str | Path, *, parse_retries: int = 3, desc: str = "llm",
                 limit: int | None = None) -> Path:
    """Send the requests whose key has no valid row in `out` yet, and append their rows to `out`."""
    out = Path(out)
    done = {r["key"] for r in read_jsonl(out) if is_ok(r)}
    todo = [r for r in reqs if r.key not in done]
    n_done = len(reqs) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    print(f"[{desc}] model={mcfg.get('name')} total={len(reqs)} already_done={n_done} "
          f"to_run={len(todo)} -> {out}", file=sys.stderr)
    if todo:
        asyncio.run(_run_async(make_backend(mcfg), mcfg, todo, out, parse_retries, desc))
    return out
