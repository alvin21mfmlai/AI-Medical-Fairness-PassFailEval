"""The whole pipeline on the toy example: offline with the mock backend, and over HTTP against a stand-in
OpenAI-compatible server (stdlib only), which exercises the real client, retries and the validator's feedback loop."""
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from passfail_eval.cli import main
from passfail_eval.io import read_csv, read_jsonl
from passfail_eval.mock import mock_judge

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy"


def toy_copy(tmp_path: Path) -> Path:
    d = tmp_path / "toy"
    shutil.copytree(TOY, d, ignore=shutil.ignore_patterns("out*"))
    return d


def test_toy_with_mock_judges(tmp_path, capsys):
    d = toy_copy(tmp_path)
    main(["run", "--config", str(d / "config_mock.yaml")])
    out = d / "out_mock"
    grades = read_csv(out / "grades.csv")
    assert len(grades) == 36 * 2 * 2 * 2                 # answers x judges x keys x repeats
    assert {g["pass"] for g in grades} == {"True", "False"}
    ladder = read_csv(out / "flips" / "ladder.csv")
    whats = {(r["scale"], r["ladder"], r["what"]) for r in ladder}
    for w in ("regrade", "judge", "key", "group", "twin", "excess"):
        assert ("pass", "all judges", w) in whats
    assert len(read_csv(out / "flips" / "questions.csv")) == 4
    capsys.readouterr()
    main(["grade", "--config", str(d / "config_mock.yaml")])   # resumes: nothing is sent again
    assert "to_run=72" not in capsys.readouterr().err
    assert all(len(read_jsonl(p)) == 72 for p in (out / "judge").glob("*/*.jsonl"))
    # edit one key: only its answers are graded again, and the older grades of that key are not counted twice
    keys = read_jsonl(d / "keys.jsonl")
    keys[0]["essential_points"][0]["text"] += " (edited)"
    (d / "keys.jsonl").write_text("".join(json.dumps(k) + "\n" for k in keys))
    capsys.readouterr()
    main(["grade", "--config", str(d / "config_mock.yaml")])
    err = capsys.readouterr().err
    assert "to_run=9" in err and "older grades of edited answers or keys left out" in err
    assert len(read_csv(out / "grades.csv")) == 36 * 2 * 2 * 2


def test_draft_keys_with_mock_author(tmp_path):
    d = toy_copy(tmp_path)
    main(["draft-keys", "--config", str(d / "config_mock.yaml"), "--author", "mock-author"])
    keys = read_jsonl(d / "out_mock" / "keys_mock-author.jsonl")
    assert len(keys) == 4 and all(k["essential_points"][0]["id"] == "E1" for k in keys)


class FakeOpenAI(BaseHTTPRequestHandler):
    """Answers /v1/chat/completions like an OpenAI-compatible server. The first request is refused with 429; every
    first attempt at a grade carries an invented quote (which the validator must send back); a retry is answered by
    the mock judge."""
    seen: list = []
    refused = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.seen.append(body)
        if not FakeOpenAI.refused:
            FakeOpenAI.refused = True
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "rate limited"}}')
            return
        msgs = body["messages"]
        grade = json.loads(mock_judge(msgs[0]["content"]))
        if len(msgs) == 1:  # first attempt: invent a quote
            grade["unsupported_claims"] = [{"quote": "this sentence is not in the answer", "explanation": "x"}]
        text = json.dumps(grade)
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
               "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_openai_compatible_server(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        d = toy_copy(tmp_path)
        cfg = yaml.safe_load((d / "config_mock.yaml").read_text())
        cfg.update(out="out_http", repeats=["a"], judges=[{
            "name": "http-judge", "backend": "openai", "base_url": f"http://127.0.0.1:{srv.server_port}/v1",
            "model": "some-model", "max_concurrency": 4, "temperature": 0.0, "max_tokens": 1234,
            "extra_body": {"reasoning_effort": "medium"}}])
        (d / "c.yaml").write_text(yaml.safe_dump(cfg))
        main(["grade", "--config", str(d / "c.yaml")])
        rows = read_jsonl(d / "out_http" / "judge" / "http-judge" / "grades__rep-a.jsonl")
        assert len(rows) == 72 and all(r["parse_error"] is None and r["error"] is None for r in rows)
        assert all(r["parse_retries"] == 1 and "not word for word" in r["first_error"] for r in rows)
        first = FakeOpenAI.seen[1]
        assert first["max_tokens"] == 1234 and first["temperature"] == 0.0 and first["reasoning_effort"] == "medium"
        assert len(read_csv(d / "out_http" / "grades.csv")) == 72
    finally:
        srv.shutdown()
