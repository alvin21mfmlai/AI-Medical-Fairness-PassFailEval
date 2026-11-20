"""The run's settings, from one YAML file. Relative paths are read relative to the YAML file's folder."""
from __future__ import annotations

from pathlib import Path

import yaml

from .keys import DEFAULT_SCALE

DEFAULTS = {"out": "out", "prompt": "generic", "draft_prompt": "generic", "rule": "lenient", "strip_thinking": True,
            "parse_retries": 3, "repeats": ["a"], "neutral_values": ["neutral"],
            "bootstrap": {"samples": 2000, "seed": 0}, "judges": [], "authors": []}
PATHS = ("answers", "keys", "questions", "out")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = {**DEFAULTS, **(yaml.safe_load(path.read_text(encoding="utf-8")) or {})}
    base = path.resolve().parent
    for k in PATHS:
        v = cfg.get(k)
        if isinstance(v, list):
            cfg[k] = [str((base / x).resolve()) for x in v]
        elif v:
            cfg[k] = str((base / v).resolve())
    if cfg["prompt"] not in ("equitymedqa_sg", "generic"):
        cfg["prompt"] = str((base / cfg["prompt"]).resolve())
    if cfg["draft_prompt"] not in ("equitymedqa_sg", "generic"):
        cfg["draft_prompt"] = str((base / cfg["draft_prompt"]).resolve())
    for m in cfg["judges"] + cfg["authors"]:
        if m.get("api_key_file"):
            m["api_key_file"] = str((base / Path(m["api_key_file"]).expanduser()).resolve())
        m.setdefault("model", m.get("name"))
    names = [m.get("name") for m in cfg["judges"]]
    if len(set(names)) != len(names) or not all(names):
        raise SystemExit("every judge needs a unique 'name'")
    cfg["urgency_scale"] = cfg.get("urgency_scale") or DEFAULT_SCALE
    cfg["bootstrap"] = {**DEFAULTS["bootstrap"], **(cfg.get("bootstrap") or {})}
    return cfg
