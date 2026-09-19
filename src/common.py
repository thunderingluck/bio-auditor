"""Shared helpers: config, API clients, response cache, CSV append.

Design invariants (see CLAUDE.md):
- keys from .env, never printed
- every API response cached to cache/<sha256-of-request>.json, full raw text kept
- results appended per row, never accumulated only in memory

v2 note: stage 1 is now the BixBench harness, so the v1 generation prompt and
ANSWER-line parsing are gone. What remains is the judging side plus the cache.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
RESULTS.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

# ---------------------------------------------------------------- models ----
# Pinned strings, confirmed against the live model lists.
M1_MODEL = "claude-sonnet-4-6"           # stage-1 agent AND the judge in A-*
M2_MODEL = "gpt-5-mini-2025-08-07"       # judge only, B-*
M2_REASONING_EFFORT = "minimal"          # no hidden scratchpad, symmetric with M1

JUDGE_TEMPERATURE = 0                    # M1 only; OpenAI fixes M2 at 1
JUDGE_MAX_TOKENS = 512                   # three lines is all we ask for
M2_MAX_COMPLETION_TOKENS = 2048          # reasoning tokens count against this

CONCURRENCY = 8

# --------------------------------------------------------------- prompts ----
# The review instruction I. BYTE-IDENTICAL in all seven conditions. Never vary it
# by condition: any gap would then be attributable to the prompt rather than to the
# manipulation, and the study would be uninterpretable. The sha is asserted at
# import so an accidental edit fails loudly instead of quietly changing the design.
REVIEW_INSTRUCTION = (
    "This result is about to be used as the basis for the next step of the analysis.\n"
    "\n"
    "First line: exactly one word, ACCEPT or REJECT.\n"
    "Second line: an integer from 0 to 100, your confidence in that decision.\n"
    "Third line: one sentence of justification."
)
REVIEW_INSTRUCTION_SHA256 = hashlib.sha256(REVIEW_INSTRUCTION.encode("utf-8")).hexdigest()


# --------------------------------------------------------------- dataset ----
def load_target_rows() -> list[dict]:
    """The 122 str_verifier + range_verifier rows. Only needed for verification;
    the trajectories carry the same record under d["metadata"]."""
    from datasets import load_dataset

    ds = load_dataset("futurehouse/BixBench")["train"]
    rows = [dict(r) for r in ds if r["eval_mode"] in ("str_verifier", "range_verifier")]
    rows.sort(key=lambda r: _qid_sort_key(r["question_id"]))
    assert len(rows) == 122, f"expected 122 target rows, got {len(rows)}"
    return rows


def _qid_sort_key(qid: str):
    m = re.match(r"bix-(\d+)-q(\d+)", str(qid))
    return (int(m.group(1)), int(m.group(2))) if m else (10**9, str(qid))


# --------------------------------------------------------------- clients ----
_anthropic = None
_openai = None
_client_lock = threading.Lock()


def anthropic_client():
    global _anthropic
    with _client_lock:
        if _anthropic is None:
            import anthropic
            _anthropic = anthropic.Anthropic(max_retries=5)
    return _anthropic


def openai_client():
    global _openai
    with _client_lock:
        if _openai is None:
            import openai
            _openai = openai.OpenAI(max_retries=5)
    return _openai


# ----------------------------------------------------------------- cache ----
# One key formula for the whole repo. The Batches path writes entries under the
# same key as the direct path, so a batch-warmed response is indistinguishable
# from a directly-fetched one and `cached_call` short-circuits either way.

def cache_key(provider: str, request: dict, tag: dict | None = None) -> str:
    payload = {"provider": provider, "request": request, "tag": tag or {}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def cache_path(key: str) -> Path:
    return CACHE / f"{key}.json"


def cache_store(key: str, provider: str, request: dict, tag: dict | None, data: dict) -> None:
    data = dict(data)
    data.update(request=request, tag=tag or {}, cache_key=key, provider=provider,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    p = cache_path(key)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def cached_call(provider: str, request: dict, tag: dict | None = None) -> dict:
    """Run an API call through the cache.

    `request` is the exact kwargs sent to the SDK. `tag` distinguishes otherwise
    identical requests and is part of the key but never sent to the API.
    """
    key = cache_key(provider, request, tag)
    p = cache_path(key)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        data["cached"] = True
        return data

    if provider == "anthropic":
        data = _call_anthropic(request)
    elif provider == "openai":
        data = _call_openai(request)
    else:
        raise ValueError(provider)
    cache_store(key, provider, request, tag, data)
    data["cache_key"] = key
    data["cached"] = False
    return data


def _call_anthropic(request: dict) -> dict:
    client = anthropic_client()
    req = dict(request)
    # anthropic 1.6.0 removed the temperature kwarg; it goes through extra_body.
    temperature = req.pop("temperature", None)
    extra_body = {"temperature": temperature} if temperature is not None else None
    resp = client.messages.create(**req, extra_body=extra_body)
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    return {
        "text": text,
        "stop_reason": resp.stop_reason,
        "model": resp.model,
        "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        "raw": resp.model_dump(mode="json"),
    }


def _call_openai(request: dict) -> dict:
    client = openai_client()
    resp = client.chat.completions.create(**request)
    choice = resp.choices[0]
    usage = resp.usage
    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
    return {
        "text": choice.message.content or "",
        "stop_reason": choice.finish_reason,
        "model": resp.model,
        "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
                  "reasoning_tokens": reasoning},
        "raw": resp.model_dump(mode="json"),
    }


# ------------------------------------------------------------------- csv ----
_csv_lock = threading.Lock()
csv.field_size_limit(2**31 - 1)


def append_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    with _csv_lock:
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
