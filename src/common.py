"""Shared helpers: config, dataset loading, API clients, response cache, CSV append.

Design invariants (see CLAUDE.md):
- keys from .env, never printed
- every API response cached to cache/<sha256-of-request>.json, full raw text kept
- results appended per row
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
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
# Pinned strings. Confirmed against client.models.list() on 2026-09-16.
M1_MODEL = "claude-sonnet-4-6"           # answerer + self-judge. Accepts temperature; thinking off by default.
M2_MODEL = "gpt-5-mini-2025-08-07"       # other-family judge. Temperature fixed at 1 by OpenAI (400 otherwise).
M2_REASONING_EFFORT = "minimal"          # no hidden scratchpad, symmetric with M1 (no thinking)

GEN_TEMPERATURE = 1.0
JUDGE_TEMPERATURE = 0.0                  # M1 only; M2 cannot be set
GEN_MAX_TOKENS = 4096
JUDGE_MAX_TOKENS = 512
M2_MAX_COMPLETION_TOKENS = 2048

CONCURRENCY = 8
PILOT_SEED = 20250916
PILOT_N_PER_MODE = 10                    # 10 str + 10 range = 20 questions

# --------------------------------------------------------------- prompts ----
# No system prompt anywhere: the generation user turn is the complete TASK, so
# condition A can reconstruct the conversation byte-for-byte and B/C can quote it.
GEN_PROMPT_TEMPLATE = (
    "You are an expert bioinformatician. You will be asked a question that was originally "
    "answered by analysing a specific dataset. You do NOT have access to the data files or to "
    "a code execution environment. Use your domain knowledge and reasoning to give your best "
    "estimate of the answer.\n\n"
    "QUESTION:\n{question}\n\n"
    "Think step by step. Then end your response with a final line of exactly this form:\n"
    "ANSWER: <value>\n"
    "where <value> is a bare number or a short string with no units, explanation, or extra words on that line."
)

# The review instruction I. Byte-identical across all four conditions. Never vary it.
REVIEW_INSTRUCTION = (
    "Is the answer correct? Reply with exactly one word on the first line: CORRECT or INCORRECT. "
    "Then one sentence of justification."
)

ANSWER_RE = re.compile(r"^\s*\**\s*ANSWER\s*:\s*\**\s*(.+?)\s*\**\s*$", re.IGNORECASE | re.MULTILINE)


def build_gen_prompt(question: str) -> str:
    return GEN_PROMPT_TEMPLATE.format(question=question)


def extract_answer(response_text: str) -> str | None:
    """V = the value on the last ANSWER: line. None if absent."""
    matches = ANSWER_RE.findall(response_text or "")
    if not matches:
        return None
    return matches[-1].strip().strip("`").strip()


# --------------------------------------------------------------- dataset ----
def load_target_rows() -> list[dict]:
    """The 122 str_verifier + range_verifier rows, sorted by question_id. Never uses llm_verifier."""
    from datasets import load_dataset

    ds = load_dataset("futurehouse/BixBench")["train"]
    rows = [dict(r) for r in ds if r["eval_mode"] in ("str_verifier", "range_verifier")]
    rows.sort(key=lambda r: _qid_sort_key(r["question_id"]))
    assert len(rows) == 122, f"expected 122 target rows, got {len(rows)}"
    return rows


def _qid_sort_key(qid: str):
    m = re.match(r"bix-(\d+)-q(\d+)", qid)
    return (int(m.group(1)), int(m.group(2))) if m else (10**9, qid)


def pilot_question_ids(rows: list[dict]) -> list[str]:
    """Deterministic stratified pilot: PILOT_N_PER_MODE per eval_mode, seeded."""
    rng = random.Random(PILOT_SEED)
    out = []
    for mode in ("str_verifier", "range_verifier"):
        ids = sorted(r["question_id"] for r in rows if r["eval_mode"] == mode)
        out.extend(rng.sample(ids, PILOT_N_PER_MODE))
    return sorted(out, key=_qid_sort_key)


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
def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def cached_call(provider: str, request: dict, tag: dict | None = None) -> dict:
    """Run an API call through the cache.

    `request` is the exact kwargs sent to the SDK. `tag` is extra metadata that
    distinguishes otherwise-identical requests (e.g. sample_idx for temperature-1
    sampling) and is part of the cache key but never sent to the API.
    Returns dict with keys: text, stop_reason, model, usage, cached, raw.
    """
    key = _cache_key({"provider": provider, "request": request, "tag": tag or {}})
    path = CACHE / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["cached"] = True
        return data

    if provider == "anthropic":
        data = _call_anthropic(request)
    elif provider == "openai":
        data = _call_openai(request)
    else:
        raise ValueError(provider)
    data["request"] = request
    data["tag"] = tag or {}
    data["cache_key"] = key
    data["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    data["cached"] = False
    return data


def _call_anthropic(request: dict) -> dict:
    client = anthropic_client()
    req = dict(request)
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
    text = choice.message.content or ""
    usage = resp.usage
    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
    return {
        "text": text,
        "stop_reason": choice.finish_reason,
        "model": resp.model,
        "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens, "reasoning_tokens": reasoning},
        "raw": resp.model_dump(mode="json"),
    }


# ------------------------------------------------------------------- csv ----
_csv_lock = threading.Lock()


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
