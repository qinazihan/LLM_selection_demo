#!/usr/bin/env python3
"""
Probe a list of OpenAI/Anthropic chat models and record which ones respond.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from tqdm import tqdm


DEFAULT_OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.2-chat-latest",
    "gpt-5.1",
    "gpt-5.1-chat-latest",
    "gpt-5",
    "gpt-5-chat-latest",
    "gpt-5-pro",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "o1",
    "o1-pro",
    "o3",
    "o3-mini",
    "o3-pro",
    "o4-mini",
]

DEFAULT_ANTHROPIC_MODELS = [
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
]


def parse_models_file(path: Path) -> Tuple[List[str], List[str]]:
    """Extract model ids from a mixed JSON/JSONL-like file."""
    if not path.exists():
        return [], []
    text = path.read_text(encoding="utf-8")
    # Pull all "id": "<model>" occurrences
    ids = re.findall(r'"id"\s*:\s*"([^"]+)"', text)
    openai_ids: List[str] = []
    anthropic_ids: List[str] = []
    for mid in ids:
        if mid.startswith("claude"):
            anthropic_ids.append(mid)
        else:
            openai_ids.append(mid)
    # de-dupe preserving order
    def dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedupe(openai_ids), dedupe(anthropic_ids)


def save_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_openai(model: str, prompt: str, base_url: str, api_key: str, timeout: float) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        latency = time.perf_counter() - t0
        data = resp.json()
        return {
            "status": "ok",
            "latency_s": latency,
            "http_status": resp.status_code,
            "response_excerpt": data["choices"][0]["message"]["content"][:120] if data.get("choices") else None,
        }


def call_anthropic(model: str, prompt: str, base_url: str, api_key: str, timeout: float) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 32,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        latency = time.perf_counter() - t0
        data = resp.json()
        return {
            "status": "ok",
            "latency_s": latency,
            "http_status": resp.status_code,
            "response_excerpt": data.get("content", [{}])[0].get("text", "")[:120],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check which models respond successfully.")
    parser.add_argument("--out", default="outputs/model_checks.jsonl", help="Output JSONL path.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds.")
    parser.add_argument("--prompt", default="Quick availability check: reply with 'ok'.", help="Probe prompt.")
    parser.add_argument(
        "--models-file",
        default="model_access.txt",
        help="Path to file containing model ids to probe (if present).",
    )
    args = parser.parse_args()

    dotenv_path = None
    for candidate in ("API_configs/.env", ".env"):
        if Path(candidate).exists():
            dotenv_path = candidate
            break
    load_dotenv(dotenv_path=dotenv_path)

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_base = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    anthropic_base = os.getenv("ANTHROPIC_API_BASE_URL", "https://api.anthropic.com")

    # Load model ids (from file if present, else defaults)
    openai_models_file, anthropic_models_file = parse_models_file(Path(args.models_file))
    openai_models = openai_models_file or DEFAULT_OPENAI_MODELS
    anthropic_models = anthropic_models_file or DEFAULT_ANTHROPIC_MODELS

    rows: List[Dict[str, Any]] = []

    # OpenAI models
    if openai_key:
        for model in tqdm(openai_models, desc="OpenAI models"):
            row: Dict[str, Any] = {"provider": "openai", "model": model}
            try:
                result = call_openai(model, args.prompt, openai_base, openai_key, args.timeout)
                row.update(result)
            except Exception as exc:
                row.update({"status": "error", "error": str(exc)})
            rows.append(row)
    else:
        rows.append({"provider": "openai", "model": None, "status": "error", "error": "OPENAI_API_KEY not set"})

    # Anthropic models
    if anthropic_key:
        for model in tqdm(anthropic_models, desc="Anthropic models"):
            row = {"provider": "anthropic", "model": model}
            try:
                result = call_anthropic(model, args.prompt, anthropic_base, anthropic_key, args.timeout)
                row.update(result)
            except Exception as exc:
                row.update({"status": "error", "error": str(exc)})
            rows.append(row)
    else:
        rows.append({"provider": "anthropic", "model": None, "status": "error", "error": "ANTHROPIC_API_KEY not set"})

    save_jsonl(rows, Path(args.out))
    print(f"Wrote {len(rows)} rows to {args.out}")

    usable = [
        r for r in rows
        if r.get("http_status") == 200 and r.get("response_excerpt")
    ]
    if usable:
        print("Usable models (HTTP 200 with non-empty response):")
        for r in usable:
            print(f"- {r['provider']}: {r['model']} (latency {r.get('latency_s'):.3f}s)")
    else:
        print("No usable models found.")


if __name__ == "__main__":
    main()
