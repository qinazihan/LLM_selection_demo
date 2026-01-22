#!/usr/bin/env python3
"""
End-to-end run: collect model responses then judge them in one pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

# ---------------------- Pricing (per 1K tokens) ---------------------- #
OPENAI_PRICING = {
    "gpt-4.1": (5.0, 15.0),
    "gpt-4.1-mini": (0.15, 0.60),
}

ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-latest": (3.0, 15.0),
    "claude-3-5-haiku-latest": (1.0, 5.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
}

SYSTEM_PROMPT = (
    "You are a strict evaluator. Score the model response against the rubric. "
    "Be consistent and only use the given rubric/policy."
)


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_case_map(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    case_map: Dict[str, Dict[str, Any]] = {}
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("case_id")
            if cid:
                case_map[cid] = row
    return case_map


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"__parse_error__": line.strip()}


def provider_for_model(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    tables = {"openai": OPENAI_PRICING, "anthropic": ANTHROPIC_PRICING}
    table = tables.get(provider, {})
    price = table.get(model)
    if price is None:
        for name, val in table.items():
            if model.startswith(name):
                price = val
                break
    if price is None:
        return None
    in_price, out_price = price
    return (prompt_tokens / 1000) * in_price + (completion_tokens / 1000) * out_price


@dataclass
class CallResult:
    response: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: Optional[float]


def call_openai(model: str, prompt: str, base_url: str, api_key: str, max_tokens: int, timeout: float) -> CallResult:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
    latency = time.perf_counter() - t0
    data = resp.json()
    message = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cost = estimate_cost("openai", model, prompt_tokens, completion_tokens)
    return CallResult(message.strip(), latency, prompt_tokens, completion_tokens, cost)


def call_anthropic(model: str, prompt: str, base_url: str, api_key: str, max_tokens: int, timeout: float) -> CallResult:
    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
    latency = time.perf_counter() - t0
    data = resp.json()
    message = data["content"][0]["text"]
    usage = data.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    cost = estimate_cost("anthropic", model, prompt_tokens, completion_tokens)
    return CallResult(message.strip(), latency, prompt_tokens, completion_tokens, cost)


def bullets(items: Optional[list[str]]) -> str:
    items = items or []
    if not items:
        return "- (none)"
    return "\n".join(f"- {i}" for i in items)


def build_user_prompt(case_id: str, model: str, prompt: str, expected: str, rubric: Dict[str, Any], response_text: str) -> str:
    hard = bullets(rubric.get("hard_requirements"))
    soft = bullets(rubric.get("soft_requirements"))
    penalties = bullets(rubric.get("penalties"))
    return (
        f"CASE_ID: {case_id}\n"
        f"MODEL: {model}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"EXPECTED (reference):\n{expected}\n\n"
        f"RUBRIC:\n"
        f"Hard requirements:\n{hard}\n"
        f"Soft requirements:\n{soft}\n"
        f"Penalties:\n{penalties}\n\n"
        f"MODEL RESPONSE:\n{response_text}\n\n"
        "Return ONLY valid JSON with schema:\n"
        "{\n"
        '"score": integer 1-10,\n'
        '"hard_pass": boolean,\n'
        '"hard_fail_reasons": [string],\n'
        '"reasons": [string],\n'
        '"error_tags": [string]\n'
        "}\n\n"
        "Rules:\n"
        "- If any hard requirement is violated -> hard_pass=false and score <= 4.\n"
        "- Penalize invented facts/policies/links heavily.\n"
        "- No extra text outside JSON."
    )


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def call_openai_judge(model: str, system_prompt: str, user_prompt: str, base_url: str, api_key: str, timeout: float) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 512,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def call_anthropic_judge(model: str, system_prompt: str, user_prompt: str, base_url: str, api_key: str, timeout: float) -> str:
    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 512,
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]


def parse_judge_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def save_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------- Plotting helpers ---------------------- #
def percentile(arr: List[float], pct: float) -> Optional[float]:
    if not arr:
        return None
    return float(np.percentile(arr, pct))


def aggregate(responses: List[Dict[str, Any]], judgements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    per_model: Dict[str, Dict[str, Any]] = {}
    for row in responses:
        model = row.get("model")
        if not model:
            continue
        stats = per_model.setdefault(model, {"costs": [], "latencies": []})
        cost = row.get("cost_usd")
        if isinstance(cost, (int, float)):
            stats["costs"].append(float(cost))
        lat = row.get("latency_s")
        if isinstance(lat, (int, float)):
            stats["latencies"].append(float(lat))

    for row in judgements:
        model = row.get("model")
        if not model or model not in per_model:
            continue
        stats = per_model[model]
        stats.setdefault("scores", [])
        stats.setdefault("hard_pass", [])
        score = row.get("score")
        if isinstance(score, (int, float)):
            stats["scores"].append(float(score))
        hard_pass = row.get("hard_pass")
        if isinstance(hard_pass, bool):
            stats["hard_pass"].append(hard_pass)

    aggregates: List[Dict[str, Any]] = []
    for model, stats in per_model.items():
        costs = stats.get("costs", [])
        lats = stats.get("latencies", [])
        scores = stats.get("scores", [])
        passes = stats.get("hard_pass", [])
        aggregates.append(
            {
                "model": model,
                "cost_mean": float(np.mean(costs)) if costs else None,
                "lat_mean": float(np.mean(lats)) if lats else None,
                "lat_p95": percentile(lats, 95),
                "score_mean": float(np.mean(scores)) if scores else None,
                "hard_pass_rate": (sum(passes) / len(passes)) if passes else None,
                "n_samples": len(scores) or len(lats) or len(costs),
            }
        )
    return aggregates


def pareto_frontier(points: List[Dict[str, Any]], cost_key: str, quality_key: str) -> List[Dict[str, Any]]:
    pts = [p for p in points if p.get(cost_key) is not None and p.get(quality_key) is not None]
    pts = sorted(pts, key=lambda x: (x[cost_key], -x[quality_key]))
    frontier: List[Dict[str, Any]] = []
    best_quality = -float("inf")
    for p in pts:
        q = p[quality_key]
        if q > best_quality:
            frontier.append(p)
            best_quality = q
    return frontier


def scatter_cost_quality(models: List[Dict[str, Any]], out_path: Path) -> None:
    if not models:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = []
    ys = []
    sizes = []
    labels = []
    for m in models:
        if m["cost_mean"] is None or m["score_mean"] is None:
            continue
        xs.append(m["cost_mean"])
        ys.append(m["score_mean"])
        sizes.append(300 * (m["lat_mean"] or 0.5))
        labels.append(m["model"])
    if not xs:
        plt.close(fig)
        return
    ax.scatter(xs, ys, s=sizes, alpha=0.6, edgecolor="k")
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, label, fontsize=9, ha="center", va="center")
    ax.set_xlabel("Cost (mean, USD)")
    ax.set_ylabel("Quality (mean score)")
    ax.set_title("Cost vs Quality (bubble = avg latency)")

    frontier = pareto_frontier(models, "cost_mean", "score_mean")
    if frontier:
        fx = [p["cost_mean"] for p in frontier]
        fy = [p["score_mean"] for p in frontier]
        ax.plot(fx, fy, "r--", label="Pareto frontier")
        ax.scatter(fx, fy, s=80, facecolors="none", edgecolors="red", linewidths=2)
        ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def scatter_latency_quality(models: List[Dict[str, Any]], out_path: Path) -> None:
    if not models:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = []
    ys = []
    sizes = []
    colors = []
    labels = []
    for m in models:
        lat = m["lat_p95"] if m["lat_p95"] is not None else m["lat_mean"]
        if lat is None or m["score_mean"] is None:
            continue
        xs.append(lat)
        ys.append(m["score_mean"])
        cost = m["cost_mean"]
        colors.append(cost if cost is not None else 0.0)
        sizes.append(300)
        labels.append(m["model"])
    if not xs:
        plt.close(fig)
        return
    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", alpha=0.7, edgecolor="k")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Cost (mean USD)")
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, label, fontsize=9, ha="center", va="center")
    ax.set_xlabel("Latency (p95 seconds)")
    ax.set_ylabel("Quality (mean score)")
    ax.set_title("Latency vs Quality (color = cost)")

    frontier = pareto_frontier(models, "lat_p95", "score_mean")
    if frontier:
        fx = [p["lat_p95"] for p in frontier if p["lat_p95"] is not None]
        fy = [p["score_mean"] for p in frontier if p["lat_p95"] is not None]
        if fx and fy:
            ax.plot(fx, fy, "r--", label="Pareto frontier (lat-quality)")
            ax.scatter(fx, fy, s=80, facecolors="none", edgecolors="red", linewidths=2)
            ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect responses then judge them.")
    parser.add_argument("--dataset", default="data/dataset_simple.jsonl", help="Path to JSONL dataset.")
    parser.add_argument("--case-id", help="Case id to run. If omitted, run all entries in the dataset.")
    parser.add_argument("--models", help="Comma-separated model ids (overrides MODEL_IDS env).")
    parser.add_argument(
        "--collect-out",
        help="Responses JSONL. Default: outputs/<dataset-stem>/<dataset-stem>_responses.jsonl",
    )
    parser.add_argument(
        "--judgements-out",
        help="Judgements JSONL. Default: outputs/<dataset-stem>/<dataset-stem>_judgements.jsonl",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens per model.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds.")
    args = parser.parse_args()

    dotenv_path = None
    for candidate in ("API_configs/.env", ".env"):
        if Path(candidate).exists():
            dotenv_path = candidate
            break
    load_dotenv(dotenv_path=dotenv_path)

    model_ids_env = args.models or os.getenv("MODEL_IDS")
    if not model_ids_env:
        raise SystemExit("MODEL_IDS not set (in env or via --models).")
    model_ids = [m.strip() for m in model_ids_env.split(",") if m.strip()]

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_base = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    anthropic_base = os.getenv("ANTHROPIC_API_BASE_URL", "https://api.anthropic.com")

    run_name = os.getenv("RUN_NAME", "demo_run")
    judge_model = os.getenv("JUDGE_MODEL_ID")
    if not judge_model:
        raise SystemExit("JUDGE_MODEL_ID is required.")
    judge_provider = provider_for_model(judge_model)
    if judge_provider == "openai" and not openai_key:
        raise SystemExit("OPENAI_API_KEY is required for the chosen judge model.")
    if judge_provider == "anthropic" and not anthropic_key:
        raise SystemExit("ANTHROPIC_API_KEY is required for the chosen judge model.")

    dataset_path = Path(args.dataset)
    dataset_rows = load_dataset(dataset_path)
    if args.case_id:
        dataset_rows = [row for row in dataset_rows if row.get("case_id") == args.case_id]
        if not dataset_rows:
            raise SystemExit(f"case_id {args.case_id} not found in {args.dataset}")
    dataset_stem = dataset_path.stem
    collect_out_path = (
        Path(args.collect_out)
        if args.collect_out
        else Path("outputs") / dataset_stem / f"{dataset_stem}_responses.jsonl"
    )
    judgements_out_path = (
        Path(args.judgements_out)
        if args.judgements_out
        else Path("outputs") / dataset_stem / f"{dataset_stem}_judgements.jsonl"
    )

    now = datetime.now(tz=timezone.utc).isoformat()

    # -------- Collect --------
    collect_rows: List[Dict[str, Any]] = []
    for entry in tqdm(dataset_rows, desc="Collecting", unit="case"):
        prompt = entry.get("prompt", "")
        case_id = entry.get("case_id")
        for model in model_ids:
            provider = provider_for_model(model)
            try:
                if provider == "openai":
                    if not openai_key:
                        raise RuntimeError("OPENAI_API_KEY not set")
                    result = call_openai(model, prompt, openai_base, openai_key, args.max_tokens, args.timeout)
                else:
                    if not anthropic_key:
                        raise RuntimeError("ANTHROPIC_API_KEY not set")
                    result = call_anthropic(model, prompt, anthropic_base, anthropic_key, args.max_tokens, args.timeout)
                row = {
                    "timestamp": now,
                    "run_name": run_name,
                    "case_id": case_id,
                    "model": model,
                    "provider": provider,
                    "prompt": prompt,
                    "response": result.response,
                    "latency_s": result.latency_s,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cost_usd": result.cost_usd,
                }
            except Exception as exc:
                row = {
                    "timestamp": now,
                    "run_name": run_name,
                    "case_id": case_id,
                    "model": model,
                    "provider": provider,
                    "prompt": prompt,
                    "response": None,
                    "latency_s": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": None,
                    "error": str(exc),
                }
            collect_rows.append(row)
    save_jsonl(collect_rows, collect_out_path)

    # -------- Judge --------
    case_map = load_case_map(dataset_path)
    judgements_rows: List[Dict[str, Any]] = []
    for resp_row in tqdm(iter_jsonl(collect_out_path), desc="Judging", unit="resp"):
        raw_error = None
        if "__parse_error__" in resp_row:
            raw_error = "response_json_parse_failed"
            case_id = None
            model = None
            response_text = resp_row["__parse_error__"]
        else:
            case_id = resp_row.get("case_id")
            model = resp_row.get("model")
            response_text = resp_row.get("response") or resp_row.get("response_text")

        case = case_map.get(case_id) if case_id else None
        row_out: Dict[str, Any] = {
            "run_name": run_name,
            "case_id": case_id,
            "model": model,
            "judge_model": judge_model,
            "score": None,
            "hard_pass": None,
            "hard_fail_reasons": [],
            "reasons": [],
            "error_tags": [],
            "judge_latency_ms": None,
            "error": None,
        }

        if raw_error:
            row_out["error"] = raw_error
            row_out["error_tags"] = ["response_json_parse_failed"]
            save_jsonl([row_out], judgements_out_path)
            continue

        if not case:
            row_out["error"] = "case_not_found"
            row_out["error_tags"] = ["case_not_found"]
            save_jsonl([row_out], judgements_out_path)
            continue

        if not response_text:
            row_out["error"] = "missing_response_text"
            row_out["error_tags"] = ["missing_response_text"]
            save_jsonl([row_out], judgements_out_path)
            continue

        user_prompt = build_user_prompt(
            case_id=case_id,
            model=model or "<unknown>",
            prompt=case.get("prompt", ""),
            expected=case.get("expected", ""),
            rubric=case.get("rubric", {}),
            response_text=response_text,
        )

        try:
            t0 = time.perf_counter()
            if judge_provider == "openai":
                judge_text = call_openai_judge(judge_model, SYSTEM_PROMPT, user_prompt, openai_base, openai_key, args.timeout)
            else:
                judge_text = call_anthropic_judge(judge_model, SYSTEM_PROMPT, user_prompt, anthropic_base, anthropic_key, args.timeout)
            latency_ms = (time.perf_counter() - t0) * 1000
            parsed = parse_judge_json(judge_text)
            if parsed is None:
                row_out["error"] = "judge_json_parse_failed"
                row_out["error_tags"] = ["judge_json_parse_failed"]
                row_out["judge_latency_ms"] = latency_ms
            else:
                row_out.update(
                    score=parsed.get("score"),
                    hard_pass=parsed.get("hard_pass"),
                    hard_fail_reasons=parsed.get("hard_fail_reasons") or [],
                    reasons=parsed.get("reasons") or [],
                    error_tags=parsed.get("error_tags") or [],
                    judge_latency_ms=latency_ms,
                    error=None,
                )
        except Exception as exc:
            row_out["error"] = str(exc)
            row_out["error_tags"] = ["judge_call_failed"]

        judgements_rows.append(row_out)
        save_jsonl([row_out], judgements_out_path)

    # -------- Plots --------
    plots_dir = Path("outputs") / dataset_stem / "plots"
    models = aggregate(collect_rows, judgements_rows)
    scatter_cost_quality(models, plots_dir / "cost_quality.png")
    scatter_latency_quality(models, plots_dir / "latency_quality.png")


if __name__ == "__main__":
    main()
