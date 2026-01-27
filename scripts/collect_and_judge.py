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

# ---------------------- Pricing (loaded from JSON per 1M tokens) ---------------------- #
PRICING_TABLE: Dict[str, Dict[str, tuple[float, float]]] = {"openai": {}, "anthropic": {}}

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


def load_pricing(pricing_path: Path) -> None:
    """Populate PRICING_TABLE from a JSON file with USD_per_1M_tokens."""
    global PRICING_TABLE
    PRICING_TABLE = {"openai": {}, "anthropic": {}}
    if not pricing_path.exists():
        return
    data = json.loads(pricing_path.read_text(encoding="utf-8"))
    # OpenAI
    openai_models = data.get("openai", {}).get("models", {})
    for name, prices in openai_models.items():
        input_price = prices.get("input")
        output_price = prices.get("output")
        if input_price is None or output_price is None:
            continue
        # Convert per 1M tokens to per token
        PRICING_TABLE["openai"][name] = (input_price / 1_000_000, output_price / 1_000_000)
    # Anthropic
    anthropic_models = data.get("anthropic", {}).get("models", {})
    for name, prices in anthropic_models.items():
        input_price = prices.get("base_input")
        output_price = prices.get("output")
        if input_price is None or output_price is None:
            continue
        PRICING_TABLE["anthropic"][name] = (input_price / 1_000_000, output_price / 1_000_000)


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    table = PRICING_TABLE.get(provider, {})
    price = table.get(model)
    if price is None:
        for name, val in table.items():
            if model.startswith(name):
                price = val
                break
    if price is None:
        return None
    in_price, out_price = price
    return prompt_tokens * in_price + completion_tokens * out_price


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
    # print(usage)
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
    # print(usage)
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


def place_labels(ax, xs: List[float], ys: List[float], labels: List[str], base_offset: float) -> None:
    """
    Place labels with a simple force-directed pass in screen space to reduce overlap.
    Stable on log axes; no extra deps.
    """
    if not xs:
        return
    transform = ax.transData
    inv = transform.inverted()
    pts = np.column_stack([xs, ys])
    screen_pts = transform.transform(pts)
    n = len(labels)
    # Initial positions: nudge upward in screen space
    base_pix = abs(transform.transform((0, base_offset))[1] - transform.transform((0, 0))[1])
    base_pix = max(base_pix, 10)
    positions = screen_pts.copy()
    positions[:, 1] += base_pix
    min_sep = 14.0  # desired minimum separation in pixels
    step = 0.01
    for _ in range(200):
        disp = np.zeros_like(positions)
        # Repel labels from each other
        for i in range(n):
            delta = positions[i] - positions
            dist2 = np.sum(delta * delta, axis=1)
            mask = dist2 > 1e-6
            delta = delta[mask]
            dist2 = dist2[mask]
            close = dist2 < (min_sep * min_sep)
            if not np.any(close):
                continue
            force = (delta[close] / (dist2[close][:, None] + 1e-6)) * ((min_sep * min_sep - dist2[close])[:, None])
            disp[i] += force.sum(axis=0)
        # Repel labels from their anchor points
        delta_anchor = positions - screen_pts
        dist2_anchor = np.sum(delta_anchor * delta_anchor, axis=1)
        close_anchor = dist2_anchor < (min_sep * min_sep)
        if np.any(close_anchor):
            force_anchor = (delta_anchor[close_anchor] / (dist2_anchor[close_anchor][:, None] + 1e-6)) * (
                (min_sep * min_sep - dist2_anchor[close_anchor])[:, None]
            )
            disp[close_anchor] += force_anchor
        positions += disp * step
        # Keep x near the anchor to avoid large sideways drift
        positions[:, 0] = screen_pts[:, 0]
    # Explicitly place gpt-5 below its point to reduce collisions
    for idx, label in enumerate(labels):
        if label == "gpt-5":
            positions[idx, 0] = screen_pts[idx, 0]
            positions[idx, 1] = screen_pts[idx, 1] - base_pix
    data_coords = inv.transform(positions)
    for (x_new, y_new), label in zip(data_coords, labels):
        ax.text(
            x_new,
            y_new,
            label,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
        )


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
        hard_pass = row.get("hard_pass")
        score = row.get("score")
        if isinstance(hard_pass, bool):
            stats["hard_pass"].append(hard_pass)
        if isinstance(score, (int, float)):
            stats["scores"].append(float(score))

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
                "lat_p95": float(np.mean(lats)) if lats else None,
                "score_mean": float(np.mean(scores)) if scores else None,
                "hard_pass_rate": (sum(passes) / len(passes)) if passes else None,
                "n_samples": len(scores) or len(lats) or len(costs),
            }
        )
    return aggregates


def display_label(model: str) -> str:
    """Format model name for plotting."""
    if model == "gpt-4o-2024-11-20":
        return "gpt-4o"
    if model == "gpt-5-chat-latest":
        return "gpt-5"
    if model.startswith("claude"):
        m = re.match(r"(.+)-\d{6,}$", model)
        if m:
            return m.group(1)
    return model


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
    colors = []
    labels = []
    for m in models:
        if m["cost_mean"] is None or m["score_mean"] is None:
            continue
        xs.append(m["cost_mean"])
        ys.append(m["score_mean"])
        colors.append(m["lat_mean"] or 0.0)
        labels.append(display_label(m["model"]))
    if not xs:
        plt.close(fig)
        return
    # Axis padding
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # For log scale, avoid non-positive; pad multiplicatively.
    xmin_adj = xmin * 0.8 if xmin > 0 else 1e-6
    xmax_adj = xmax * 1.2 if xmax > 0 else 1
    dy = (ymax - ymin) * 0.1 or max(abs(ymax), 1e-3) * 0.1
    ax.set_xlim(xmin_adj, xmax_adj)
    ax.set_ylim(ymin - dy, ymax + dy)
    sc = ax.scatter(xs, ys, s=200, c=colors, cmap="viridis", alpha=0.7, edgecolor="k")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Latency (mean seconds)")
    place_labels(ax, xs, ys, labels, dy * 0.4)
    ax.set_xlabel("Cost (mean, USD, log scale)")
    ax.set_ylabel("Quality (mean score)")
    ax.set_title("Cost vs Quality")
    ax.set_xscale("log")

    frontier = pareto_frontier(models, "cost_mean", "score_mean")
    if frontier:
        fx = [p["cost_mean"] for p in frontier]
        fy = [p["score_mean"] for p in frontier]
        ax.plot(fx, fy, "r--", label="Pareto frontier")
        ax.scatter(fx, fy, s=80, facecolors="none", edgecolors="red", linewidths=2)
        ax.legend(loc="upper left")

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
        lat = m["lat_mean"]
        if lat is None or m["score_mean"] is None:
            continue
        xs.append(lat)
        ys.append(m["score_mean"])
        cost = m["cost_mean"]
        colors.append(cost if cost is not None else 0.0)
        sizes.append(300)
        labels.append(display_label(m["model"]))
    if not xs:
        plt.close(fig)
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = (xmax - xmin) * 0.1 or max(abs(xmax), 1e-3) * 0.1
    dy = (ymax - ymin) * 0.1 or max(abs(ymax), 1e-3) * 0.1
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)
    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", alpha=0.7, edgecolor="k")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Cost (mean USD)")
    place_labels(ax, xs, ys, labels, dy * 0.4)
    ax.set_xlabel("Latency (mean seconds)")
    ax.set_ylabel("Quality (mean score)")
    ax.set_title("Latency vs Quality")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def scatter_cost_hardpass(models: List[Dict[str, Any]], out_path: Path) -> None:
    if not models:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = []
    ys = []
    labels = []
    for m in models:
        if m["cost_mean"] is None or m["hard_pass_rate"] is None:
            continue
        xs.append(m["cost_mean"])
        ys.append(m["hard_pass_rate"])
        labels.append(display_label(m["model"]))
    if not xs:
        plt.close(fig)
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xmin_adj = xmin * 0.8 if xmin > 0 else 1e-6
    xmax_adj = xmax * 1.2 if xmax > 0 else 1
    dy = (ymax - ymin) * 0.1 or max(abs(ymax), 1e-3) * 0.1
    ax.set_xlim(xmin_adj, xmax_adj)
    ax.set_ylim(0, 1.1)
    ax.scatter(xs, ys, s=200, alpha=0.7, edgecolor="k")
    place_labels(ax, xs, ys, labels, dy * 0.5)
    ax.set_xlabel("Cost (mean, USD, log scale)")
    ax.set_ylabel("Hard pass rate")
    ax.set_title("Cost vs Hard-pass rate")
    ax.set_xscale("log")

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
    parser.add_argument("--plot-only", action="store_true", help="Only plot existing responses/judgements; skip collect/judge.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens per model.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds.")
    parser.add_argument(
        "--responses-per-prompt",
        type=int,
        default=3,
        help="Number of responses to collect per model per prompt (default: 3).",
    )
    parser.add_argument(
        "--pricing",
        default="data/pricing_rates_usd_per_1m_tokens_2026-01-22.json",
        help="Path to pricing JSON (USD_per_1M_tokens).",
    )
    args = parser.parse_args()

    dotenv_path = None
    for candidate in ("API_configs/.env", ".env"):
        if Path(candidate).exists():
            dotenv_path = candidate
            break
    load_dotenv(dotenv_path=dotenv_path)

    plot_only = args.plot_only

    model_ids: List[str] = []
    if not plot_only:
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
    judge_provider = provider_for_model(judge_model) if judge_model else None
    if not plot_only:
        if not judge_model:
            raise SystemExit("JUDGE_MODEL_ID is required.")
        if judge_provider == "openai" and not openai_key:
            raise SystemExit("OPENAI_API_KEY is required for the chosen judge model.")
        if judge_provider == "anthropic" and not anthropic_key:
            raise SystemExit("ANTHROPIC_API_KEY is required for the chosen judge model.")

    dataset_path = Path(args.dataset)
    dataset_rows = load_dataset(dataset_path)
    if not plot_only:
        load_pricing(Path(args.pricing))
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

    # Fresh outputs per run unless plot-only
    if not plot_only:
        collect_out_path.parent.mkdir(parents=True, exist_ok=True)
        judgements_out_path.parent.mkdir(parents=True, exist_ok=True)
        if collect_out_path.exists():
            collect_out_path.unlink()
        if judgements_out_path.exists():
            judgements_out_path.unlink()

    now = datetime.now(tz=timezone.utc).isoformat()

    # -------- Collect --------
    collect_rows: List[Dict[str, Any]] = []
    if not plot_only:
        for entry in dataset_rows:
            prompt = entry.get("prompt", "")
            case_id = entry.get("case_id")
            for model in tqdm(model_ids, desc=f"Collecting {case_id}", unit="model", leave=False):
                provider = provider_for_model(model)
                for sample_idx in range(args.responses_per_prompt):
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
                            "sample_idx": sample_idx,
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
                            "sample_idx": sample_idx,
                        }
                    collect_rows.append(row)
        save_jsonl(collect_rows, collect_out_path)
    else:
        if collect_out_path.exists():
            collect_rows = list(iter_jsonl(collect_out_path))
        else:
            print(f"No responses file found at {collect_out_path}; cannot plot.")
            return

    # -------- Judge --------
    case_map = load_case_map(dataset_path)
    judgements_rows: List[Dict[str, Any]] = []
    if not plot_only:
        for resp_row in tqdm(iter_jsonl(collect_out_path), desc="Judging", unit="resp"):
            raw_error = None
            sample_idx = resp_row.get("sample_idx")
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
                "sample_idx": sample_idx,
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
    else:
        if judgements_out_path.exists():
            judgements_rows = list(iter_jsonl(judgements_out_path))
        else:
            print(f"No judgements file found at {judgements_out_path}; cannot plot.")
            return

    # -------- Plots --------
    plots_dir = Path("outputs") / dataset_stem / "plots"
    models = aggregate(collect_rows, judgements_rows)
    scatter_cost_quality(models, plots_dir / "cost_quality.png")
    scatter_latency_quality(models, plots_dir / "latency_quality.png")
    scatter_cost_hardpass(models, plots_dir / "cost_hardpass.png")


if __name__ == "__main__":
    main()
