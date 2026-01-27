from __future__ import annotations

import json
import os
import select
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.collect_and_judge import aggregate, display_label, iter_jsonl, pareto_frontier


st.set_page_config(
    page_title="LLM Selection Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_templates() -> Dict[str, Dict]:
    templates: Dict[str, Dict] = {}
    dataset_path = Path("data/dataset_simple.jsonl")
    if not dataset_path.exists():
        return templates
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            cid = row.get("case_id")
            if cid:
                templates[cid] = row
        except json.JSONDecodeError:
            continue
    return templates


TEMPLATES = load_templates()


def ensure_session_state() -> None:
    defaults = {
        "case_id": "custom_case",
        "prompt_text": "",
        "expected_text": "",
        "hard_requirements": "",
        "soft_requirements": "",
        "penalties": "",
        "tags": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


ensure_session_state()


def apply_template(selected: str) -> None:
    if selected == "blank":
        st.session_state.update(
            case_id="custom_case",
            prompt_text="",
            expected_text="",
            hard_requirements="",
            soft_requirements="",
            penalties="",
            tags="",
        )
        return
    template = TEMPLATES.get(selected)
    if not template:
        return
    rubric = template.get("rubric", {})
    st.session_state.update(
        case_id=template.get("case_id", "custom_case"),
        prompt_text=template.get("prompt", ""),
        expected_text=template.get("expected", ""),
        hard_requirements="\n".join(rubric.get("hard_requirements", [])),
        soft_requirements="\n".join(rubric.get("soft_requirements", [])),
        penalties="\n".join(rubric.get("penalties", [])),
        tags=",".join(template.get("tags", [])),
    )


def parse_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_tags(text: str) -> List[str]:
    tags: List[str] = []
    for part in text.replace("\n", ",").split(","):
        tag = part.strip()
        if tag:
            tags.append(tag)
    return tags


def build_case_payload(case_id: str) -> Dict:
    return {
        "case_id": case_id,
        "prompt": st.session_state["prompt_text"].strip(),
        "expected": st.session_state["expected_text"].strip(),
        "rubric": {
            "hard_requirements": parse_lines(st.session_state["hard_requirements"]),
            "soft_requirements": parse_lines(st.session_state["soft_requirements"]),
            "penalties": parse_lines(st.session_state["penalties"]),
        },
        "tags": parse_tags(st.session_state["tags"]),
    }


def run_pipeline(
    dataset_path: Path,
    dataset_stem: str,
    case_id: str,
    responses_per_prompt: int,
    log_placeholder,
    stage_callback,
) -> tuple[int, str]:
    cmd = [
        "python",
        "scripts/collect_and_judge.py",
        "--dataset",
        str(dataset_path),
        "--case-id",
        case_id,
        "--responses-per-prompt",
        str(responses_per_prompt),
    ]
    env = os.environ.copy()
    # Hide tqdm progress bars from stdout
    env["TQDM_DISABLE"] = "1"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    judgements_path = Path("outputs") / dataset_stem / f"{dataset_stem}_judgements.jsonl"
    log_lines: List[str] = []
    stage_callback("Collecting responses...")
    judging_notified = False

    if process.stdout:
        while True:
            ready, _, _ = select.select([process.stdout], [], [], 0.2)
            if ready:
                raw = process.stdout.readline()
                if raw:
                    line = raw.rstrip("\n")
                    if line:
                        log_lines.append(line)
                        log_placeholder.code("\n".join(log_lines[-300:]) or "(no output)")
            if not judging_notified and judgements_path.exists():
                judging_notified = True
                stage_callback("Collecting judgements...")
            if process.poll() is not None:
                # Drain any remaining output
                remaining = process.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        if line:
                            log_lines.append(line)
                break
            time.sleep(0.05)

    # If judging file appeared but callback not yet fired
    if not judging_notified and judgements_path.exists():
        stage_callback("Collecting judgements...")

    returncode = process.wait()
    return returncode, "\n".join(log_lines)


def load_models_data(dataset_stem: str) -> Optional[List[Dict]]:
    base_dir = Path("outputs") / dataset_stem
    responses_path = base_dir / f"{dataset_stem}_responses.jsonl"
    judgements_path = base_dir / f"{dataset_stem}_judgements.jsonl"
    if not responses_path.exists() or not judgements_path.exists():
        return None
    collect_rows = list(iter_jsonl(responses_path))
    judgement_rows = list(iter_jsonl(judgements_path))
    return aggregate(collect_rows, judgement_rows)


def interactive_scatter(df: pd.DataFrame, x: str, y: str, color: str, label_field: str, title: str, log_x: bool = False) -> px.scatter:
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        hover_name=label_field,
        color_continuous_scale="Viridis",
    )
    fig.update_traces(
        mode="markers",
        marker=dict(size=16, line=dict(width=1, color="black"), opacity=0.85),
        hovertemplate="<b>%{hovertext}</b><br>"
        + f"{x}: %{x}<br>{y}: %{y}<br>{color}: %{{marker.color}}<extra></extra>",
        selected=dict(marker=dict(opacity=1, size=18)),
        unselected=dict(marker=dict(opacity=0.2)),
    )
    fig.update_layout(title=title, hovermode="closest", legend_title=color)
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def plot_cost_quality(models: List[Dict]) -> Optional[px.scatter]:
    rows = [
        {
            "cost_mean": m["cost_mean"],
            "score_mean": m["score_mean"],
            "lat_mean": m["lat_mean"],
            "label": display_label(m["model"]),
        }
        for m in models
        if m.get("cost_mean") is not None and m.get("score_mean") is not None
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = interactive_scatter(df, "cost_mean", "score_mean", "lat_mean", "label", "Cost vs Quality", log_x=True)
    frontier = pareto_frontier(models, "cost_mean", "score_mean")
    if frontier:
        fx = [p["cost_mean"] for p in frontier]
        fy = [p["score_mean"] for p in frontier]
        flabels = [display_label(p["model"]) for p in frontier]
        fig.add_scatter(
            x=fx,
            y=fy,
            mode="lines+markers",
            name="Pareto frontier",
            line=dict(color="red", dash="dash"),
            marker=dict(size=10, color="red", symbol="circle-open"),
            hovertext=flabels,
            hovertemplate="<b>%{hovertext}</b><br>Cost: %{x}<br>Quality: %{y}<extra></extra>",
            showlegend=True,
        )
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
    return fig


def plot_latency_quality(models: List[Dict]) -> Optional[px.scatter]:
    rows = [
        {
            "latency_mean": m["lat_mean"],
            "score_mean": m["score_mean"],
            "cost_mean": m["cost_mean"],
            "label": display_label(m["model"]),
        }
        for m in models
        if m.get("lat_mean") is not None and m.get("score_mean") is not None
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = interactive_scatter(df, "latency_mean", "score_mean", "cost_mean", "label", "Latency vs Quality")
    frontier = pareto_frontier(models, "lat_mean", "score_mean")
    if frontier:
        fx = [p["lat_mean"] for p in frontier]
        fy = [p["score_mean"] for p in frontier]
        flabels = [display_label(p["model"]) for p in frontier]
        fig.add_scatter(
            x=fx,
            y=fy,
            mode="lines+markers",
            name="Pareto frontier",
            line=dict(color="red", dash="dash"),
            marker=dict(size=10, color="red", symbol="circle-open"),
            hovertext=flabels,
            hovertemplate="<b>%{hovertext}</b><br>Latency: %{x}<br>Quality: %{y}<extra></extra>",
            showlegend=True,
        )
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
    return fig


def plot_cost_hardpass(models: List[Dict]) -> Optional[px.scatter]:
    rows = [
        {
            "cost_mean": m["cost_mean"],
            "hard_pass_rate": m["hard_pass_rate"],
            "label": display_label(m["model"]),
        }
        for m in models
        if m.get("cost_mean") is not None and m.get("hard_pass_rate") is not None
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.scatter(df, x="cost_mean", y="hard_pass_rate", hover_name="label")
    fig.update_traces(
        mode="markers",
        marker=dict(size=16, line=dict(width=1, color="black"), opacity=0.85),
        hovertemplate="<b>%{hovertext}</b><br>Cost: %{x}<br>Hard pass rate: %{y:.2f}<extra></extra>",
        selected=dict(marker=dict(opacity=1, size=18)),
        unselected=dict(marker=dict(opacity=0.2)),
    )
    fig.update_layout(title="Cost vs Hard-pass rate", hovermode="closest")
    fig.update_xaxes(type="log", title="Cost (mean USD)")
    fig.update_yaxes(range=[0, 1.1], title="Hard-pass rate")
    return fig


st.title("LLM Selection & Judging")
st.caption("Build a prompt/rubric, collect responses, judge, and visualize.")

with st.sidebar:
    st.subheader("Template")
    template_choice = st.selectbox("Prefill fields with", ["biz_001", "it_001", "blank"])
    st.button("Apply template", on_click=apply_template, args=(template_choice,))
    st.subheader("Run Options")
    num_responses = st.selectbox("Responses per model", options=[1, 2, 3, 4, 5], index=2, help="Passed to --responses-per-prompt.")
    st.info("This takes minutes to run. The model responses are collected and then judged. More responses will need more time.")

col_left, col_right = st.columns(2)

with col_left:
    st.text_input("Case ID", key="case_id", help="Used for --case-id and dataset row.")
    st.text_area("Prompt", key="prompt_text", height=220)
    st.text_area("Expected", key="expected_text", height=180)

with col_right:
    st.text_area("Hard requirements (one per line)", key="hard_requirements", height=160)
    st.text_area("Soft requirements (one per line)", key="soft_requirements", height=120)
    st.text_area("Penalties (one per line)", key="penalties", height=120)
    st.text_input("Tags (comma or newline separated)", key="tags")

run_clicked = st.button("Run collect & judge", type="primary", use_container_width=True)

if run_clicked:
    case_id = st.session_state["case_id"].strip() or f"custom_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    payload = build_case_payload(case_id)
    if not payload["prompt"]:
        st.error("Prompt is required.")
    elif not payload["expected"]:
        st.error("Expected text is required.")
    elif not payload["rubric"]["hard_requirements"]:
        st.error("At least one hard requirement is required.")
    else:
        dataset_dir = Path("outputs/streamlit_runs")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_stem = f"{case_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        dataset_path = dataset_dir / f"{dataset_stem}.jsonl"
        dataset_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

        status_box = st.empty()
        status_box.info("Collecting responses...")
        log_box = st.empty()
        return_code, log_text = run_pipeline(
            dataset_path,
            dataset_stem,
            case_id,
            num_responses,
            log_box,
            stage_callback=lambda label: status_box.info(label),
        )
        if return_code != 0:
            status_box.error("Pipeline failed")
            st.error("Pipeline failed. Check log below.")
            st.code(log_text or "(empty)")
        else:
            status_box.success("Pipeline complete")
            st.success("Run complete. Plots below use fresh outputs.")
            with st.expander("Run log"):
                st.code(log_text or "(empty)")

            models = load_models_data(dataset_stem)
            if not models:
                st.warning("No model data found; nothing to plot.")
            else:
                plots_dir = Path("outputs") / dataset_stem / "plots"
                col_a, col_b, col_c = st.columns(3)

                fig_cost_quality = plot_cost_quality(models)
                fig_latency_quality = plot_latency_quality(models)
                fig_cost_hardpass = plot_cost_hardpass(models)

                with col_a:
                    if fig_cost_quality:
                        st.plotly_chart(fig_cost_quality, use_container_width=True)

                with col_b:
                    if fig_latency_quality:
                        st.plotly_chart(fig_latency_quality, use_container_width=True)

                with col_c:
                    if fig_cost_hardpass:
                        st.plotly_chart(fig_cost_hardpass, use_container_width=True)

            st.caption(
                "Hover a point to see its label; click to focus and dim others. "
                "Plots are recomputed each run from the newest responses/judgements."
            )
