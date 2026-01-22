#!/usr/bin/env python3
import argparse
import json
import textwrap
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Load newline-delimited JSON records from the given path."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_wrapped(title: str, text: str, width: int = 96, indent: int = 2) -> str:
    wrapped = textwrap.fill(
        text,
        width=width - indent,
        subsequent_indent=" " * indent,
    )
    return f"{title}:\n{textwrap.indent(wrapped, ' ' * indent)}"


def format_list(title: str, items: list[str], width: int = 96, indent: int = 2) -> str:
    if not items:
        return ""
    lines: list[str] = [f"{title}:"]
    for item in items:
        wrapped = textwrap.fill(
            item,
            width=width - indent - 2,
            subsequent_indent=" " * (indent + 2),
        )
        lines.append(" " * indent + f"- {wrapped}")
    return "\n".join(lines)


def print_entry(entry: dict, index: int, total: int, width: int = 96) -> None:
    header = f"=== Entry {index}/{total} — {entry.get('case_id', '<no id>')} ==="
    print(header)
    print(format_wrapped("Prompt", entry.get("prompt", ""), width=width))
    print()
    print(format_wrapped("Expected", entry.get("expected", ""), width=width))

    rubric = entry.get("rubric", {})
    if rubric:
        hard = rubric.get("hard_requirements") or []
        soft = rubric.get("soft_requirements") or []
        penalties = rubric.get("penalties") or []
        for block in (
            format_list("Hard requirements", hard, width=width),
            format_list("Soft requirements", soft, width=width),
            format_list("Penalties", penalties, width=width),
        ):
            if block:
                print()
                print(block)

    tags = entry.get("tags")
    if tags:
        print()
        print(format_list("Tags", tags, width=width))

    if index != total:
        print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretty-print JSONL dataset entries.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="data/dataset_simple.jsonl",
        help="Path to the JSONL file to read.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    entries = load_jsonl(path)
    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        print_entry(entry, idx, total)


if __name__ == "__main__":
    main()
