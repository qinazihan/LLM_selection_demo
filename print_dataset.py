#!/usr/bin/env python3
import argparse
import json
import re
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
    """Wrap text while preserving blank lines and bullets for readability."""
    base_wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=False,
    )
    bullet_wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=" " * indent + "- ",
        subsequent_indent=" " * (indent + 2),
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=False,
    )

    lines: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        combined = " ".join(part.strip() for part in paragraph if part.strip())
        if combined:
            lines.append(base_wrapper.fill(combined))
        paragraph.clear()

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            flush_paragraph()
            lines.append("")
            continue

        if stripped.startswith(("- ", "* ")):
            flush_paragraph()
            content = stripped[2:].lstrip()
            lines.append(bullet_wrapper.fill(content))
            continue

        number_match = re.match(r"(\d+[\.\)])\s+(.*)", stripped)
        if number_match:
            flush_paragraph()
            prefix, content = number_match.groups()
            prefix_len = len(prefix) + 1
            numbered_wrapper = textwrap.TextWrapper(
                width=width,
                initial_indent=" " * indent + prefix + " ",
                subsequent_indent=" " * (indent + prefix_len),
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=False,
            )
            lines.append(numbered_wrapper.fill(content))
            continue

        paragraph.append(stripped)

    flush_paragraph()

    return f"{title}:\n" + "\n".join(lines)


def format_expected(title: str, text: str, width: int = 96, indent: int = 2) -> str:
    """Render expected answers with bullets when possible for quick scanning."""
    if not text.strip():
        return f"{title}:"

    base_wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=False,
    )
    bullet_wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=" " * indent + "- ",
        subsequent_indent=" " * (indent + 2),
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=False,
    )

    intro = ""
    remainder = text.strip()

    colon_split = re.split(r":\s+", remainder, maxsplit=1)
    if len(colon_split) == 2:
        intro, remainder = colon_split
    else:
        intro = remainder
        remainder = ""

    bullet_parts: list[str] = []
    final_notes: list[str] = []

    if remainder:
        bullet_parts = re.split(r";\s+", remainder)
    else:
        bullet_parts = []

    processed_bullets: list[str] = []
    for part in bullet_parts:
        trimmed = part.strip(" ;")
        if not trimmed:
            continue
        note_match = re.search(r"(It must not.*)", trimmed)
        if note_match and note_match.start() != 0:
            main = trimmed[: note_match.start()].strip(" .")
            note = note_match.group(1).strip()
            if main:
                processed_bullets.append(main)
            if note:
                final_notes.append(note)
        else:
            processed_bullets.append(trimmed)

    lines: list[str] = [f"{title}:"]
    if intro:
        lines.append(base_wrapper.fill(intro))

    if processed_bullets:
        for bullet in processed_bullets:
            lines.append(bullet_wrapper.fill(bullet))

    if not processed_bullets and intro:
        # Fall back to plain wrapping when no bullet structure is detected.
        lines = [f"{title}:", base_wrapper.fill(intro)]

    for note in final_notes:
        lines.append(base_wrapper.fill(note))

    return "\n".join(lines)


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
    print(format_expected("Expected", entry.get("expected", ""), width=width))

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
