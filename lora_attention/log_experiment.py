#!/usr/bin/env python3
"""Append a structured experiment entry to the MoELoRA knowledge base."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kb_path", required=True)
    p.add_argument("--jsonl_path", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--job_id", required=True)
    p.add_argument("--run_tag", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--synthesis", required=True)
    p.add_argument("--prompt_type", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--verdict", required=True)
    p.add_argument("--notes", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    record = {
        "timestamp": now,
        "stage": args.stage,
        "job_id": args.job_id,
        "run_tag": args.run_tag,
        "checkpoint": args.checkpoint,
        "synthesis": args.synthesis,
        "prompt_type": args.prompt_type,
        "result": args.result,
        "verdict": args.verdict,
        "notes": args.notes,
    }

    jsonl_path = Path(args.jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")

    kb_path = Path(args.kb_path)
    text = kb_path.read_text(encoding="utf-8")
    table_header = "| Date | Job | Stage | Checkpoint | Synthesis | Prompt type | Result | Verdict | Notes |\n"
    divider = "|------|-----|-------|------------|-----------|-------------|--------|---------|-------|\n"
    if table_header not in text or divider not in text:
        raise SystemExit(f"Could not find run ledger table in {kb_path}")

    row = (
        f"| {now[:10]} | {args.job_id} | {args.stage} | `{args.checkpoint}` | "
        f"{args.synthesis} | {args.prompt_type} | {args.result} | {args.verdict} | "
        f"{args.notes or '-'} |"
    )

    marker = divider
    new_text = text.replace(marker, marker + row + "\n", 1)
    kb_path.write_text(new_text, encoding="utf-8")

    print(f"updated {kb_path}")
    print(f"appended {jsonl_path}")


if __name__ == "__main__":
    main()
