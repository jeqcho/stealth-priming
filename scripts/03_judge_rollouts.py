"""Stage 4: GPT-5.4-nano judges every rollout.

Walks a rollouts JSONL (or a baselines JSONL), classifies the `answer`
field (already thinking-stripped) into option_a / option_b / ambiguous
via judge_v2.classify_response, and writes a parallel `*.judged.jsonl`.

Concurrency: ThreadPoolExecutor with --judge-workers parallel API calls.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))


import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from judge_v2 import classify_response

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

JUDGE_MODEL = "openai/gpt-5.4-nano"
JUDGE_REASONING = "none"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-file", required=True,
                   help="path to a JSONL with rows containing "
                        "{entry_id, options, prompt, answer}")
    p.add_argument("--out-file", default=None,
                   help="defaults to <in-file>.judged.jsonl")
    p.add_argument("--judge-workers", type=int, default=256,
                   help="OpenAI tier 5 -> push high; gpt-5.4-nano can take "
                        "thousands of req/min")
    p.add_argument("--resume", action="store_true",
                   help="if set, skip rows already judged in --out-file")
    return p.parse_args()


def judge_one(row: dict) -> dict:
    options = row["options"]
    label = classify_response(
        question=row["prompt"],
        option_a=options[0],
        option_b=options[1],
        response=row["answer"],
        model=JUDGE_MODEL,
        reasoning_effort=JUDGE_REASONING,
    )
    out = dict(row)
    out["judge_label"] = label  # option_a / option_b / ambiguous
    return out


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_file)
    out_path = Path(args.out_file) if args.out_file else \
        in_path.with_suffix(".judged.jsonl")

    rows = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"[judge] {in_path}: {len(rows)} rows to judge")

    # Resume: skip rows whose (entry_id, cell_id?, sample_idx) already
    # appear in the output file.
    seen: set[tuple] = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                key = _row_key(d)
                seen.add(key)
        print(f"[judge] resume: {len(seen)} rows already judged")

    todo = [r for r in rows if _row_key(r) not in seen]
    print(f"[judge] {len(todo)} rows to do this run")

    t0 = time.time()
    out_mode = "a" if args.resume and out_path.exists() else "w"
    with open(out_path, out_mode) as fout:
        with ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
            futs = {ex.submit(judge_one, r): i for i, r in enumerate(todo)}
            done = 0
            for fut in as_completed(futs):
                try:
                    out = fut.result()
                except Exception as exc:
                    print(f"[judge] error: {exc!r}")
                    continue
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                done += 1
                if done % 1000 == 0 or done == len(todo):
                    dt = time.time() - t0
                    rate = done / max(dt, 1e-3)
                    eta = (len(todo) - done) / max(rate, 1e-3)
                    print(f"[judge] {done}/{len(todo)} ({rate:.1f}/s, "
                          f"eta {eta:.0f}s)")
                fout.flush()

    print(f"[judge] wrote {out_path}")


def _row_key(row: dict) -> tuple:
    return (row.get("entry_id"), row.get("cell_id"), row.get("sample_idx"))


if __name__ == "__main__":
    main()
