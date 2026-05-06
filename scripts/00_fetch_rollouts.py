"""L3 prerequisite: download the judged-rollout dataset.

The 4 JSONL files (~1.7 GB total) are hosted externally; the dataset
URL is withheld for double-blind review and will be linked on
publication.

Place the following files under  artefacts/rollouts/  in the project
root before running L3+ stages (04_logprobs.py, 04b_join_logprobs.py):

  gemma_orig_thinkoff.jsonl   (~793 MB; 250K Gemma 4 31B-it rollouts)
  gemma_flipped.jsonl         (~430 MB; 124K rollouts, flipped option order)
  qwen_orig.jsonl             (~544 MB; 169K Qwen3.6-27B replication rollouts)
  audit_per_pair.jsonl        (~2.5 MB; 10K bias-judge votes)

L1 and L2 reproductions reproduce the headline numbers from the
artefacts already bundled in outputs/ and need no external download.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "artefacts" / "rollouts"

REQUIRED = [
    "gemma_orig_thinkoff.jsonl",
    "gemma_flipped.jsonl",
    "qwen_orig.jsonl",
    "audit_per_pair.jsonl",
]


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    missing = [f for f in REQUIRED if not (TARGET / f).exists()]
    if not missing:
        print(f"[fetch] all {len(REQUIRED)} rollout files present in {TARGET}")
        return
    print("[fetch] dataset URL withheld for double-blind review.")
    print(f"[fetch] place the following files under {TARGET} :")
    for f in missing:
        print(f"          {f}")
    print("[fetch] then re-run L3+ stages.")
    sys.exit(1)


if __name__ == "__main__":
    main()
