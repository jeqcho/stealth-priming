"""L4 stage 3: GPT-5.4-nano classifies each rollout into option_a /
option_b / ambiguous.

Reads the raw rollouts from stage 02 (~250K records), invokes
src.judge_v2.classify_response in a thread pool, writes a parallel
.judged.jsonl. Cost: ~$5-10 in OpenAI tier-5 credits.

The committed HF dataset includes the judged version; reviewers do not
need to re-judge.
"""
from __future__ import annotations

import sys

print("L4 judge stage. Reads raw rollouts and adds judge_label per row.")
print()
print("For reviewers: `make fetch-rollouts` already grabs the judged")
print("version from HuggingFace.")
print()
print("To re-run: adapt experiments/22-neutral-prompts/scripts/04_judge_rollouts.py")
print("from https://github.com/jeqcho/in-context-steganography")
sys.exit(0)
