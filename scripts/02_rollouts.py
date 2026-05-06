"""L4 stage 2: 25 rollouts per (scenario, cell) pair on Gemma 4 31B-it.

10K (scen, cell) pairs × 25 samples = 250K rollouts.
Compute: ~4-8 hr on a 4× H200 box with vLLM.

For reproducibility verification, prefer fetching the canonical
rollouts (judged) from the HF dataset:

    make fetch-rollouts

The committed dataset
  https://huggingface.co/datasets/jeqcho/stealth-priming-rollouts
contains gemma_orig_thinkoff.jsonl (these rollouts), gemma_flipped.jsonl
(exp 28 ablation), qwen_orig.jsonl (exp 31 replication), and
audit_per_pair.jsonl (judge votes).

This script is adapted from
  experiments/22-neutral-prompts/scripts/03_run_rollouts.py +
  experiments/22-neutral-prompts/src/vllm_runner.py +
  experiments/22-neutral-prompts/src/vllm_worker.py
in the dev repo at
  https://github.com/jeqcho/in-context-steganography
"""
from __future__ import annotations

import sys

print("L4 rollout regeneration. ~4-8 hr GPU + non-trivial compute.")
print()
print("For reviewers verifying results: prefer `make fetch-rollouts`")
print("to download the canonical judged rollouts from HuggingFace.")
print()
print("To re-run: adapt experiments/22-neutral-prompts/scripts/03_*.py")
print("from https://github.com/jeqcho/in-context-steganography")
sys.exit(0)
