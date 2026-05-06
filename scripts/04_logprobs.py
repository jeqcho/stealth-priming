"""L3 stage: re-derive log-odds and MDCL from Gemma 4 31B-it forward
passes at the rec-prefix position.

Pipeline (one-shot):
1. For each (scenario, cell) pair, build the chat
     system: PROD_SYSTEM_PROMPT_22 + cell_text
     user:   scenario.prompt
     assistant prefix: "My recommendation: "
   Forward-pass Gemma; record log P(option_a_string) and
   log P(option_b_string) at the prefill end.
2. For each scenario, compute the same log-probs under the BASELINE
   prefix (PROD only, no cell).
3. Compute delta_sum = log P(A) - log P(B), and te_a_sum / te_b_sum
   as MDCL signals (basis-induced absolute log-prob shift).
4. Join with rate_a / rate_b from the fetched judged rollouts.
5. Emit outputs/calibration_rec.csv with the same schema as the
   committed file.

Compute: ~30 min on 1× H100/H200 with vLLM 0.20.1+; requires HF_TOKEN
for gated model access.

This script is adapted from
  experiments/23-logits/scripts/01_compute_logprobs.py +
  experiments/23-logits/scripts/02_join_with_rollouts.py
in the dev repo at
  https://github.com/jeqcho/in-context-steganography
"""
from __future__ import annotations

import sys

print("L3: re-derive calibration_rec.csv from Gemma forward passes.")
print()
print("Skeleton intentionally not implemented in this release; the")
print("vLLM-based logprob runner from the dev repo is the canonical")
print("implementation. See experiments/23-logits/scripts/ in")
print("https://github.com/jeqcho/in-context-steganography")
print()
print("For reviewers verifying L1-L2: use the committed")
print("outputs/calibration_rec.csv as-is (no GPU needed).")
sys.exit(0)
