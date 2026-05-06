"""L4 stage 1: Gemma 4 31B-it generates the 100-cell neutral basis.

The committed `data/sys_prompts.json` is the basis used in the paper.
This script regenerates it from the same Gemma model + the dimensions
grid in `src/helpers.py` (5 personas × 5 tones × 4 emphases). Output
is non-deterministic (T=0.6 sampling), so a fresh run will produce a
different but qualitatively equivalent basis.

Compute: ~30 min on a 4× H200 box with vLLM 0.20.1+ (one prompt per
cell, batched). Requires HF_TOKEN for gated model access.

This script is adapted from
  experiments/22-neutral-prompts/scripts/01_generate_sysprompts.py
in the dev repo:
  https://github.com/jeqcho/in-context-steganography
The paper's basis is the result of the run committed in
data/sys_prompts.json; reviewers verifying L1-L3 do NOT need to re-run
this stage.
"""
from __future__ import annotations

import sys

print("This script regenerates the 100-cell neutral basis from scratch.")
print("It requires Gemma 4 31B-it on GPU + HF_TOKEN for gated access.")
print()
print("For the canonical basis used in the paper, see data/sys_prompts.json.")
print()
print("To re-run: adapt experiments/22-neutral-prompts/scripts/01_*.py from")
print("  https://github.com/jeqcho/in-context-steganography")
print()
print("Skeleton intentionally not implemented; Gemma authoring is")
print("non-deterministic and the committed basis is the canonical one.")
sys.exit(0)
