"""L4 stage: GPT-5.4-nano bias judge over (scenario, cell) pairs.

For each pair, send the chat (PROD + cell, scenario, options) to
GPT-5.4-nano with the bias-judge prompt template (src.audit_prompts);
collect 3 votes per pair; emit outputs/audit_per_pair.jsonl.

Total: 30K API calls, ~$2-5 in OpenAI tier-5 credits.

The committed audit_per_pair.jsonl contains the canonical run; reviewers
verifying L1-L2 do not need to re-audit.
"""
from __future__ import annotations

import sys

print("L4: re-run the bias-judge audit. 30K GPT-5.4-nano calls, ~$5.")
print()
print("Committed outputs/audit_per_pair.jsonl is the canonical run.")
print()
print("To re-run: adapt experiments/26-bias-judge/scripts/01_audit_basis.py")
print("from https://github.com/jeqcho/in-context-steganography")
sys.exit(0)
