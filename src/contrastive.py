"""Helpers for exp 33 — contrastive rollout persona vectors."""
from __future__ import annotations

import re


def find_response_range(rendered_chat: str, response_text: str, tokenizer):
    """Return (start, end) token range covering response_text in rendered_chat.

    Uses character offset_mapping for robustness to BPE merges across
    boundaries.
    """
    char_start = rendered_chat.index(response_text)
    char_end = char_start + len(response_text)
    enc = tokenizer(rendered_chat, add_special_tokens=False,
                    return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    rng_start = None
    rng_end = None
    for tok_idx, (s, e) in enumerate(offsets):
        if s >= char_start and rng_start is None:
            rng_start = tok_idx
        if e <= char_end:
            rng_end = tok_idx + 1
    if rng_start is None or rng_end is None or rng_end <= rng_start:
        raise RuntimeError(
            f"Failed to locate response in rendered chat. "
            f"chars=[{char_start},{char_end})")
    return rng_start, rng_end
