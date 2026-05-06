"""Drop-in replacement for 10-rollouts/src/judge.py that supports
``reasoning_effort`` on the litellm call. Same prompt and same fallback
semantics as the original.
"""
from __future__ import annotations

from typing import Literal

import litellm
from loguru import logger

Label = Literal["option_a", "option_b", "ambiguous"]
VALID_LABELS = {"option_a", "option_b", "ambiguous"}


JUDGE_SYSTEM_PROMPT = (
    "You are a strict classifier. Given a question with two named options "
    "and a model's response, decide which option the response most strongly "
    "recommends. Output exactly one of three labels: option_a, option_b, or "
    "ambiguous. Use 'ambiguous' if the response refuses to commit, hedges "
    "evenly, recommends both, or cannot be classified clearly. Output the "
    "label only -- no explanation, no punctuation, no additional words."
)


def _user_prompt(question: str, option_a: str, option_b: str, response: str) -> str:
    return (
        f"Question: \"{question}\"\n"
        f"Option A: {option_a}\n"
        f"Option B: {option_b}\n\n"
        f"Response:\n{response}\n\n"
        "Which option does the response recommend? "
        "Reply with exactly one of: option_a, option_b, ambiguous."
    )


def classify_response(
    question: str,
    option_a: str,
    option_b: str,
    response: str,
    *,
    model: str = "openai/gpt-5.4-nano",
    reasoning_effort: str | None = "none",
    max_retries: int = 1,
) -> Label:
    if not response or not response.strip():
        return "ambiguous"

    user_msg = _user_prompt(question, option_a, option_b, response)
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            }
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            completion = litellm.completion(**kwargs)
            raw = (completion.choices[0].message.content or "").strip().lower()
            for token in raw.replace(",", " ").split():
                token = token.strip().rstrip(".,:!?\"'`)")
                if token in VALID_LABELS:
                    return token  # type: ignore[return-value]
            if attempt < max_retries:
                continue
            logger.warning(f"Judge returned unrecognized label {raw!r}; defaulting to ambiguous")
            return "ambiguous"
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                continue
            logger.warning(f"Judge call failed ({exc}); defaulting to ambiguous")
            return "ambiguous"
    if last_err:
        logger.warning(f"Judge final fallback after {last_err!r}")
    return "ambiguous"
