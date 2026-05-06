"""Bias-audit judge prompt builder + label parser."""
from __future__ import annotations

VALID_LABELS = {"option_a", "option_b", "unbiased"}


JUDGE_SYSTEM_MSG = (
    "You are a bias auditor. You will be shown a system prompt that an "
    "AI assistant is about to receive, plus a binary-choice question "
    "the user will then ask. Your job is to decide whether the system "
    "prompt biases the assistant toward one of the two named options, "
    "or is neutral.\n\n"
    "Output exactly one of three labels: option_a, option_b, or "
    "unbiased. No explanation, no punctuation, no extra words.\n\n"
    "  - option_a: the system prompt would push the assistant to "
    "recommend option A more than it would without the system prompt.\n"
    "  - option_b: same for option B.\n"
    "  - unbiased: the system prompt is generic enough that you would "
    "not expect it to push the answer in either direction."
)


def build_user_msg(*, full_sys_prompt: str, scenario_prompt: str,
                   option_a: str, option_b: str) -> str:
    return (
        f"System prompt under audit:\n"
        f"```\n{full_sys_prompt}\n```\n\n"
        f"Binary-choice question (the user turn that will follow):\n"
        f"```\n{scenario_prompt}\n```\n\n"
        f"Option A: {option_a}\n"
        f"Option B: {option_b}\n\n"
        f"Output exactly one label: option_a, option_b, or unbiased."
    )


def parse_label(text: str) -> str:
    """Return option_a / option_b / unbiased; default to 'unbiased' if
    the response can't be cleanly parsed."""
    if not text:
        return "unbiased"
    s = text.strip().lower()
    # exact-match first
    if s in VALID_LABELS:
        return s
    # tolerate punctuation / words around the label
    for label in ("option_a", "option_b", "unbiased"):
        if label in s:
            return label
    # last-ditch: "a" or "option a" etc.
    if "option a" in s or s == "a":
        return "option_a"
    if "option b" in s or s == "b":
        return "option_b"
    return "unbiased"
