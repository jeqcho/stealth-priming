"""Stage 1: judge each (scenario, sys_prompt) pair with GPT-5.4-nano,
3 votes per pair. Write outputs/judgements_<ts>/all.jsonl with one row
per (entry_id, cell_id, vote_idx, label, raw, latency).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))


import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import litellm

from audit_prompts import JUDGE_SYSTEM_MSG, build_user_msg, parse_label
# PROD prefix from exp 22's helpers.py (different module name now)
from helpers import PROD_SYSTEM_PROMPT_22

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = PROJECT_ROOT / "data" / "scenarios.json"
SYSPROMPTS_PATH = PROJECT_ROOT / "data" / "sys_prompts.json"
OUTPUTS = PROJECT_ROOT / "outputs"

JUDGE_MODEL = "openai/gpt-5.4-nano"
JUDGE_REASONING = "none"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-votes", type=int, default=3)
    p.add_argument("--workers", type=int, default=256)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="if set, only judge the first N pairs (for smoke)")
    return p.parse_args()


def judge_one(*, full_sys: str, scen_prompt: str, opt_a: str, opt_b: str,
              n_votes: int) -> tuple[list[str], list[str]]:
    """Returns (parsed_labels, raw_texts) of length n_votes each.

    Tries n=n_votes in one call first; if that yields fewer than n_votes
    choices, tops up with single-shot calls until we have n_votes.
    """
    user_msg = build_user_msg(
        full_sys_prompt=full_sys, scenario_prompt=scen_prompt,
        option_a=opt_a, option_b=opt_b,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]
    raw_texts: list[str] = []
    # First: one call with n=n_votes.
    try:
        r = litellm.completion(
            model=JUDGE_MODEL, messages=messages,
            reasoning_effort=JUDGE_REASONING,
            temperature=1.0, n=n_votes, max_tokens=12,
        )
        for ch in r.choices:
            raw_texts.append((ch.message.content or "").strip())
    except Exception as exc:  # pragma: no cover -- network
        raw_texts.append(f"<error1: {exc!r}>")
    # Top up with single-shot calls if short.
    while len(raw_texts) < n_votes:
        try:
            r = litellm.completion(
                model=JUDGE_MODEL, messages=messages,
                reasoning_effort=JUDGE_REASONING,
                temperature=1.0, n=1, max_tokens=12,
            )
            raw_texts.append((r.choices[0].message.content or "").strip())
        except Exception as exc:  # pragma: no cover
            raw_texts.append(f"<error2: {exc!r}>")
    raw_texts = raw_texts[:n_votes]
    labels = [parse_label(t) for t in raw_texts]
    return labels, raw_texts


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    scenarios = json.loads(SCENARIOS_PATH.read_text())
    sys_prompts = json.loads(SYSPROMPTS_PATH.read_text())["prompts"]
    print(f"[exp26] {len(scenarios)} scenarios x {len(sys_prompts)} sys prompts "
          f"= {len(scenarios) * len(sys_prompts)} pairs, "
          f"{args.n_votes} votes each = "
          f"{len(scenarios) * len(sys_prompts) * args.n_votes} judge calls")

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUTS / f"judgements_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[exp26] out_dir={out_dir}")

    # Build the work list — preserve ordering by (scenario_idx, sys_idx)
    work: list[dict] = []
    for s in scenarios:
        for sp in sys_prompts:
            full_sys = PROD_SYSTEM_PROMPT_22 + "\n\n" + sp["text"]
            work.append({
                "entry_id": s["entry_id"],
                "domain": s["domain"],
                "cell_id": sp["cell_id"],
                "persona_idx": sp["persona_idx"],
                "tone_idx": sp["tone_idx"],
                "emphasis_idx": sp["emphasis_idx"],
                "option_a": s["options"][0],
                "option_b": s["options"][1],
                "scen_prompt": s["prompt"],
                "full_sys": full_sys,
            })
    if args.limit is not None:
        work = work[: args.limit]
    print(f"[exp26] {len(work)} pairs queued")

    # Manifest
    manifest = {
        "ts": ts,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": JUDGE_REASONING,
        "n_pairs": len(work),
        "n_votes_per_pair": args.n_votes,
        "n_calls_total_target": len(work) * args.n_votes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    out_path = out_dir / "all.jsonl"
    t0 = time.time()
    n_done = 0

    def _do_one(item: dict) -> list[dict]:
        t_start = time.time()
        labels, raws = judge_one(
            full_sys=item["full_sys"],
            scen_prompt=item["scen_prompt"],
            opt_a=item["option_a"], opt_b=item["option_b"],
            n_votes=args.n_votes,
        )
        latency = time.time() - t_start
        rows = []
        for vi, (lab, raw) in enumerate(zip(labels, raws)):
            rows.append({
                "entry_id": item["entry_id"],
                "domain": item["domain"],
                "cell_id": item["cell_id"],
                "persona_idx": item["persona_idx"],
                "tone_idx": item["tone_idx"],
                "emphasis_idx": item["emphasis_idx"],
                "option_a": item["option_a"],
                "option_b": item["option_b"],
                "vote_idx": vi,
                "label": lab,
                "raw": raw,
                "latency_s": latency / args.n_votes,
            })
        return rows

    with open(out_path, "w") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do_one, item): i for i, item in enumerate(work)}
            for fut in as_completed(futs):
                try:
                    rows = fut.result()
                except Exception as exc:
                    print(f"[exp26] worker failed: {exc!r}")
                    continue
                for row in rows:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % 200 == 0 or n_done == len(work):
                    dt = time.time() - t0
                    rate = n_done / max(dt, 1e-3)
                    eta = (len(work) - n_done) / max(rate, 1e-3)
                    print(f"[exp26] {n_done}/{len(work)} pairs "
                          f"({rate * args.n_votes:.0f} calls/s, eta {eta:.0f}s)")
                    fout.flush()

    dt = time.time() - t0
    print(f"[exp26] done: {n_done} pairs ({n_done * args.n_votes} calls) "
          f"in {dt:.1f}s ({n_done * args.n_votes / max(dt, 1e-3):.1f} calls/s)")
    print(f"[exp26] wrote {out_path}")


if __name__ == "__main__":
    main()
