"""Stage 1: compute logp(option_A) and logp(option_B) for every
(scenario, sys_prompt) pair, plus a PROD-only baseline.

Pipeline:
  - 100 scenarios x 100 sys_prompts x 2 options = 20,000 main passes
  - 100 scenarios x 0 sys_prompts (PROD only) x 2 options = 200 baseline passes

All 20,200 prefill passes go into one DP=N call, so the model loads once.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))


import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from helpers import build_rollout_messages  # from exp 22 src
from logprob_runner import compute_logprobs_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = PROJECT_ROOT / "data" / "scenarios.json"
SYSPROMPTS_PATH = PROJECT_ROOT / "data" / "sys_prompts.json"
OUTPUTS = PROJECT_ROOT / "outputs"
LOGS = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--continuation-prefix", type=str, default="",
                   help="If set, appended after the chat-template exit "
                        "and before the option literal. E.g. "
                        "'My recommendation: ' forces the option to be "
                        "the literal next content token.")
    p.add_argument("--tag", type=str, default="",
                   help="If set, appended to the output dir name as "
                        "logprobs_<tag>_<ts>/ to distinguish runs.")
    p.add_argument("--enable-thinking-template", action="store_true",
                   help="Apply chat template with enable_thinking=True. "
                        "Combine with --continuation-prefix '<channel|>...' "
                        "to force-close the thinking trace and score "
                        "the option there.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    scenarios = json.loads(SCENARIOS_PATH.read_text())
    sys_prompts = json.loads(SYSPROMPTS_PATH.read_text())["prompts"]
    assert len(scenarios) == 100, f"expected 100 scenarios, got {len(scenarios)}"
    assert len(sys_prompts) == 100, f"expected 100 sys prompts, got {len(sys_prompts)}"

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.tag:
        out_dir = OUTPUTS / f"logprobs_{args.tag}_{ts}"
    else:
        out_dir = OUTPUTS / f"logprobs_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[logprobs] out_dir={out_dir}")

    # Build the full work list. We tag each item with which "kind" it is
    # so downstream join knows whether it's main or baseline.
    items: list[dict] = []
    for scen in scenarios:
        # Baseline pass: PROD only, no generated sys prompt.
        msgs_baseline = build_rollout_messages(None, scen["prompt"])
        for opt_label, opt_str in (("option_a", scen["options"][0]),
                                   ("option_b", scen["options"][1])):
            items.append({
                "kind": "baseline",
                "entry_id": scen["entry_id"],
                "cell_id": "_baseline_",
                "option_label": opt_label,
                "option_str": opt_str,
                "messages": msgs_baseline,
            })
        # Main pass: PROD + each generated sys prompt.
        for sp in sys_prompts:
            msgs_main = build_rollout_messages(sp["text"], scen["prompt"])
            for opt_label, opt_str in (("option_a", scen["options"][0]),
                                       ("option_b", scen["options"][1])):
                items.append({
                    "kind": "main",
                    "entry_id": scen["entry_id"],
                    "cell_id": sp["cell_id"],
                    "option_label": opt_label,
                    "option_str": opt_str,
                    "messages": msgs_main,
                })

    n_baseline = sum(1 for it in items if it["kind"] == "baseline")
    n_main = sum(1 for it in items if it["kind"] == "main")
    assert n_baseline == 200, n_baseline
    assert n_main == 20_000, n_main
    print(f"[logprobs] total items: {len(items)} "
          f"(baseline={n_baseline}, main={n_main})")

    # Hand to runner. Worker only needs messages + option_str + optional
    # continuation prefix.
    worker_items = [
        {
            "messages": it["messages"],
            "option_str": it["option_str"],
            "prefix_continuation": args.continuation_prefix,
        }
        for it in items
    ]

    t0 = time.time()
    results = compute_logprobs_batch(
        items=worker_items,
        work_dir=LOGS / f"logprobs_tmp_{ts}",
        max_model_len=args.max_model_len,
        seed=0,
        enable_thinking_template=args.enable_thinking_template,
    )
    print(f"[logprobs] runner returned in {time.time() - t0:.1f}s")

    out_path = out_dir / "logprobs.jsonl"
    with open(out_path, "w") as f:
        for it, r in zip(items, results):
            row = {
                "kind": it["kind"],
                "entry_id": it["entry_id"],
                "cell_id": it["cell_id"],
                "option_label": it["option_label"],
                "option_str": it["option_str"],
                "logp_sum": r["logp_sum"],
                "logp_first": r["logp_first"],
                "n_option_tokens": r["n_option_tokens"],
                "prefix_n_tokens": r["prefix_n_tokens"],
                "logp_per_token": r["logp_per_token"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[logprobs] wrote {out_path} ({len(items)} rows)")

    # Manifest
    manifest = {
        "ts": ts,
        "n_scenarios": len(scenarios),
        "n_sys_prompts": len(sys_prompts),
        "n_baseline": n_baseline,
        "n_main": n_main,
        "max_model_len": args.max_model_len,
        "continuation_prefix": args.continuation_prefix,
        "tag": args.tag,
        "enable_thinking_template": args.enable_thinking_template,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
