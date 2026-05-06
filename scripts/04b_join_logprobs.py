"""Stage 2: join logprobs with exp 22 thinkOFF rollout fractions.

Inputs:
  - outputs/logprobs_<ts>/logprobs.jsonl  (this experiment, stage 1)
  - experiments/22-neutral-prompts/outputs/rollouts_thinkOFF_<ts>/all.judged.jsonl
  - experiments/22-neutral-prompts/outputs/baselines_<ts>/thinkOFF.judged.jsonl

Output: outputs/calibration.csv (one row per (entry_id, cell_id)) with
columns:
  entry_id, domain, cell_id (or _baseline_),
  option_a, option_b,
  rate_a_empirical, rate_b_empirical, rate_ambiguous_empirical, n_samples,
  delta_sum, delta_first,
  n_tokens_a, n_tokens_b,
  baseline_delta_sum, baseline_delta_first    (per scenario)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))


import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP22_OUTPUTS = PROJECT_ROOT / "artefacts" / "rollouts"
SCENARIOS_PATH = PROJECT_ROOT / "data" / "scenarios.json"


def latest_baselines_judged() -> Path:
    runs = sorted(EXP22_OUTPUTS.glob("baselines_*"))
    if not runs:
        raise SystemExit("no exp22 baselines_* found")
    return runs[-1] / "thinkOFF.judged.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logprobs-dir", required=True,
                   help="outputs/logprobs_<ts>/")
    p.add_argument("--rollouts-judged", default=None,
                   help="path to rollouts_thinkOFF_<ts>/all.judged.jsonl"
                        " (defaults to most recent)")
    p.add_argument("--baselines-judged", default=None,
                   help="path to baselines_<ts>/thinkOFF.judged.jsonl"
                        " (defaults to most recent)")
    p.add_argument("--out-csv", default=None,
                   help="defaults to outputs/calibration.csv")
    p.add_argument("--out-summary", default=None,
                   help="defaults to outputs/calibration_summary.json")
    p.add_argument("--tag", default="",
                   help="If set, write to outputs/calibration_<tag>.csv.")
    return p.parse_args()


def latest_rollouts() -> Path:
    runs = sorted(EXP22_OUTPUTS.glob("rollouts_thinkOFF_*"))
    if not runs:
        raise SystemExit("no exp22 rollouts_thinkOFF_* found")
    return runs[-1] / "all.judged.jsonl"


def main() -> None:
    args = parse_args()
    logprobs_dir = Path(args.logprobs_dir)
    rollouts_judged = (Path(args.rollouts_judged)
                       if args.rollouts_judged else latest_rollouts())
    baselines_judged = (Path(args.baselines_judged)
                        if args.baselines_judged else latest_baselines_judged())
    print(f"[join] logprobs_dir={logprobs_dir}")
    print(f"[join] rollouts_judged={rollouts_judged}")
    print(f"[join] baselines_judged={baselines_judged}")

    scenarios = {s["entry_id"]: s for s in json.loads(SCENARIOS_PATH.read_text())}

    # 1. Load logprobs into per-(entry, cell) records.
    logp_rows: list[dict] = []
    with open(logprobs_dir / "logprobs.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            logp_rows.append(json.loads(line))

    # group by (entry_id, cell_id) -> {"option_a": {logp_sum, logp_first, n_tokens},
    #                                  "option_b": {...}}
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in logp_rows:
        key = (r["entry_id"], r["cell_id"])
        grouped[key][r["option_label"]] = {
            "logp_sum": r["logp_sum"],
            "logp_first": r["logp_first"],
            "n_tokens": r["n_option_tokens"],
            "option_str": r["option_str"],
        }

    # 2. Load rollout judged labels into per-(entry, cell) rates.
    rates: dict[tuple[str, str], dict] = {}
    bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    with open(rollouts_judged) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (d["entry_id"], d["cell_id"])
            bucket[key].append(d.get("judge_label", "ambiguous"))
    for key, labels in bucket.items():
        n = len(labels)
        rates[key] = {
            "rate_a": labels.count("option_a") / n,
            "rate_b": labels.count("option_b") / n,
            "rate_ambig": labels.count("ambiguous") / n,
            "n": n,
        }

    # 3. For each scenario, look up the baseline (cell_id="_baseline_") logprobs.
    baseline_logp: dict[str, dict[str, dict[str, float]]] = {}  # eid -> {opt_label -> {sum, first}}
    baseline_delta: dict[str, dict[str, float]] = {}
    for (eid, cid), opts in grouped.items():
        if cid != "_baseline_":
            continue
        if "option_a" in opts and "option_b" in opts:
            baseline_logp[eid] = {
                "option_a": {"sum": opts["option_a"]["logp_sum"],
                             "first": opts["option_a"]["logp_first"]},
                "option_b": {"sum": opts["option_b"]["logp_sum"],
                             "first": opts["option_b"]["logp_first"]},
            }
            baseline_delta[eid] = {
                "delta_sum":
                    opts["option_a"]["logp_sum"] - opts["option_b"]["logp_sum"],
                "delta_first":
                    opts["option_a"]["logp_first"] - opts["option_b"]["logp_first"],
            }

    # 3b. Empirical baseline rates (rollout fraction from PROD-only).
    rate_a_baseline_emp: dict[str, dict] = {}
    bucket_b: dict[str, list[str]] = defaultdict(list)
    with open(baselines_judged) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            bucket_b[d["entry_id"]].append(d.get("judge_label", "ambiguous"))
    for eid, labels in bucket_b.items():
        n = len(labels)
        rate_a_baseline_emp[eid] = {
            "rate_a": labels.count("option_a") / n,
            "rate_b": labels.count("option_b") / n,
            "n": n,
        }

    # 4. Build the joined table (main rows only; the baseline column is
    #    looked up per scenario).
    out_rows: list[dict] = []
    for (eid, cid), opts in grouped.items():
        if cid == "_baseline_":
            continue
        if "option_a" not in opts or "option_b" not in opts:
            continue
        scen = scenarios.get(eid)
        if scen is None:
            continue
        rate = rates.get((eid, cid))
        if rate is None:
            # No rollout data for this cell -- skip
            continue
        bd = baseline_delta.get(eid, {})
        bl = baseline_logp.get(eid, {})
        bemp = rate_a_baseline_emp.get(eid, {})

        # Treatment effect (logit-space) per option:
        #   te_X = log P(X | sys+prompt) - log P(X | empty+prompt)
        te_a_sum = (opts["option_a"]["logp_sum"]
                    - bl.get("option_a", {}).get("sum", float("nan"))
                    if "option_a" in bl else float("nan"))
        te_b_sum = (opts["option_b"]["logp_sum"]
                    - bl.get("option_b", {}).get("sum", float("nan"))
                    if "option_b" in bl else float("nan"))
        te_a_first = (opts["option_a"]["logp_first"]
                      - bl.get("option_a", {}).get("first", float("nan"))
                      if "option_a" in bl else float("nan"))
        te_b_first = (opts["option_b"]["logp_first"]
                      - bl.get("option_b", {}).get("first", float("nan"))
                      if "option_b" in bl else float("nan"))

        # Empirical treatment effect (rollout-space):
        #   empir_te_X = rate_X(sys) - rate_X(empty)
        rate_a_baseline = bemp.get("rate_a", float("nan"))
        rate_b_baseline = bemp.get("rate_b", float("nan"))
        empir_te_a = rate["rate_a"] - rate_a_baseline if rate_a_baseline == rate_a_baseline else float("nan")
        empir_te_b = rate["rate_b"] - rate_b_baseline if rate_b_baseline == rate_b_baseline else float("nan")

        out_rows.append({
            "entry_id": eid,
            "domain": scen["domain"],
            "cell_id": cid,
            "option_a": scen["options"][0],
            "option_b": scen["options"][1],
            "n_samples": rate["n"],
            "rate_a_empirical": rate["rate_a"],
            "rate_b_empirical": rate["rate_b"],
            "rate_ambiguous_empirical": rate["rate_ambig"],
            "rate_a_baseline_empirical": rate_a_baseline,
            "rate_b_baseline_empirical": rate_b_baseline,
            "delta_sum": opts["option_a"]["logp_sum"] - opts["option_b"]["logp_sum"],
            "delta_first": opts["option_a"]["logp_first"] - opts["option_b"]["logp_first"],
            "n_tokens_a": opts["option_a"]["n_tokens"],
            "n_tokens_b": opts["option_b"]["n_tokens"],
            "baseline_delta_sum": bd.get("delta_sum"),
            "baseline_delta_first": bd.get("delta_first"),
            "te_a_sum": te_a_sum,
            "te_b_sum": te_b_sum,
            "te_a_first": te_a_first,
            "te_b_first": te_b_first,
            "empir_te_a": empir_te_a,
            "empir_te_b": empir_te_b,
        })

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = (Path(args.out_csv) if args.out_csv
               else PROJECT_ROOT / "outputs" / f"calibration{suffix}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[join] wrote {out_csv} ({len(out_rows)} rows)")

    summary = {
        "n_rows": len(out_rows),
        "n_scenarios": len(set(r["entry_id"] for r in out_rows)),
        "n_cells": len(set(r["cell_id"] for r in out_rows)),
        "n_baseline_resolved": len(baseline_delta),
        "rollouts_judged": str(rollouts_judged),
    }
    out_summary = (Path(args.out_summary) if args.out_summary
                   else PROJECT_ROOT / "outputs" / f"calibration_summary{suffix}.json")
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"[join] wrote {out_summary}")


if __name__ == "__main__":
    main()
