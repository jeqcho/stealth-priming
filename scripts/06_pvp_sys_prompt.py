"""L2 reproduction: re-fit PVP (sys prompt) from saved activations.

For each scenario, do 5-fold CV ridge regression of rate_a on the
per-cell prompt-suffix embeddings h_c (saved at outputs/h_layer48.pt).
Produces outputs/projections_pvp_sys.csv with one row per
(scenario, cell): the out-of-fold projection score.

Inputs:
  outputs/h_layer48.pt      shape (100, hidden_dim) float16
  outputs/cell_index.json   list of cell_id in row order
  outputs/calibration_rec.csv   for rate_a labels per (scen, cell)

Output:
  outputs/projections_pvp_sys.csv
    columns: entry_id, cell_id, proj_oof, fold, rate_a, layer, lambda
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from persona_vec import kfold_oof_scores  # noqa: E402

OUTPUTS = ROOT / "outputs"
LAYER = 48
LAMBDA = 100.0
N_FOLDS = 5
SEED = 0


def main() -> None:
    print(f"[pvp-sys] reading inputs from {OUTPUTS}")

    cell_ids = json.loads((OUTPUTS / "cell_index.json").read_text())
    H = torch.load(OUTPUTS / f"h_layer{LAYER}.pt", map_location="cpu",
                   weights_only=True).numpy().astype(np.float32)
    assert H.shape[0] == len(cell_ids), \
        f"h shape {H.shape[0]} vs cell_index {len(cell_ids)}"

    by_scen = defaultdict(dict)
    with (OUTPUTS / "calibration_rec.csv").open() as f:
        for r in csv.DictReader(f):
            try:
                by_scen[r["entry_id"]][r["cell_id"]] = float(r["rate_a_empirical"])
            except (ValueError, KeyError):
                continue

    # Drop scenarios where rate_a is constant across cells
    usable = []
    for eid, rates in by_scen.items():
        vals = [rates.get(c) for c in cell_ids]
        if any(v is None for v in vals):
            continue
        if max(vals) == 0.0 or min(vals) == 1.0:
            continue
        usable.append(eid)
    print(f"[pvp-sys] {len(usable)} usable scenarios "
          f"({len(by_scen) - len(usable)} dropped: constant rate_a)")

    out_path = OUTPUTS / "projections_pvp_sys.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entry_id", "cell_id", "proj_oof",
                                          "fold", "rate_a", "layer", "lambda"])
        w.writeheader()
        for eid in usable:
            y = np.array([by_scen[eid][c] for c in cell_ids], dtype=np.float32)
            proj_oof, fold_idx = kfold_oof_scores(
                H, y, n_folds=N_FOLDS, lam=LAMBDA, seed=SEED, method="ridge")
            for i, cid in enumerate(cell_ids):
                w.writerow({
                    "entry_id": eid, "cell_id": cid,
                    "proj_oof": float(proj_oof[i]),
                    "fold": int(fold_idx[i]),
                    "rate_a": float(y[i]),
                    "layer": LAYER, "lambda": LAMBDA,
                })

    print(f"[pvp-sys] wrote {out_path} "
          f"({len(usable) * len(cell_ids)} rows)")


if __name__ == "__main__":
    main()
