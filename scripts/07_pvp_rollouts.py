"""L2 reproduction: re-derive PVP (rollouts) projections from saved
v_s and h_c.

For PVP (rollouts), v_s is computed from response-token activations of
A-picking vs B-picking rollouts. Re-extracting v_s from raw rollout
activations is L3 territory (requires GPU forward passes through every
rollout). Here we re-use the committed v_s_layer48.pt (per-scenario
unit-norm directions) and re-derive the projections by computing
proj = ⟨h_c, v_s⟩ / ‖h_c‖ for each (scenario, cell), reproducing
outputs/projections_pvp_roll.csv exactly.

Inputs:
  outputs/h_layer48.pt        per-cell prompt-suffix activations
  outputs/cell_index.json     cell_id ordering
  outputs/v_s_layer48.pt      dict {entry_id: ndarray} of unit-norm directions

Output:
  outputs/projections_pvp_roll.csv
    columns: entry_id, cell_id, proj, layer
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
LAYER = 48


def main() -> None:
    cell_ids = json.loads((OUTPUTS / "cell_index.json").read_text())
    H = torch.load(OUTPUTS / f"h_layer{LAYER}.pt", map_location="cpu",
                   weights_only=True).numpy().astype(np.float32)
    v_s_dict = torch.load(OUTPUTS / f"v_s_layer{LAYER}.pt",
                          map_location="cpu", weights_only=False)
    print(f"[pvp-roll] {len(cell_ids)} cells, {len(v_s_dict)} scenarios")

    # cosine projection: row-normalize H, v_s already unit-norm
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    H_unit = H / norms

    out_path = OUTPUTS / "projections_pvp_roll.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entry_id", "cell_id", "proj", "layer"])
        w.writeheader()
        for eid, v in v_s_dict.items():
            v = np.asarray(v, dtype=np.float32)
            scores = H_unit @ v  # (100,)
            for cid, p in zip(cell_ids, scores):
                w.writerow({
                    "entry_id": eid, "cell_id": cid,
                    "proj": float(p), "layer": LAYER,
                })

    print(f"[pvp-roll] wrote {out_path} "
          f"({len(cell_ids) * len(v_s_dict)} rows)")


if __name__ == "__main__":
    main()
