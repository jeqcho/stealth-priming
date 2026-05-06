"""Regenerate the 5 Gemma paper figures from committed artefacts.

Inputs (read from outputs/):
  calibration_rec.csv          rate_a + log-odds + MDCL per (scen, cell)
  audit_per_pair.jsonl         judge votes per (scen, cell); we filter to
                               all_unbiased=True for the audit-clean subset
  projections_pvp_sys.csv      out-of-fold projections for PVP (sys prompt)
  projections_pvp_roll.csv     projections for PVP (rollouts)

Produces in plots/paper/ (overwriting):
  calibration_scatter_rec.png
  calibration_scatter_rec_per_domain.png
  calibration_scatter_rec_mdcl.png
  calibration_scatter_rec_mdcl_per_domain.png
  cell_selector_bars_non_dominant_with_contrastive_audit_gemma.png

The 3 Qwen figures (calibration_scatter_rec_qwen.png,
calibration_scatter_rec_mdcl_qwen.png, cell_selector_bars_non_dominant_qwen.png)
ship pre-rendered in plots/paper/ from the Qwen end-to-end pipeline; this
script does not regenerate them (no Qwen calibration_rec is committed).

Usage:
    python scripts/08_paper_plots.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
PLOTS = ROOT / "plots" / "paper"

CALIB = OUTPUTS / "calibration_rec.csv"
AUDIT = OUTPUTS / "audit_per_pair.jsonl"
PROJ_PVP_SYS = OUTPUTS / "projections_pvp_sys.csv"
PROJ_PVP_ROLL = OUTPUTS / "projections_pvp_roll.csv"


# ---------- shared loaders ----------

def load_calib() -> list[dict]:
    rows = []
    with CALIB.open() as f:
        for r in csv.DictReader(f):
            for k in ("rate_a_empirical", "rate_b_empirical",
                      "rate_a_baseline_empirical", "rate_b_baseline_empirical",
                      "delta_sum", "te_a_sum", "te_b_sum"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "") else float("nan")
            rows.append(r)
    return rows


def load_audit_clean() -> set[tuple[str, str]]:
    out = set()
    with AUDIT.open() as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("all_unbiased"):
                out.add((obj["entry_id"], obj["cell_id"]))
    return out


def load_proj(path: Path) -> dict[tuple[str, str], float]:
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            v = r.get("proj_oof") or r.get("proj")
            out[(r["entry_id"], r["cell_id"])] = float(v)
    return out


# ---------- calibration helpers ----------

def fit_T(delta: np.ndarray, rate: np.ndarray) -> float:
    mask = ~(np.isnan(delta) | np.isnan(rate))
    d = delta[mask]; r = rate[mask]
    Ts = np.logspace(-1, 2, 200)
    losses = [float(np.mean((1 / (1 + np.exp(-d / T)) - r) ** 2)) for T in Ts]
    return float(Ts[int(np.argmin(losses))])


def calibration_scatter(rows: list[dict], metric: str, out: Path,
                        figsize=(3.25, 2.5)):
    if metric == "log_odds":
        x_col = "delta_sum"
        x_label = r"log-odds $= \log P(A) - \log P(B)$"
    else:
        x_col = "te_a_sum"
        x_label = r"MDCL $= \log P(A \mid \mathrm{sys}) - \log P(A \mid \emptyset)$"

    delta = np.array([r[x_col] for r in rows])
    rate = np.array([r["rate_a_empirical"] for r in rows])
    mask = ~(np.isnan(delta) | np.isnan(rate))
    r_p = float(np.corrcoef(delta[mask], rate[mask])[0, 1])
    T = fit_T(delta, rate)

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.5, "lines.linewidth": 1.0,
        "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    })
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.scatter(delta, rate, c="#444444", s=2, alpha=0.10, edgecolors="none")
    xs = np.linspace(np.nanmin(delta), np.nanmax(delta), 200)
    ys = 1.0 / (1.0 + np.exp(-xs / T))
    ax.plot(xs, ys, color="#d62728", lw=1.4)
    ax.axhline(0.5, color="grey", ls=":", alpha=0.4, lw=0.5)
    ax.axvline(0.0, color="grey", ls=":", alpha=0.4, lw=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"empirical rate of option $A$")
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.03, 0.97, f"$r = {r_p:.2f}$\n$T = {T:.1f}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                      boxstyle="round,pad=0.2"))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out.name}  r={r_p:.3f}  T={T:.2f}")


def calibration_scatter_per_domain(rows: list[dict], metric: str, out: Path,
                                   figsize=(3.25, 2.5)):
    x_col = "delta_sum" if metric == "log_odds" else "te_a_sum"
    x_label = (r"log-odds $= \log P(A) - \log P(B)$" if metric == "log_odds"
               else r"MDCL $= \log P(A \mid \mathrm{sys}) - \log P(A \mid \emptyset)$")

    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append((r[x_col], r["rate_a_empirical"]))

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "lines.linewidth": 1.2,
        "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    })
    colors = ["#1F77B4", "#228833", "#EE6677", "#CCBB44", "#AA3377"]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    all_x = []
    for color, (dom, pts) in zip(colors, sorted(by_dom.items())):
        d = np.array([p[0] for p in pts]); r = np.array([p[1] for p in pts])
        T = fit_T(d, r)
        r_p = float(np.corrcoef(d[~np.isnan(d)], r[~np.isnan(r)])[0, 1])
        all_x.extend(d.tolist())
        xs = np.linspace(np.nanmin(d), np.nanmax(d), 80)
        ys = 1.0 / (1.0 + np.exp(-xs / T))
        ax.plot(xs, ys, color=color, lw=1.4,
                label=f"{dom} (r={r_p:.2f}, T={T:.1f})")

    ax.axhline(0.5, color="grey", ls=":", alpha=0.4, lw=0.5)
    ax.axvline(0.0, color="grey", ls=":", alpha=0.4, lw=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"empirical rate of option $A$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=6.5)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out.name}")


# ---------- bar plot ----------

def bar_plot_audit(rows: list[dict], proj32, proj33, audit_set, out: Path,
                   threshold=0.5, figsize=(3.6, 3.6)):
    by_scen = defaultdict(list)
    for r in rows:
        by_scen[r["entry_id"]].append(r)

    base_v, rand_v, mdcl_v, lo_v, pv32_v, pv33_v, oracle_v = (
        [], [], [], [], [], [], [])
    n_skipped = 0; n_zero33 = 0
    for eid, cells in by_scen.items():
        pool = [c for c in cells if (eid, c["cell_id"]) in audit_set]
        if not pool: continue
        p32_list = [proj32.get((eid, c["cell_id"])) for c in pool]
        if any(p is None for p in p32_list):
            n_skipped += 1; continue
        p32 = np.array(p32_list)
        p33_list = [proj33.get((eid, c["cell_id"])) for c in pool]
        if any(p is None for p in p33_list):
            p33 = np.zeros(len(pool)); n_zero33 += 1
        else:
            p33 = np.array(p33_list)

        base_a = pool[0]["rate_a_baseline_empirical"]
        base_b = pool[0]["rate_b_baseline_empirical"]
        for target in ("A", "B"):
            if target == "A":
                base = base_a
                rate = np.array([c["rate_a_empirical"] for c in pool])
                d = np.array([c["delta_sum"] for c in pool])
                te = np.array([c["te_a_sum"] for c in pool])
                idx_d = int(np.argmax(d)); idx_t = int(np.argmax(te))
                idx_p32 = int(np.argmax(p32)); idx_p33 = int(np.argmax(p33))
            else:
                base = base_b
                rate = np.array([c["rate_b_empirical"] for c in pool])
                d = np.array([c["delta_sum"] for c in pool])
                te = np.array([c["te_b_sum"] for c in pool])
                idx_d = int(np.argmin(d)); idx_t = int(np.argmax(te))
                idx_p32 = int(np.argmin(p32)); idx_p33 = int(np.argmin(p33))
            if not (base == base) or base >= threshold: continue
            base_v.append(base); rand_v.append(float(rate.mean()))
            oracle_v.append(float(rate.max()))
            mdcl_v.append(float(rate[idx_t])); lo_v.append(float(rate[idx_d]))
            pv32_v.append(float(rate[idx_p32])); pv33_v.append(float(rate[idx_p33]))

    arrs = {
        "random":      np.array(rand_v),
        "MDCL":        np.array(mdcl_v),
        "PVP\n(rollouts)":   np.array(pv33_v),
        "PVP\n(sys prompt)": np.array(pv32_v),
        "log-odds":    np.array(lo_v),
        "oracle":      np.array(oracle_v),
    }
    color_of = {
        "random": "#7F7F7F",
        "MDCL": "#228833",
        "PVP\n(rollouts)": "#44AA99",
        "PVP\n(sys prompt)": "#CCBB44",
        "log-odds": "#1F77B4",
        "oracle": "#EE6677",
    }
    sorted_keys = sorted(arrs.keys(), key=lambda k: arrs[k].mean())
    means = [arrs[k].mean() for k in sorted_keys]
    cis = [1.96 * arrs[k].std(ddof=1) / np.sqrt(len(arrs[k])) for k in sorted_keys]
    colors = [color_of[k] for k in sorted_keys]

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7,
        "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    })
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    bars = ax.bar(sorted_keys, means, yerr=cis, color=colors,
                  edgecolor="none", linewidth=0,
                  width=0.78, capsize=2.5, alpha=0.9,
                  error_kw={"elinewidth": 0.7, "ecolor": "black"})
    for b, val, ci in zip(bars, means, cis):
        ax.text(b.get_x() + b.get_width() / 2, val + ci + 0.015,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean rate of choosing target")
    ax.grid(axis="y", alpha=0.25, lw=0.4); ax.set_axisbelow(True)
    ax.tick_params(axis="x", pad=2, labelrotation=35)
    ax.tick_params(axis="y", pad=1)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    n = len(base_v)
    print(f"[plot] {out.name}  n={n} (skipped {n_skipped})")
    print("       means: " +
          ", ".join(f"{k.replace(chr(10),' ')}={m:.3f}"
                    for k, m in zip(sorted_keys, means)))


# ---------- driver ----------

def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    print(f"[plot] reading inputs from {OUTPUTS}")
    rows = load_calib()
    audit_set = load_audit_clean()
    proj32 = load_proj(PROJ_PVP_SYS)
    proj33 = load_proj(PROJ_PVP_ROLL)
    print(f"[plot] {len(rows)} (scen, cell) pairs; "
          f"{len(audit_set)} audit-clean; "
          f"{len(proj32)} proj_sys; {len(proj33)} proj_roll")

    audit_rows = [r for r in rows if (r["entry_id"], r["cell_id"]) in audit_set]
    print(f"[plot] audit-clean subset: {len(audit_rows)} rows")

    # 4 calibration scatters on audit-clean Gemma data
    calibration_scatter(audit_rows, "log_odds",
                        PLOTS / "calibration_scatter_rec.png")
    calibration_scatter(audit_rows, "mdcl",
                        PLOTS / "calibration_scatter_rec_mdcl.png")
    calibration_scatter_per_domain(audit_rows, "log_odds",
                        PLOTS / "calibration_scatter_rec_per_domain.png")
    calibration_scatter_per_domain(audit_rows, "mdcl",
                        PLOTS / "calibration_scatter_rec_mdcl_per_domain.png")

    # Headline bar chart (audit-clean Gemma, threshold 0.5)
    bar_plot_audit(rows, proj32, proj33, audit_set,
                   PLOTS / "cell_selector_bars_non_dominant_with_contrastive_audit_gemma.png")

    print("[plot] done. Qwen plots ship pre-rendered (not regenerated here).")


if __name__ == "__main__":
    main()
