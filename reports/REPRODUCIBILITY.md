# Reproducibility levels

Each level subsumes the lower. The `Makefile` exposes the entry points.

## L1 — Re-render plots from committed artefacts

```bash
make plots
```

- Inputs: `outputs/calibration_rec.csv`, `outputs/audit_per_pair.jsonl`,
  `outputs/projections_pvp_sys.csv`, `outputs/projections_pvp_roll.csv`.
- Outputs: 5 PNGs in `plots/paper/` (4 calibration scatters + headline bar).
- Wall: ~5 seconds. CPU only. No HF or OpenAI access.

What this verifies: every selector's mean rate, every calibration
correlation, every per-domain sigmoid in the paper's figures matches
when computed from the committed CSVs.

## L2 — Re-fit PVP selectors from saved activations

```bash
make selectors
```

- Inputs: above + `outputs/h_layer48.pt`, `outputs/v_s_layer48.pt`,
  `outputs/cell_index.json`.
- Outputs: `outputs/projections_pvp_sys.csv`,
  `outputs/projections_pvp_roll.csv`, then 5 PNGs as L1.
- Wall: ~30 s. CPU only.

What this verifies: the per-scenario PVP directions and out-of-fold
projection scores can be re-derived from the saved activations. The
final selector picks should be identical to the committed CSVs.

## L3 — Re-derive log-odds and MDCL from rollouts

```bash
make fetch-rollouts    # downloads ~1.7 GB from HuggingFace
make logprobs           # prefills Gemma 4 31B-it on (sys, user, rec) chats
```

- Inputs: rollouts at `artefacts/rollouts/{gemma_orig_thinkoff,
  gemma_flipped, qwen_orig, audit_per_pair}.jsonl` (auto-fetched).
- Outputs: regenerated `outputs/calibration_rec.csv`, then 5 PNGs.
- Wall: ~30 min on 1 H100/H200. Requires `HF_TOKEN` (Gemma is gated).

Equivalent to the original Gemma forward pass at the rec-prefix
position. The judged rollouts come pre-bundled in the HF dataset, so
no fresh API calls are needed at this level.

## L4 — Re-run rollouts end-to-end

```bash
make rollouts
```

- Generates the basis (Gemma authors 100 system prompts), runs 250K
  rollouts (25 samples × 100 cells × 100 scenarios), judges every
  rollout via GPT-5.4-nano, runs the audit, fits the selectors, and
  re-renders plots.
- Wall: 4-8 hr GPU + ~$50 OpenAI tier-5 credits.
- Requires `HF_TOKEN`, `OPENAI_API_KEY`, and a 4× H100/H200 (or
  similar) box.

## What each level verifies

| level | verifies |
|---|---|
| L1 | The plots match the saved selector scores exactly. |
| L2 | The selector scores can be re-derived from saved activations. |
| L3 | Log-odds + MDCL reflect Gemma's actual logits on the rollouts. |
| L4 | The rollouts and judge labels reproduce within sampling noise. |

L2 numerical agreement is exact (deterministic mean-diff and ridge with
fixed seed). L3 should be exact for log-odds (deterministic prefill).
L4 results should match within ±2pp on aggregate selector means
(GPT-5.4-nano sampling at temperature 1.0 introduces some noise).
