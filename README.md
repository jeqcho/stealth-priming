# Stealth priming

Code and committed artefacts to reproduce the paper *(NeurIPS 2026 supp.
material; anonymised for review)*. The headline result: a basis of
neutral-looking system prompts can shift a target model's binary
recommendations by ~50 percentage points on the underdog option, without
the prompts being detectable as biased by a strong LM judge.

## Quickstart (level 1: re-render plots, ~5 s, no GPU, no API)

```bash
git clone <this-repo> stealth-priming
cd stealth-priming
pip install -e .
make plots
```

That's it. `plots/paper/` will contain the 5 Gemma figures (4
calibration scatters + the 6-bar headline) regenerated from the
committed CSVs and tensors in `outputs/`. Compare against the
pre-shipped versions: they should match exactly.

The 3 Qwen replication figures ship pre-rendered (`*_qwen.png`); the
Qwen pipeline is documented but its calibration data is not bundled.

## Reproducibility levels

Each level subsumes the lower. `make plots` is L1; the others are
optional and have explicit dependencies.

| level | what it re-does | requires | wall |
|---|---|---|---|
| **L1** `make plots` | Re-render the 5 Gemma figures from committed artefacts | matplotlib only | ~5 s |
| **L2** `make selectors` | Re-fit PVP selectors from saved h_layer48.pt + v_s_layer48.pt and re-render plots | numpy + torch (CPU) | ~1 min |
| **L3** `make logprobs` | Fetch rollouts, re-derive log-odds + MDCL by prefilling Gemma 4 31B-it | + 1 H100/H200 (gated HF access) | ~30 min |
| **L4** `make rollouts` | Re-run the rollout sweep + judging from scratch | + GPU 4-8 hr + OpenAI ≈ \$50 | half-day |

Level details are in [`reports/REPRODUCIBILITY.md`](reports/REPRODUCIBILITY.md).

## What's in the box

```
.
├── data/
│   ├── scenarios.json              100 binary-decision scenarios
│   └── sys_prompts.json            100 neutral-basis cells (5×5×4 grid)
├── outputs/
│   ├── calibration_rec.csv         rate_a + log-odds + MDCL per (scen, cell)
│   ├── audit_per_pair.jsonl        GPT-5.4-nano bias-judge votes
│   ├── projections_pvp_sys.csv     PVP (sys prompt) scores per cell
│   ├── projections_pvp_roll.csv    PVP (rollouts) scores per cell
│   ├── h_layer48.pt                per-cell prompt-suffix activations (5376-d)
│   ├── v_s_layer48.pt              per-scenario contrastive directions
│   └── cell_index.json             cell ordering for the .pt tensors
├── plots/paper/                    8 paper figures (5 regen, 3 pre-rendered)
├── src/                            shared helpers (PROD prompt, judge_v2, etc.)
├── scripts/                        numbered pipeline stages
├── reports/
│   └── REPRODUCIBILITY.md          level-by-level guide with timings
├── pyproject.toml
├── Makefile                        L1-L4 entry points
├── .env.example                    HF_TOKEN, OPENAI_API_KEY templates
└── LICENSE                         MIT
```

## External artefacts (not bundled in the repo)

- **Rollouts** (~1.7 GB across 4 JSONL files): hosted on HuggingFace
  at [`jeqcho/stealth-priming-rollouts`](https://huggingface.co/datasets/jeqcho/stealth-priming-rollouts).
  Fetch via `make fetch-rollouts` (downloads to `artefacts/rollouts/`).
  Required for L3+.
- **Model weights**: `google/gemma-4-31B-it`. Gated; request access at
  [HF model page](https://huggingface.co/google/gemma-4-31B-it). Required for L3+.
- **Optional**: Qwen3.6-27B for the appendix replication. Not required
  for the headline figure.

## Setup

1. Python 3.11+. We recommend `uv`:
   ```bash
   uv venv .venv --python 3.11
   source .venv/bin/activate
   uv pip install -e .
   ```
   Or plain pip:
   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -e .
   ```

2. (L3+ only) Configure tokens:
   ```bash
   cp .env.example .env
   # fill in HF_TOKEN and OPENAI_API_KEY
   ```

3. Verify L1 works:
   ```bash
   make plots
   ```

## Pipeline overview

The full pipeline (L4) runs in this order:

1. **`scripts/01_generate_basis.py`** — Gemma 4 31B-it authors the 5×5×4=100
   neutral system prompts (the *basis*).
2. **`scripts/02_rollouts.py`** — sample 25 rollouts of Gemma per
   `(scenario, cell)` pair (10K pairs × 25 = 250K rollouts).
3. **`scripts/03_judge_rollouts.py`** — GPT-5.4-nano classifies each
   rollout into `option_a`, `option_b`, or `ambiguous`.
4. **`scripts/04_logprobs.py`** — prefill Gemma over the rec-prefix
   `"My recommendation: "` and compute `log P(A | sys+user, rec)`,
   `log P(B | ...)`, and the MDCL baseline; produce
   `calibration_rec.csv`.
5. **`scripts/05_audit_basis.py`** — GPT-5.4-nano bias judge with 3
   votes per (scenario, cell); produce `audit_per_pair.jsonl`.
6. **`scripts/06_pvp_sys_prompt.py`** — extract per-cell prompt-suffix
   activations from Gemma at layer 48; per-scenario 5-fold ridge
   regression of `rate_a` on these embeddings; emit out-of-fold
   projections.
7. **`scripts/07_pvp_rollouts.py`** — forward-pass each kept rollout
   through Gemma, mean-pool response-token activations, build per-
   scenario `v_s = mean(POS) − mean(NEG)`; emit projections.
8. **`scripts/08_paper_plots.py`** — regenerate the 5 Gemma figures
   from the artefacts produced by 04, 05, 06, 07.

## Method summary

We test whether a 100-cell basis of *neutral* system prompts (varying
persona × tone × emphasis but never naming options) can systematically
shift a model's choice on a binary decision. Five cell-selector
strategies rank the basis cells per scenario:

- `random`: expected rate of a uniformly drawn cell.
- `MDCL`: $\log P(A\,|\,\mathrm{sys}{+}\mathrm{cell}) - \log P(A\,|\,\mathrm{sys}\text{-only})$.
- `PVP (sys prompt)`: per-scenario ridge regression of empirical rate
  on Gemma's prompt-suffix activations.
- `PVP (rollouts)`: per-scenario `mean(POS) − mean(NEG)` of response-token
  activations from A-picking vs B-picking rollouts.
- `log-odds`: $\log P(A\,|\,\mathrm{sys}{+}\mathrm{rec}) - \log P(B\,|\,\mathrm{sys}{+}\mathrm{rec})$.

We additionally compare against an empirical `oracle` (max rate across
cells). All selectors are evaluated on the *audit-clean* subset --- the
60% of cells that pass a 3-vote GPT-5.4-nano bias judge as
`unbiased` --- so the demonstration is that bias persists in prompts a
strong LM auditor cannot flag.

Headline result on Gemma 4 31B-it (audit-clean, baseline rate of target
$<$ 0.5):

| selector | mean rate of choosing target |
|---|---|
| random | 0.17 |
| MDCL | 0.28 |
| PVP (rollouts) | 0.48 |
| PVP (sys prompt) | 0.49 |
| log-odds | 0.50 |
| oracle | 0.64 |

## License

MIT (see [LICENSE](LICENSE)).

## Citation

```bibtex
@inproceedings{stealth-priming-2026,
  title  = {Stealth priming: hidden bias in neutral system prompts},
  author = {Anonymous},
  booktitle = {NeurIPS},
  year   = {2026},
  note   = {Anonymized for review.}
}
```
