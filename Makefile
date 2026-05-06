.PHONY: help install plots selectors fetch-rollouts logprobs rollouts clean

PY ?= python

help:
	@echo "Targets (each higher level subsumes the lower):"
	@echo "  install         pip install -e ."
	@echo "  plots           [L1] regenerate the 8 paper figures from committed artefacts (~1 min, no GPU/API)"
	@echo "  selectors       [L2] re-fit PVP selectors from saved activations (~5 min, CPU)"
	@echo "  fetch-rollouts  [L3 prereq] download judged rollouts from HF dataset (~1.7 GB)"
	@echo "  logprobs        [L3] re-derive log-odds + MDCL by prefilling Gemma 4 31B-it (~30 min, 1 H100)"
	@echo "  rollouts        [L4] re-run the rollout sweep end-to-end (half-day, GPU + ~\\\$50 OpenAI)"
	@echo ""
	@echo "Quickstart for reviewers:    make install && make plots"

install:
	$(PY) -m pip install -e .

plots:
	$(PY) scripts/08_paper_plots.py

selectors:
	$(PY) scripts/06_pvp_sys_prompt.py
	$(PY) scripts/07_pvp_rollouts.py
	$(PY) scripts/08_paper_plots.py

fetch-rollouts:
	$(PY) scripts/00_fetch_rollouts.py

logprobs: fetch-rollouts
	$(PY) scripts/04_logprobs.py
	$(PY) scripts/08_paper_plots.py

rollouts:
	$(PY) scripts/01_generate_basis.py
	$(PY) scripts/02_rollouts.py
	$(PY) scripts/03_judge_rollouts.py
	$(MAKE) logprobs
	$(PY) scripts/05_audit_basis.py
	$(MAKE) selectors

clean:
	rm -rf logs __pycache__ .pytest_cache
