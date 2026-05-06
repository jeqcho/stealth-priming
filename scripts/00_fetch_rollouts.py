"""L3 prerequisite: download the judged-rollout dataset from HuggingFace.

Pulls the 4 JSONL files from
https://huggingface.co/datasets/jeqcho/stealth-priming-rollouts to
artefacts/rollouts/ in the project root.

Total size: ~1.7 GB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "jeqcho/stealth-priming-rollouts"
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "artefacts" / "rollouts"

FILES = [
    "gemma_orig_thinkoff.jsonl",
    "gemma_flipped.jsonl",
    "qwen_orig.jsonl",
    "audit_per_pair.jsonl",
]


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")  # optional: dataset is public
    print(f"[fetch] target: {TARGET}")
    for fname in FILES:
        local = TARGET / fname
        if local.exists():
            print(f"[fetch] already present: {fname}  "
                  f"({local.stat().st_size / 1024 / 1024:.1f} MB)")
            continue
        print(f"[fetch] downloading {fname}...")
        path = hf_hub_download(
            repo_id=REPO, filename=fname, repo_type="dataset",
            token=token, local_dir=str(TARGET),
        )
        print(f"[fetch] {fname} → {path}")

    print(f"[fetch] all files in {TARGET}")


if __name__ == "__main__":
    main()
