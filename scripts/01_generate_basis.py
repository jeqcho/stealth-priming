"""Stage 1: Gemma 4 31B (thinking ON) writes 100 system prompts.

For each (P, T, E) cell from helpers.all_cells(), Gemma writes one
generic system prompt. Output saved to data/sys_prompts.json with the
schema documented in PLAN.md §3.

Dedupe pass: if any two cells produce identical text (post-strip,
case-insensitive), regenerate the duplicate at higher temperature with
a different seed.
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

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from helpers import (
    PERSONAS, TONES, EMPHASES, all_cells, cell_id,
    build_sys_prompt_gen_msgs, strip_thinking,
)
from vllm_runner import generate_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
LOGS = PROJECT_ROOT / "logs"
DATA.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

OUT_PATH = DATA / "sys_prompts.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Generation budget per cell (incl. thinking trace)")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--retry-temperature", type=float, default=1.1)
    p.add_argument("--max-model-len", type=int, default=4096)
    return p.parse_args()


def build_items() -> list[dict]:
    items = []
    for pi, ti, ei in all_cells():
        msgs = build_sys_prompt_gen_msgs(PERSONAS[pi], TONES[ti], EMPHASES[ei])
        items.append({"messages": msgs, "pi": pi, "ti": ti, "ei": ei})
    return items


def main() -> None:
    args = parse_args()
    cells = all_cells()
    items = build_items()
    assert len(items) == 100

    print(f"[gen] generating 100 sys prompts (thinking ON, "
          f"max_tokens={args.max_tokens}, temp={args.temperature})")
    t0 = time.time()
    work_dir = LOGS / "sysprompt_gen_tmp"
    samples = generate_batch(
        items=[{"messages": it["messages"]} for it in items],
        thinking=True,
        max_tokens=args.max_tokens,
        n_samples=1,
        temperature=args.temperature,
        top_p=0.95,
        seed=0,
        work_dir=work_dir,
        max_model_len=args.max_model_len,
    )
    dt = time.time() - t0
    print(f"[gen] vLLM batch done in {dt:.1f}s")

    # Strip thinking traces, store
    prompts_out: list[dict] = []
    for it, sample_list in zip(items, samples):
        raw = sample_list[0]
        text = strip_thinking(raw)
        # Defensive: collapse internal whitespace runs but preserve paragraphs
        text = "\n".join(line.strip() for line in text.splitlines())
        text = "\n\n".join(p.strip() for p in text.split("\n\n") if p.strip())
        prompts_out.append({
            "persona_idx": it["pi"],
            "tone_idx": it["ti"],
            "emphasis_idx": it["ei"],
            "cell_id": cell_id(it["pi"], it["ti"], it["ei"]),
            "text": text,
            "n_chars": len(text),
            "n_words": len(text.split()),
            "raw_with_thinking": raw,
        })

    # Dedupe pass
    text_to_first_idx: dict[str, int] = {}
    dupes: list[int] = []
    for idx, p in enumerate(prompts_out):
        key = " ".join(p["text"].lower().split())
        if key in text_to_first_idx:
            dupes.append(idx)
        else:
            text_to_first_idx[key] = idx

    print(f"[gen] {len(dupes)} duplicate cells; regenerating at "
          f"temperature={args.retry_temperature}")
    if dupes:
        retry_items = [items[i] for i in dupes]
        retry_samples = generate_batch(
            items=[{"messages": it["messages"]} for it in retry_items],
            thinking=True,
            max_tokens=args.max_tokens,
            n_samples=1,
            temperature=args.retry_temperature,
            top_p=0.98,
            seed=42,
            work_dir=LOGS / "sysprompt_regen_tmp",
            max_model_len=args.max_model_len,
        )
        for orig_idx, sample_list in zip(dupes, retry_samples):
            raw = sample_list[0]
            text = strip_thinking(raw)
            text = "\n".join(line.strip() for line in text.splitlines())
            text = "\n\n".join(p.strip() for p in text.split("\n\n") if p.strip())
            prompts_out[orig_idx]["text"] = text
            prompts_out[orig_idx]["n_chars"] = len(text)
            prompts_out[orig_idx]["n_words"] = len(text.split())
            prompts_out[orig_idx]["raw_with_thinking"] = raw
            prompts_out[orig_idx]["regenerated"] = True

    # Final state
    out = {
        "dimensions": {
            "persona": [
                {"code": p.code, "name": p.name, "description": p.description}
                for p in PERSONAS
            ],
            "tone": [
                {"code": t.code, "name": t.name, "description": t.description}
                for t in TONES
            ],
            "emphasis": [
                {"code": e.code, "name": e.name, "description": e.description}
                for e in EMPHASES
            ],
        },
        "prompts": prompts_out,
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": "google/gemma-4-31B-it",
            "thinking": True,
            "temperature": args.temperature,
            "retry_temperature": args.retry_temperature,
            "n_dupes_regenerated": len(dupes),
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[gen] wrote {OUT_PATH}")

    # Print sanity stats
    n_words = [p["n_words"] for p in prompts_out]
    print(f"[gen] word count: min={min(n_words)} max={max(n_words)} "
          f"mean={sum(n_words)/len(n_words):.0f}")
    print(f"[gen] sample (cell {prompts_out[0]['cell_id']}):\n"
          f"---\n{prompts_out[0]['text'][:500]}\n---")


if __name__ == "__main__":
    main()
