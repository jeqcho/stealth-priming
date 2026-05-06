"""Stage 3: 100 scenarios x 100 sys prompts x N=25 rollouts.

Run twice: once with thinking OFF, once with thinking ON. Each run lands
in its own outputs/rollouts_<split>_<ts>/ directory.

Output layout:
  outputs/rollouts_thinkOFF_<ts>/
    ├── manifest.json
    ├── all.jsonl                # one row per (entry_id, cell_id, sample_idx)
    └── (judge labels are added later by 04_judge_rollouts.py to all.judged.jsonl)
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

from helpers import build_rollout_messages, strip_thinking, cell_id
from vllm_runner import generate_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = PROJECT_ROOT / "data" / "scenarios.json"
SYSPROMPTS_PATH = PROJECT_ROOT / "data" / "sys_prompts.json"
OUTPUTS = PROJECT_ROOT / "outputs"
LOGS = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--thinking", required=True, choices=["off", "on"])
    p.add_argument("--n-samples", type=int, default=25)
    p.add_argument("--max-tokens-off", type=int, default=300)
    p.add_argument("--max-tokens-on", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--chunk-size", type=int, default=100000,
                   help="Cells per vLLM batch call. Default ~no chunking: "
                       "spawning workers takes ~140s each, so a single "
                       "10K-prompt batch is much faster than 4 chunks.")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--resume-dir", type=str, default=None,
                   help="Path to a previously-started outputs/rollouts_*/ "
                        "directory. If set, resumes from worker shard "
                        "JSONLs in the corresponding logs/*_tmp/ dirs; "
                        "skips already-done (entry_id, cell_id, sample) "
                        "tuples. Lets a killed run pick up where it left "
                        "off.")
    p.add_argument("--worker-batch-size", type=int, default=50,
                   help="Mini-batch size inside each worker; lower = "
                        "less work lost on kill (default 50)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    thinking = args.thinking == "on"
    name = "thinkON" if thinking else "thinkOFF"
    max_tokens = args.max_tokens_on if thinking else args.max_tokens_off

    scenarios = json.loads(SCENARIOS_PATH.read_text())
    sys_prompts_data = json.loads(SYSPROMPTS_PATH.read_text())
    sys_prompts = sys_prompts_data["prompts"]
    assert len(scenarios) == 100, f"expected 100 scenarios, got {len(scenarios)}"
    assert len(sys_prompts) == 100, f"expected 100 sys prompts, got {len(sys_prompts)}"

    if args.resume_dir:
        out_dir = Path(args.resume_dir)
        if not out_dir.exists():
            raise SystemExit(f"--resume-dir does not exist: {out_dir}")
        # Recover the original ts from the dir name if possible.
        try:
            ts = out_dir.name.split("_", 2)[-1]
        except Exception:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        resume = True
        print(f"[rollouts {name}] RESUMING from {out_dir}")
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_dir = (Path(args.out_dir) if args.out_dir
                   else OUTPUTS / f"rollouts_{name}_{ts}")
        out_dir.mkdir(parents=True, exist_ok=True)
        resume = False
    print(f"[rollouts {name}] out_dir={out_dir}, max_tokens={max_tokens}, resume={resume}")

    manifest = {
        "ts": ts,
        "thinking": thinking,
        "n_scenarios": len(scenarios),
        "n_sys_prompts": len(sys_prompts),
        "n_samples": args.n_samples,
        "max_tokens": max_tokens,
        "max_model_len": args.max_model_len,
        "chunk_size": args.chunk_size,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Build the full (scenario, sys_prompt) cross-product
    all_cells = []
    for s in scenarios:
        for sp in sys_prompts:
            msgs = build_rollout_messages(sp["text"], s["prompt"])
            all_cells.append({
                "messages": msgs,
                "entry_id": s["entry_id"],
                "options": s["options"],
                "prompt": s["prompt"],
                "cell_id": sp["cell_id"],
            })
    print(f"[rollouts {name}] total cells: {len(all_cells)} "
          f"(x n={args.n_samples} samples)")

    # Process in chunks to bound peak memory
    out_file = out_dir / "all.jsonl"
    n_done = 0
    t_start = time.time()
    with open(out_file, "w") as fout:
        for chunk_start in range(0, len(all_cells), args.chunk_size):
            chunk = all_cells[chunk_start: chunk_start + args.chunk_size]
            print(f"[rollouts {name}] chunk {chunk_start}..{chunk_start+len(chunk)} "
                  f"({len(chunk)} cells)")
            t0 = time.time()
            samples = generate_batch(
                items=[{"messages": c["messages"]} for c in chunk],
                thinking=thinking,
                max_tokens=max_tokens,
                n_samples=args.n_samples,
                temperature=1.0,
                top_p=0.95,
                seed=300 + chunk_start,
                work_dir=LOGS / f"rollouts_{name}_{ts}_chunk_{chunk_start}_tmp",
                max_model_len=args.max_model_len,
                resume=resume,
                worker_batch_size=args.worker_batch_size,
                keep_tmp=True,  # keep shards so a future resume can pick up
            )
            dt = time.time() - t0
            n_chunk_total = sum(len(s) for s in samples)
            print(f"[rollouts {name}]   chunk done in {dt:.1f}s "
                  f"({n_chunk_total/max(dt,1e-3):.1f}/s)")

            for cell, raw_list in zip(chunk, samples):
                for k, raw in enumerate(raw_list):
                    row = {
                        "entry_id": cell["entry_id"],
                        "cell_id": cell["cell_id"],
                        "sample_idx": k,
                        "options": cell["options"],
                        "prompt": cell["prompt"],
                        "raw": raw,
                        "answer": strip_thinking(raw),
                        "thinking": thinking,
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_done += 1
            fout.flush()

    dt_total = time.time() - t_start
    print(f"[rollouts {name}] all done: {n_done} rollouts in {dt_total:.1f}s "
          f"({n_done/max(dt_total,1e-3):.1f}/s)")
    print(f"[rollouts {name}] wrote {out_file}")

    # Sentinel for the judge tmux
    (out_dir / ".rollouts_done").write_text(ts)


if __name__ == "__main__":
    main()
