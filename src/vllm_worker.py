"""Single-GPU vLLM worker — resumable mini-batch version.

Reads a JSONL of prompts from --in-file, generates --n-samples each via
vLLM, writes a JSONL of {orig_idx, samples} to --out-file.

Each input row: {"orig_idx": int, "messages": list[{"role", "content"}]}.

The chat template is applied here (with `enable_thinking={thinking}`) so
the worker is responsible for the full text rendering. Output `samples`
is a list of strings (length = n_samples), each the raw generation
including any <think>...</think> block.

Resume semantics:
  - On startup, if --out-file already exists, the worker reads it and
    builds a set of orig_idx values whose generations are already
    persisted. Those prompts are skipped.
  - Generation is broken into mini-batches of --batch-size prompts.
    After each mini-batch the worker appends its rows to --out-file
    and fsyncs. A SIGTERM/SIGINT mid-run loses at most one
    mini-batch worth of work per worker.

Environment: caller sets CUDA_VISIBLE_DEVICES=<i> so this worker sees
only one GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-file", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--model", default="google/gemma-4-31B-it")
    p.add_argument("--thinking", required=True, choices=["true", "false"])
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=50,
                   help="Mini-batch size for incremental writes. Lower "
                        "= less work lost on kill, but more LLM.generate "
                        "overhead. Default 50 prompts per mini-batch.")
    return p.parse_args()


def _load_completed(out_path: Path) -> set[int]:
    """Read the existing output JSONL and return the set of orig_idx
    values already persisted. Tolerates trailing partial lines (from a
    SIGKILL mid-write)."""
    if not out_path.exists():
        return set()
    done: set[int] = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                # Trailing garbage from an interrupted write.
                continue
            if "orig_idx" in d and "samples" in d:
                done.add(int(d["orig_idx"]))
    return done


def main() -> None:
    args = parse_args()
    thinking = args.thinking == "true"

    # Defer heavy imports until after CUDA_VISIBLE_DEVICES is set.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"[worker] loading model {args.model} on visible GPU "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}). "
          f"thinking={thinking}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Read VLLM_GDN_PREFILL_BACKEND to override flashinfer for hybrid
    # GDN-attention models (e.g. Qwen3-Next, Qwen3.6) when flashinfer's
    # JIT compile fails on the local CUDA toolkit. Set =triton to fall
    # back to a pure-Python/Triton kernel.
    additional_config = {}
    gdn_backend = os.environ.get("VLLM_GDN_PREFILL_BACKEND")
    if gdn_backend:
        additional_config["gdn_prefill_backend"] = gdn_backend
        print(f"[worker] gdn_prefill_backend={gdn_backend} (via env var)",
              flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=False,
        trust_remote_code=False,
        **({"additional_config": additional_config}
           if additional_config else {}),
    )
    print(f"[worker] model loaded in {time.time()-t0:.1f}s", flush=True)

    # Read input prompts.
    items: list[dict] = []
    with open(args.in_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if not items:
        print("[worker] empty input file; writing empty output", flush=True)
        Path(args.out_file).touch()
        return

    # Resume: drop items whose orig_idx is already in the output file.
    out_path = Path(args.out_file)
    done = _load_completed(out_path)
    remaining = [it for it in items if it["orig_idx"] not in done]
    print(f"[worker] resume: {len(done)}/{len(items)} already done, "
          f"{len(remaining)} remaining", flush=True)
    if not remaining:
        print(f"[worker] nothing to do, exiting", flush=True)
        return

    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        skip_special_tokens=False,  # preserve <|channel>...<channel|> for thinking strip
    )

    # Process in mini-batches with incremental appends.
    batch_size = max(1, int(args.batch_size))
    n_total = len(remaining)
    n_done_cum = 0
    t_total = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fout:
        for batch_start in range(0, n_total, batch_size):
            batch = remaining[batch_start: batch_start + batch_size]
            # Apply chat template.
            formatted: list[str] = []
            for item in batch:
                msgs = item["messages"]
                try:
                    text = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=thinking,
                    )
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                formatted.append(text)

            t_batch = time.time()
            outputs = llm.generate(formatted, sp)
            dt_batch = time.time() - t_batch

            for item, out in zip(batch, outputs):
                samples = [o.text for o in out.outputs]
                fout.write(json.dumps({
                    "orig_idx": item["orig_idx"],
                    "samples": samples,
                }) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

            n_done_cum += len(batch)
            n_out_tokens = sum(len(o.token_ids) for r in outputs for o in r.outputs)
            print(f"[worker] batch {batch_start}..{batch_start+len(batch)} "
                  f"({n_done_cum}/{n_total} cum) — {dt_batch:.1f}s, "
                  f"{n_out_tokens} out toks "
                  f"({n_out_tokens/max(dt_batch, 1e-3):.0f} tok/s)",
                  flush=True)

    dt_total = time.time() - t_total
    print(f"[worker] all {n_total} new prompts done in {dt_total:.1f}s",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
