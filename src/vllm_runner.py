"""DP=N driver for vLLM workers.

Spawns one ``vllm_worker.py`` subprocess per visible GPU, shards the
prompt list round-robin across workers, waits for completion, merges
results in original order.

Usage:
    from vllm_runner import generate_batch
    samples = generate_batch(
        items=[{"messages": [...], }, ...],
        thinking=False, max_tokens=300, n_samples=25,
        work_dir="outputs/run_xxx/_vllm_tmp",
    )
    # samples[i] is a list[str] of length n_samples for items[i].
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_MODEL = "google/gemma-4-31B-it"
WORKER_PATH = Path(__file__).resolve().parent / "vllm_worker.py"


def _detect_n_gpus() -> int:
    """Number of GPUs we can dispatch workers on.

    Prefers `CUDA_VISIBLE_DEVICES` if set in our env; otherwise probes
    via nvidia-smi.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return len([x for x in cvd.split(",") if x.strip()])
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        return len([line for line in out.strip().splitlines() if line.strip()])
    except Exception:
        return 1


def _shard_round_robin(items: Sequence[dict], n_shards: int) -> list[list[dict]]:
    """Round-robin shard so each worker sees a representative mix."""
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    for i, item in enumerate(items):
        shards[i % n_shards].append({"orig_idx": i, **item})
    return shards


def generate_batch(
    *,
    items: Sequence[dict],
    thinking: bool,
    max_tokens: int,
    n_samples: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 0,
    model: str = DEFAULT_MODEL,
    work_dir: str | Path,
    n_gpus: int | None = None,
    gpu_mem_util: float = 0.85,
    max_model_len: int = 4096,
    keep_tmp: bool = False,
    resume: bool = False,
    worker_batch_size: int = 50,
) -> list[list[str]]:
    """Run vLLM generation across multiple GPUs.

    Each item in `items` is a dict with at least a key "messages"
    (list of {"role", "content"}). Other keys are ignored by the
    worker. Returns a list parallel to `items`, each entry a
    list[str] of length `n_samples`.
    """
    if n_gpus is None:
        n_gpus = _detect_n_gpus()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if not items:
        return []

    shards = _shard_round_robin(items, n_gpus)
    nonempty_shards = [(g, s) for g, s in enumerate(shards) if s]

    # Write input shards.
    in_files = {}
    out_files = {}
    for gpu_id, shard in nonempty_shards:
        in_path = work_dir / f"shard_{gpu_id}_in.jsonl"
        out_path = work_dir / f"shard_{gpu_id}_out.jsonl"
        with open(in_path, "w") as f:
            for row in shard:
                f.write(json.dumps(row) + "\n")
        in_files[gpu_id] = in_path
        out_files[gpu_id] = out_path
        # Pre-clear any stale output, UNLESS we're resuming, in which
        # case the worker reads existing output and skips already-done
        # orig_idx values.
        if not resume and out_path.exists():
            out_path.unlink()

    # Spawn workers.
    procs: list[tuple[int, subprocess.Popen]] = []
    log_handles: list = []
    for gpu_id, _ in nonempty_shards:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Quiet down vLLM's noisy progress bar in subprocess output.
        env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Disable FP8 path -- model is bf16, and the deep_gemm package
        # is non-trivial to install (CUTLASS submodule build).
        env.setdefault("VLLM_USE_DEEP_GEMM", "0")
        env.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")
        log_path = work_dir / f"shard_{gpu_id}.log"
        log_f = open(log_path, "w")
        log_handles.append(log_f)
        cmd = [
            sys.executable, "-u", str(WORKER_PATH),
            "--in-file", str(in_files[gpu_id]),
            "--out-file", str(out_files[gpu_id]),
            "--model", model,
            "--thinking", "true" if thinking else "false",
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--top-p", str(top_p),
            "--n-samples", str(n_samples),
            "--gpu-mem-util", str(gpu_mem_util),
            "--max-model-len", str(max_model_len),
            "--seed", str(seed + gpu_id),  # different seed per worker
            "--batch-size", str(worker_batch_size),
        ]
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((gpu_id, p))

    # Wait for workers, fail-fast on first non-zero exit.
    t0 = time.time()
    failures: list[tuple[int, int]] = []
    for gpu_id, p in procs:
        rc = p.wait()
        if rc != 0:
            failures.append((gpu_id, rc))
    for h in log_handles:
        h.close()
    dt = time.time() - t0
    print(f"[runner] all workers finished in {dt:.1f}s", flush=True)
    if failures:
        for gpu_id, rc in failures:
            log_path = work_dir / f"shard_{gpu_id}.log"
            tail = "\n".join(log_path.read_text().splitlines()[-40:])
            print(f"[runner] shard {gpu_id} exited {rc}; log tail:\n{tail}",
                  flush=True)
        raise RuntimeError(f"{len(failures)} worker(s) failed")

    # Collect results in original order.
    results: list[list[str] | None] = [None] * len(items)
    for gpu_id, _ in nonempty_shards:
        with open(out_files[gpu_id]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                results[d["orig_idx"]] = d["samples"]

    missing = [i for i, r in enumerate(results) if r is None]
    if missing:
        raise RuntimeError(f"{len(missing)} items missing from worker output "
                           f"(first few: {missing[:5]})")

    if not keep_tmp:
        # Clean up shard files but keep the .log files for debugging.
        for gpu_id, _ in nonempty_shards:
            in_files[gpu_id].unlink(missing_ok=True)
            out_files[gpu_id].unlink(missing_ok=True)

    return results  # type: ignore[return-value]
