"""Helpers for exp 32 — persona-vector cell selector.

- find_suffix_range: locate cell-text suffix tokens in a chat-template-rendered prompt
- ridge_solve: closed-form ridge regression
- cross_cov: ridge-free fallback
- kfold_oof_scores: 5-fold OOF projection scoring per scenario
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SuffixRange:
    """[start, end) token range for cell-text suffix in a chat-template output."""
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


def find_suffix_range(
    rendered: str,
    cell_text: str,
    tokenizer,
) -> SuffixRange:
    """Tokenize `rendered` and return the [start, end) range covering cell_text.

    Uses character offsets — robust to BPE merges across boundaries.
    """
    # Find character offset of cell_text in the rendered chat string.
    char_start = rendered.index(cell_text)
    char_end = char_start + len(cell_text)

    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    suffix_start = None
    suffix_end = None
    for tok_idx, (s, e) in enumerate(offsets):
        if s >= char_start and suffix_start is None:
            suffix_start = tok_idx
        if e <= char_end:
            suffix_end = tok_idx + 1

    if suffix_start is None or suffix_end is None:
        raise RuntimeError(
            f"Failed to locate cell text in rendered chat. "
            f"chars=[{char_start},{char_end}), offsets[0..]={offsets[:3]}, ..."
        )
    if suffix_end <= suffix_start:
        raise RuntimeError(
            f"Suffix range degenerate: [{suffix_start}, {suffix_end})"
        )
    return SuffixRange(suffix_start, suffix_end)


def ridge_solve(H: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge regression. H shape (n, d), y shape (n,). Returns w shape (d,).

    Both H and y should be mean-centered before calling.
    """
    n, d = H.shape
    if d <= n:
        # Primal form (faster when d <= n)
        A = H.T @ H + lam * np.eye(d, dtype=H.dtype)
        b = H.T @ y
        w = np.linalg.solve(A, b)
    else:
        # Dual form (d > n): w = H^T (HH^T + lam I)^-1 y
        K = H @ H.T + lam * np.eye(n, dtype=H.dtype)
        alpha = np.linalg.solve(K, y)
        w = H.T @ alpha
    return w


def cross_cov(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """No-inversion fallback: w ∝ Σ (h_i - h̄)(y_i - ȳ).

    Equivalent to ridge as λ → ∞ (up to scale).
    """
    Hc = H - H.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    return Hc.T @ yc


def kfold_oof_scores(
    H: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    lam: float = 1.0,
    seed: int = 0,
    method: str = "ridge",
) -> tuple[np.ndarray, np.ndarray]:
    """K-fold OOF projection scores.

    Args:
        H: (n_cells, hidden_dim) cell embeddings
        y: (n_cells,) rate_a labels
        n_folds: usually 5
        lam: ridge regularization
        seed: rng for fold assignment
        method: "ridge" or "cross_cov"

    Returns:
        proj_oof: (n_cells,) per-cell OOF projection scores
        fold_idx: (n_cells,) which fold each cell was held out in
    """
    n = H.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_idx = np.zeros(n, dtype=int)
    for k in range(n_folds):
        fold_idx[perm[k::n_folds]] = k

    proj = np.zeros(n, dtype=np.float32)
    for k in range(n_folds):
        train_mask = fold_idx != k
        test_mask = fold_idx == k
        H_train = H[train_mask]
        y_train = y[train_mask]
        H_train_centered = H_train - H_train.mean(axis=0, keepdims=True)
        y_train_centered = y_train - y_train.mean()

        if method == "ridge":
            w = ridge_solve(H_train_centered.astype(np.float64),
                            y_train_centered.astype(np.float64),
                            lam=lam)
        elif method == "cross_cov":
            w = cross_cov(H_train_centered.astype(np.float64),
                          y_train_centered.astype(np.float64))
        else:
            raise ValueError(f"Unknown method: {method}")

        norm = np.linalg.norm(w)
        if norm < 1e-12:
            proj[test_mask] = 0
        else:
            proj[test_mask] = (H[test_mask].astype(np.float64) @ w / norm).astype(np.float32)

    return proj, fold_idx
