"""
plot_scalability_vldb.py — Publication-oriented scalability figures (all K, per metric).

Reads *_pharma_scalability.json (or *_pharma_combined_scalability.json) produced by
run_scalability_pharma.py and writes one image per metric, showing every K value.

Notes for the paper:
  - MAP, Score_AP, and Mean_Rank are *not* indexed by K in our JSON; they are
    rank-based summaries over the (restricted) pool. Use the dedicated summary figure.
  - When every query has exactly |GT| = G and you report at K = G, P@K = R@K = F1@K
    by definition (same numerator G·hits and denominators G). Other K values separate
    precision vs recall behavior and are worth showing.

Usage:
  python plot_scalability_vldb.py --results_dir Ex-2_Scalibility_Results/scalability
  python plot_scalability_vldb.py --styles paper_v1 paper_v2 paper_v3 --metric_type macro \\
      --results_dir Ex-2_Scalibility_Results/scalability
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _size_label(size: int) -> str:
    return "Full" if size == 0 else str(size)


def _size_sort_key(size: int) -> int:
    return 999_999_999 if size == 0 else size


def load_scalability_jsons(results_dir: str) -> Tuple[Dict[str, Dict[int, Any]], List[int], List[int]]:
    """
    Returns:
      all_data: {model_name: {pool_size_int: {macro: ..., micro: ...}}}
      sizes: sorted pool sizes present
      k_values: sorted K values from macro.per_k
    """
    all_data: Dict[str, Dict[int, Any]] = {}
    sizes_set = set()
    k_set = set()

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        if "scalability" not in fname or "full_scores" in fname:
            continue
        # Match both pharma-runner and unified-runner output patterns:
        #   Pharma:  {Model}_pharma{suffix}_scalability.json
        #   Unified: {dataset}_{task}_{split}_{Model}{suffix}_scalability.json
        # In both cases the model name is the segment right before _scalability.json
        # (after stripping an optional suffix like _combined).
        m = re.match(r"^(.+)_scalability\.json$", fname)
        if not m:
            continue
        prefix = m.group(1)  # e.g. "LOKI_pharma" or "pharma_DOC_TO_TABLE_test_LOKI"
        # Model name is the last '_'-delimited token (strip known suffixes first)
        prefix_clean = re.sub(r"_combined$", "", prefix)
        model_name = prefix_clean.rsplit("_", 1)[-1]
        if not model_name:
            continue
        path = os.path.join(results_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        parsed: Dict[int, Any] = {}
        for sk, block in raw.items():
            si = int(sk)
            parsed[si] = block
            sizes_set.add(si)
            per_k = block.get("macro", {}).get("per_k") or {}
            for kk in per_k.keys():
                k_set.add(int(kk))
        all_data[model_name] = parsed

    if not all_data:
        raise FileNotFoundError(
            f"No *_scalability.json files found under {results_dir!r}"
        )

    sizes = sorted(sizes_set, key=_size_sort_key)
    k_values = sorted(k_set)
    return all_data, sizes, k_values


def _get_k_val(results: Dict[str, Any], k: int, metric_key: str) -> float:
    per_k = results.get("per_k", {})
    kd = per_k.get(k, per_k.get(str(k), {}))
    v = kd.get(metric_key, 0.0)
    if v == float("inf"):
        return float("nan")
    return float(v)


# ---------------------------------------------------------------------------
# Style: facet by K (one subplot per K, x = pool size, lines = models)
# ---------------------------------------------------------------------------

PALETTE = ["#2ec4b6", "#e74c3c", "#ffb703", "#8e44ad", "#3498db", "#f39c12", "#1abc9c", "#d35400"]
MARKERS = ["o", "s", "D", "^", "v", "p", "*", "h"]


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_facet_by_k(
    all_data: Dict[str, Dict[int, Any]],
    sizes: List[int],
    k_values: List[int],
    metric_key: str,
    metric_type: str,
    output_path: str,
    title_suffix: str = "",
):
    """One figure: grid of subplots, each subplot = fixed K, x = pool, lines = models."""
    plt = _setup_matplotlib()

    models = list(all_data.keys())
    sorted_sizes = sorted(sizes, key=_size_sort_key)
    x_labels = [_size_label(s) for s in sorted_sizes]
    x_pos = np.arange(len(sorted_sizes))
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    n_k = len(k_values)
    n_cols = min(3, n_k)
    n_rows = int(np.ceil(n_k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), squeeze=False)

    pretty = metric_key.replace("@K", "@k").replace("_", " ")
    fig.suptitle(
        f"Scalability — {metric_type.capitalize()} {pretty} (all K){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    for idx, k in enumerate(k_values):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r][c]
        for mi, model in enumerate(models):
            ys = []
            for sz in sorted_sizes:
                block = all_data[model].get(sz, {}).get(metric_type)
                if block is None:
                    ys.append(np.nan)
                else:
                    ys.append(_get_k_val(block, k, metric_key))
            ax.plot(
                x_pos,
                ys,
                marker=MARKERS[mi % len(MARKERS)],
                color=colors[model],
                label=model,
                linewidth=1.8,
                markersize=6,
            )
        ax.set_title(f"K = {k}", fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Candidate pool size", fontsize=9)

    # Hide empty axes
    for idx in range(len(k_values), n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r][c].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(models)), fontsize=9, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[VLDB-PLOT] {output_path}")


# ---------------------------------------------------------------------------
# Style: facet by pool (one subplot per pool, x = K, lines = models)
# ---------------------------------------------------------------------------

def plot_facet_by_pool(
    all_data: Dict[str, Dict[int, Any]],
    sizes: List[int],
    k_values: List[int],
    metric_key: str,
    metric_type: str,
    output_path: str,
    title_suffix: str = "",
):
    plt = _setup_matplotlib()
    models = list(all_data.keys())
    sorted_sizes = sorted(sizes, key=_size_sort_key)
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    x_pos = np.arange(len(k_values))

    n_p = len(sorted_sizes)
    n_cols = min(3, n_p)
    n_rows = int(np.ceil(n_p / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.6 * n_rows), squeeze=False)

    pretty = metric_key.replace("@K", "@k").replace("_", " ")
    fig.suptitle(
        f"Scalability — {metric_type.capitalize()} {pretty} vs K{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    for idx, sz in enumerate(sorted_sizes):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r][c]
        for mi, model in enumerate(models):
            ys = []
            for k in k_values:
                block = all_data[model].get(sz, {}).get(metric_type)
                if block is None:
                    ys.append(np.nan)
                else:
                    ys.append(_get_k_val(block, k, metric_key))
            ax.plot(
                x_pos,
                ys,
                marker=MARKERS[mi % len(MARKERS)],
                color=colors[model],
                label=model,
                linewidth=1.8,
                markersize=6,
            )
        ax.set_title(f"Pool = {_size_label(sz)}", fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(k) for k in k_values], fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("K", fontsize=9)

    for idx in range(len(sorted_sizes), n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r][c].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(models)), fontsize=9, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[VLDB-PLOT] {output_path}")


# ---------------------------------------------------------------------------
# Heatmap: rows = models, cols = K, one panel per pool size
# ---------------------------------------------------------------------------

def plot_heatmaps_by_pool(
    all_data: Dict[str, Dict[int, Any]],
    sizes: List[int],
    k_values: List[int],
    metric_key: str,
    metric_type: str,
    output_path: str,
    title_suffix: str = "",
):
    plt = _setup_matplotlib()
    models = list(all_data.keys())
    sorted_sizes = sorted(sizes, key=_size_sort_key)

    n_p = len(sorted_sizes)
    n_cols = min(3, n_p)
    n_rows = int(np.ceil(n_p / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.8 * n_rows), squeeze=False)

    pretty = metric_key.replace("@K", "@k").replace("_", " ")
    fig.suptitle(
        f"Heatmap — {metric_type.capitalize()} {pretty} (model × K){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    vmin, vmax = 0.0, 1.0
    im = None
    for idx, sz in enumerate(sorted_sizes):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r][c]
        mat = np.zeros((len(models), len(k_values)))
        for mi, model in enumerate(models):
            for ki, k in enumerate(k_values):
                block = all_data[model].get(sz, {}).get(metric_type)
                if block is None:
                    mat[mi, ki] = np.nan
                else:
                    mat[mi, ki] = _get_k_val(block, k, metric_key)
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(np.arange(len(k_values)))
        ax.set_xticklabels([str(k) for k in k_values], fontsize=8)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(f"Pool = {_size_label(sz)}", fontsize=10, fontweight="bold")
        ax.set_xlabel("K", fontsize=9)

    used_axes = []
    for idx in range(len(sorted_sizes), n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r][c].set_visible(False)
    for idx in range(len(sorted_sizes)):
        r, c = idx // n_cols, idx % n_cols
        used_axes.append(axes[r][c])

    if im is not None and used_axes:
        fig.colorbar(im, ax=used_axes, shrink=0.85, label=metric_key, fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[VLDB-PLOT] {output_path}")


# ---------------------------------------------------------------------------
# Global metrics (no K): MAP, Score_AP, Mean_Rank
# ---------------------------------------------------------------------------

def plot_global_rank_metrics(
    all_data: Dict[str, Dict[int, Any]],
    sizes: List[int],
    metric_type: str,
    output_path: str,
    title_suffix: str = "",
):
    plt = _setup_matplotlib()
    models = list(all_data.keys())
    sorted_sizes = sorted(sizes, key=_size_sort_key)
    x_labels = [_size_label(s) for s in sorted_sizes]
    x_pos = np.arange(len(sorted_sizes))
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    keys = [("MAP", "MAP", False), ("Score_AP", "Score AP", False), ("Mean_Rank", "Mean rank (lower is better)", True)]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    fig.suptitle(
        f"Scalability — {metric_type.capitalize()} global rank metrics (not @K){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (json_key, disp, invert) in zip(axes, keys):
        for mi, model in enumerate(models):
            ys = []
            for sz in sorted_sizes:
                block = all_data[model].get(sz, {}).get(metric_type)
                if block is None:
                    ys.append(np.nan)
                else:
                    v = block.get(json_key, 0.0)
                    if v == float("inf"):
                        ys.append(np.nan)
                    else:
                        ys.append(float(v))
            ax.plot(
                x_pos,
                ys,
                marker=MARKERS[mi % len(MARKERS)],
                color=colors[model],
                label=model,
                linewidth=1.8,
                markersize=6,
            )
        ax.set_title(disp, fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Candidate pool size", fontsize=9)
        if invert:
            ax.invert_yaxis()
    axes[0].legend(fontsize=8, loc="best")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[VLDB-PLOT] {output_path}")


# ---------------------------------------------------------------------------
# Timing comparison: simulated_time_sec and cumulative_time_sec vs pool size
# ---------------------------------------------------------------------------

def plot_timing_comparison(
    all_data: Dict[str, Dict[int, Any]],
    sizes: List[int],
    output_path: str,
    title_suffix: str = "",
):
    """
    Two-panel timing figure:
      Left  — Line chart: simulated_time_sec vs candidate pool size (log y-scale).
              Each line is one model; pool=0 shown as "Full" on the far right.
      Right — Grouped bar chart: same data expressed in minutes, allowing direct
              model-vs-model comparison at each pool size.
    """
    plt = _setup_matplotlib()
    import matplotlib.ticker as mticker

    models = list(all_data.keys())
    sorted_sizes = sorted(sizes, key=_size_sort_key)
    x_labels = [_size_label(s) for s in sorted_sizes]
    x_pos = np.arange(len(sorted_sizes), dtype=float)
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    # Collect simulated_time_sec from macro block (pool-size agnostic)
    time_data: Dict[str, List[float]] = {m: [] for m in models}
    for model in models:
        for sz in sorted_sizes:
            mac = all_data[model].get(sz, {}).get("macro", {})
            v = mac.get("simulated_time_sec", np.nan)
            time_data[model].append(float(v) if v is not None else np.nan)

    fig, (ax_line, ax_bar) = plt.subplots(1, 2, figsize=(14.0, 5.0))
    fig.suptitle(
        f"Scalability — Simulated inference time per candidate pool size{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    # --- Left: line chart (log y) ---
    for mi, model in enumerate(models):
        ys = time_data[model]
        ax_line.plot(
            x_pos,
            ys,
            marker=MARKERS[mi % len(MARKERS)],
            color=colors[model],
            label=model,
            linewidth=2.0,
            markersize=7,
        )
    ax_line.set_yscale("log")
    ax_line.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}s" if v >= 1 else f"{v:.2f}s"))
    ax_line.set_xticks(x_pos)
    ax_line.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=9)
    ax_line.set_xlabel("Candidate pool size", fontsize=10)
    ax_line.set_ylabel("Simulated time (seconds, log scale)", fontsize=10)
    ax_line.set_title("Time vs pool size (log scale)", fontsize=11, fontweight="bold")
    ax_line.grid(True, which="both", alpha=0.3)
    ax_line.legend(fontsize=9, loc="upper left")

    # --- Right: grouped bar chart (minutes) ---
    n_models = len(models)
    bar_width = 0.75 / max(n_models, 1)
    offsets = np.linspace(-(n_models - 1) / 2 * bar_width, (n_models - 1) / 2 * bar_width, n_models)
    for mi, model in enumerate(models):
        ys_min = [v / 60.0 for v in time_data[model]]
        ax_bar.bar(
            x_pos + offsets[mi],
            ys_min,
            width=bar_width,
            color=colors[model],
            label=model,
            alpha=0.85,
        )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=9)
    ax_bar.set_xlabel("Candidate pool size", fontsize=10)
    ax_bar.set_ylabel("Simulated time (minutes)", fontsize=10)
    ax_bar.set_title("Time per pool size (minutes)", fontsize=11, fontweight="bold")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend(fontsize=9, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[VLDB-PLOT] {output_path}")


# ---------------------------------------------------------------------------
# Paper variants: one axes per metric, scalability vs pool (three figure styles)
# ---------------------------------------------------------------------------

def _lighten_rgb(rgb: Tuple[float, float, float], toward_white: float) -> Tuple[float, float, float]:
    """toward_white in [0, 1]: 0 = unchanged, 1 = white."""
    t = max(0.0, min(1.0, toward_white))
    return tuple(c + (1.0 - c) * t for c in rgb)


def plot_paper_v1_focal_minmax_band(
    all_data: Dict[str, Dict[int, Any]],
    sizes: List[int],
    k_values: List[int],
    metric_key: str,
    metric_type: str,
    output_path: str,
    focal_k: int = 8,
    title_suffix: str = "",
    band_k_values: Optional[List[int]] = None,
):
    """
    Solid line = metric at focal K; shaded band = min–max over a chosen set of K.

    Default band uses every K present in the JSON. Narrow the set (e.g. 4,8,16) when
    you want the ribbon to match a small K grid (without K=1 / K=32 pulling the envelope).

    Note: For F1@K with |GT| = focal K, F1 often *peaks* at that K, so the focal line can
    sit on the *upper* edge of an all-K band — that is expected, not a calculation error.
    Precision@K is usually highest at small K and lowest at large K, so K=focal sits
    strictly inside an all-K band.
    """
    plt = _setup_matplotlib()

    ks_band = list(band_k_values) if band_k_values is not None else list(k_values)
    ks_band = sorted(set(ks_band))

    models = list(all_data.keys())
    sorted_sizes = sorted(sizes, key=_size_sort_key)
    x_labels = [_size_label(s) for s in sorted_sizes]
    x_pos = np.arange(len(sorted_sizes), dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    pretty = metric_key.replace("@K", "@k").replace("_", " ")
    ks_str = ", ".join(str(k) for k in ks_band)
    ax.set_title(
        f"{metric_type.capitalize()} {pretty} — solid K={focal_k}, band = min–max over K ∈ {{{ks_str}}}{title_suffix}",
        fontsize=12,
        fontweight="bold",
    )

    for mi, model in enumerate(models):
        base = PALETTE[mi % len(PALETTE)]
        mids, lows, highs = [], [], []
        for sz in sorted_sizes:
            block = all_data[model].get(sz, {}).get(metric_type)
            if block is None:
                mids.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                continue
            per_k_vals = [_get_k_val(block, k, metric_key) for k in ks_band]
            finite = [v for v in per_k_vals if np.isfinite(v)]
            if not finite:
                mids.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                continue
            fk = _get_k_val(block, focal_k, metric_key)
            mids.append(fk if np.isfinite(fk) else float(np.nanmean(finite)))
            lows.append(min(finite))
            highs.append(max(finite))

        y_m = np.asarray(mids, dtype=float)
        y_lo = np.asarray(lows, dtype=float)
        y_hi = np.asarray(highs, dtype=float)
        valid = np.isfinite(y_m) & np.isfinite(y_lo) & np.isfinite(y_hi)
        if np.any(valid):
            ax.fill_between(
                x_pos[valid],
                y_lo[valid],
                y_hi[valid],
                color=base,
                alpha=0.28,
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            x_pos,
            y_m,
            color=base,
            marker=MARKERS[mi % len(MARKERS)],
            linewidth=2.4,
            markersize=7,
            label=model,
            zorder=3,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=10)
    ax.set_xlabel("Candidate pool size", fontsize=11)
    ax.set_ylabel(pretty, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[VLDB-PLOT] {output_path}")


def plot_paper_v2_k4_8_16_family(
    all_data: Dict[str, Dict[int, Any]],
    sizes: List[int],
    k_values: List[int],
    metric_key: str,
    metric_type: str,
    output_path: str,
    title_suffix: str = "",
):
    """K=8 solid (saturated); K=4 and K=16 dashed with lighter shades of the same model color."""
    plt = _setup_matplotlib()
    import matplotlib.colors as mcolors

    want = [4, 8, 16]
    k_use = [k for k in want if k in k_values]
    if len(k_use) < 1:
        print(f"[WARN] paper_v2: none of K∈{{4,8,16}} present in results; skipping {metric_key}")
        return

    models = list(all_data.keys())
    sorted_sizes = sorted(sizes, key=_size_sort_key)
    x_labels = [_size_label(s) for s in sorted_sizes]
    x_pos = np.arange(len(sorted_sizes), dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    pretty = metric_key.replace("@K", "@k").replace("_", " ")
    ax.set_title(
        f"{metric_type.capitalize()} {pretty} — K∈{{4,8,16}}: K=8 solid, K=4/16 dashed (lighter){title_suffix}",
        fontsize=12,
        fontweight="bold",
    )

    # Shade steps: outer Ks lighter; 8 strongest
    k_style = {
        4: ("--", 0.45),
        8: ("-", 0.0),
        16: ("--", 0.28),
    }

    handles = []
    for mi, model in enumerate(models):
        base = PALETTE[mi % len(PALETTE)]
        rgb = mcolors.to_rgb(base)
        for k in k_use:
            ls, lighten = k_style.get(k, ("-", 0.15))
            col = _lighten_rgb(rgb, lighten) if lighten > 0 else base
            ys = []
            for sz in sorted_sizes:
                block = all_data[model].get(sz, {}).get(metric_type)
                if block is None:
                    ys.append(np.nan)
                else:
                    ys.append(_get_k_val(block, k, metric_key))
            lw = 2.6 if k == 8 else 1.85
            (line,) = ax.plot(
                x_pos,
                ys,
                color=col,
                linestyle=ls,
                linewidth=lw,
                marker=MARKERS[mi % len(MARKERS)] if k == 8 else "o",
                markersize=6 if k == 8 else 4,
                markevery=1 if k == 8 else 2,
                label=f"{model}  K={k}",
            )
            handles.append(line)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=10)
    ax.set_xlabel("Candidate pool size", fontsize=11)
    ax.set_ylabel(pretty, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.35)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[VLDB-PLOT] {output_path}")


def plot_paper_v3_all_k_multicolor(
    all_data: Dict[str, Dict[int, Any]],
    sizes: List[int],
    k_values: List[int],
    metric_key: str,
    metric_type: str,
    output_path: str,
    title_suffix: str = "",
):
    """Color encodes K; linestyle encodes model (all K on one axes)."""
    plt = _setup_matplotlib()
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    models = list(all_data.keys())
    sorted_sizes = sorted(sizes, key=_size_sort_key)
    x_labels = [_size_label(s) for s in sorted_sizes]
    x_pos = np.arange(len(sorted_sizes), dtype=float)
    n_k = len(k_values)
    try:
        cmap = mpl.colormaps["turbo"].resampled(max(n_k, 1))
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        cmap = cm.get_cmap("turbo", max(n_k, 2))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    pretty = metric_key.replace("@K", "@k").replace("_", " ")
    ax.set_title(
        f"{metric_type.capitalize()} {pretty} — color = K, linestyle = model{title_suffix}",
        fontsize=12,
        fontweight="bold",
    )

    model_lss = ["-", "--", "-.", (0, (5, 2, 1, 2))]

    def _k_color(ki: int) -> Any:
        if n_k <= 1:
            return cmap(0.5)
        return cmap(ki / max(n_k - 1, 1))

    for ki, k in enumerate(k_values):
        col = _k_color(ki)
        for mi, model in enumerate(models):
            ys = []
            for sz in sorted_sizes:
                block = all_data[model].get(sz, {}).get(metric_type)
                if block is None:
                    ys.append(np.nan)
                else:
                    ys.append(_get_k_val(block, k, metric_key))
            ax.plot(
                x_pos,
                ys,
                color=col,
                linestyle=model_lss[mi % len(model_lss)],
                linewidth=2.0,
                label=None,
            )

    k_handles = [
        Line2D([0], [0], color=_k_color(i), lw=3, label=f"K = {k}")
        for i, k in enumerate(k_values)
    ]
    m_handles = [
        Line2D([0], [0], color="0.2", linestyle=model_lss[mi % len(model_lss)], lw=2.2, label=model)
        for mi, model in enumerate(models)
    ]
    leg_k = ax.legend(handles=k_handles, title="K", loc="upper left", fontsize=9)
    ax.add_artist(leg_k)
    ax.legend(handles=m_handles, title="Model", loc="upper right", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=10)
    ax.set_xlabel("Candidate pool size", fontsize=11)
    ax.set_ylabel(pretty, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[VLDB-PLOT] {output_path}")


# ---------------------------------------------------------------------------
# Radar / spider chart (one per pool size): axes = K, one closed line per model
# ---------------------------------------------------------------------------

def plot_radar_by_pool(
    all_data: Dict[str, Dict[int, Any]],
    sizes: List[int],
    k_values: List[int],
    metric_key: str,
    metric_type: str,
    output_dir: str,
    suffix: str,
    title_suffix: str = "",
    max_pools_per_fig: int = 6,
):
    plt = _setup_matplotlib()
    from math import pi

    models = list(all_data.keys())
    sorted_sizes = sorted(sizes, key=_size_sort_key)
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    n_axes = len(k_values)
    angles = [pi * 2 * i / n_axes for i in range(n_axes)]
    angles += angles[:1]

    pretty = metric_key.replace("@K", "@k").replace("_", " ")
    for batch_start in range(0, len(sorted_sizes), max_pools_per_fig):
        batch = sorted_sizes[batch_start : batch_start + max_pools_per_fig]
        n_p = len(batch)
        n_cols = min(3, n_p)
        n_rows = int(np.ceil(n_p / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.6 * n_rows), subplot_kw=dict(polar=True), squeeze=False)
        fig.suptitle(
            f"Radar — {metric_type.capitalize()} {pretty}{title_suffix}",
            fontsize=14,
            fontweight="bold",
        )

        for idx, sz in enumerate(batch):
            r, c = idx // n_cols, idx % n_cols
            ax = axes[r][c]
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([f"K={k}" for k in k_values], fontsize=8)
            ax.set_ylim(0, 1)

            for mi, model in enumerate(models):
                vals = []
                for k in k_values:
                    block = all_data[model].get(sz, {}).get(metric_type)
                    if block is None:
                        vals.append(0.0)
                    else:
                        vals.append(_get_k_val(block, k, metric_key))
                vals += vals[:1]
                ax.plot(angles, vals, marker=MARKERS[mi % len(MARKERS)], color=colors[model], label=model, linewidth=1.5)
                ax.fill(angles, vals, color=colors[model], alpha=0.08)
            ax.set_title(f"Pool = {_size_label(sz)}", fontsize=10, fontweight="bold", pad=12)
            ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=7)

        for idx in range(len(batch), n_rows * n_cols):
            r, c = idx // n_cols, idx % n_cols
            fig.delaxes(axes[r][c])

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        part = f"_part{batch_start // max_pools_per_fig}" if len(sorted_sizes) > max_pools_per_fig else ""
        out = os.path.join(output_dir, f"scalability_{metric_type}_{metric_key.replace('@', 'at')}_radar{part}{suffix}.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[VLDB-PLOT] {out}")


# ---------------------------------------------------------------------------
# Bundle driver
# ---------------------------------------------------------------------------

PER_K_METRICS_MACRO = ["P@K", "R@K", "F1@K", "NDCG@K", "MRR@K", "All@K"]
PER_K_METRICS_MICRO = ["P@K", "R@K", "F1@K"]


def run_all(
    results_dir: str,
    output_dir: str,
    metric_types: Sequence[str],
    styles: Sequence[str],
    suffix: str,
    focal_k: int = 8,
    skip_global: bool = False,
    band_ks: Optional[List[int]] = None,
):
    all_data, sizes, k_values = load_scalability_jsons(results_dir)
    title_suffix = ""

    # Timing comparison — emitted once (not per metric_type)
    plot_timing_comparison(
        all_data,
        sizes,
        os.path.join(output_dir, f"scalability_timing_comparison{suffix}.png"),
        title_suffix=title_suffix,
    )

    band_ks_effective: Optional[List[int]] = band_ks
    if band_ks_effective is not None:
        band_ks_effective = sorted({k for k in band_ks_effective if k in k_values})
        dropped = set(band_ks or []) - set(band_ks_effective)
        if dropped:
            print(f"[WARN] band_ks not in results JSON, ignored: {sorted(dropped)}")
        if not band_ks_effective:
            print("[WARN] band_ks empty after filtering; paper_v1 will use all K")
            band_ks_effective = None

    for mt in metric_types:
        per_k_list = PER_K_METRICS_MACRO if mt == "macro" else PER_K_METRICS_MICRO

        out_sub = os.path.join(output_dir, mt)
        os.makedirs(out_sub, exist_ok=True)

        if not skip_global:
            plot_global_rank_metrics(
                all_data,
                sizes,
                mt,
                os.path.join(out_sub, f"scalability_{mt}_global_MAP_ScoreAP_MeanRank{suffix}.png"),
                title_suffix=title_suffix,
            )

        for mk in per_k_list:
            safe = mk.replace("@", "at").replace("/", "_")

            if "facet_k" in styles:
                plot_facet_by_k(
                    all_data,
                    sizes,
                    k_values,
                    mk,
                    mt,
                    os.path.join(out_sub, f"scalability_{mt}_{safe}_facet_by_K{suffix}.png"),
                    title_suffix=title_suffix,
                )
            if "facet_pool" in styles:
                plot_facet_by_pool(
                    all_data,
                    sizes,
                    k_values,
                    mk,
                    mt,
                    os.path.join(out_sub, f"scalability_{mt}_{safe}_facet_by_pool{suffix}.png"),
                    title_suffix=title_suffix,
                )
            if "heatmap" in styles:
                plot_heatmaps_by_pool(
                    all_data,
                    sizes,
                    k_values,
                    mk,
                    mt,
                    os.path.join(out_sub, f"scalability_{mt}_{safe}_heatmap{suffix}.png"),
                    title_suffix=title_suffix,
                )
            if "radar" in styles:
                plot_radar_by_pool(
                    all_data,
                    sizes,
                    k_values,
                    mk,
                    mt,
                    out_sub,
                    suffix,
                    title_suffix=title_suffix,
                )

            if "paper_v1" in styles:
                v1_name = f"scalability_{mt}_{safe}_paper_v1_focal_minmax_band"
                if band_ks_effective is not None:
                    v1_name += "_K" + "_".join(str(k) for k in band_ks_effective)
                v1_name += f"{suffix}.png"
                plot_paper_v1_focal_minmax_band(
                    all_data,
                    sizes,
                    k_values,
                    mk,
                    mt,
                    os.path.join(out_sub, v1_name),
                    focal_k=focal_k,
                    title_suffix=title_suffix,
                    band_k_values=band_ks_effective,
                )
            if "paper_v2" in styles:
                plot_paper_v2_k4_8_16_family(
                    all_data,
                    sizes,
                    k_values,
                    mk,
                    mt,
                    os.path.join(out_sub, f"scalability_{mt}_{safe}_paper_v2_K4_8_16{suffix}.png"),
                    title_suffix=title_suffix,
                )
            if "paper_v3" in styles:
                plot_paper_v3_all_k_multicolor(
                    all_data,
                    sizes,
                    k_values,
                    mk,
                    mt,
                    os.path.join(out_sub, f"scalability_{mt}_{safe}_paper_v3_allK_colors{suffix}.png"),
                    title_suffix=title_suffix,
                )


def main():
    parser = argparse.ArgumentParser(description="VLDB-style scalability plots (all K, per metric).")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "Ex-2_Scalibility_Results", "scalability"),
        help="Folder containing *_pharma_scalability.json files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output folder (default: <results_dir>/vldb_plots)",
    )
    parser.add_argument(
        "--metric_type",
        type=str,
        nargs="+",
        default=["macro", "micro"],
        choices=["macro", "micro"],
    )
    parser.add_argument(
        "--styles",
        type=str,
        nargs="+",
        default=["facet_k", "radar", "paper_v1", "paper_v2", "paper_v3"],
        choices=[
            "facet_k",
            "facet_pool",
            "heatmap",
            "radar",
            "paper_v1",
            "paper_v2",
            "paper_v3",
        ],
        help="Visualization styles. paper_v1=focal K + min–max band; paper_v2=K 4/8/16; paper_v3=color=K",
    )
    parser.add_argument(
        "--focal_k",
        type=int,
        default=8,
        help="Focal K for paper_v1 solid line (default: 8)",
    )
    parser.add_argument(
        "--band_ks",
        type=str,
        default="4,8,16",
        help=(
            "paper_v1 only: comma-separated K used for min–max band (default: all K in JSON). "
            "Example: 4,8,16 — matches a tight grid so the focal line sits between those curves "
            "for metrics like P@K; omit to include K=1…32 (wide band; F1@K may touch the top at K=|GT|)."
        ),
    )
    parser.add_argument("--suffix", type=str, default="", help="Suffix for filenames, e.g. _combined")
    parser.add_argument(
        "--skip_global",
        action="store_true",
        help="Do not emit MAP / Score AP / Mean Rank figure (faster when tuning paper variants only)",
    )

    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(results_dir, "vldb_plots")

    band_ks: Optional[List[int]] = None
    if args.band_ks.strip():
        band_ks = [int(x.strip()) for x in args.band_ks.split(",") if x.strip()]

    run_all(
        results_dir,
        output_dir,
        args.metric_type,
        args.styles,
        args.suffix,
        focal_k=args.focal_k,
        skip_global=args.skip_global,
        band_ks=band_ks,
    )
    print(f"\n[VLDB-PLOT] Done. Outputs under: {output_dir}")


if __name__ == "__main__":
    main()
