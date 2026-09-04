"""
One-row composite scalability figures (macro and micro) for publication.

Panels (macro): P@k, R@k, F1@k, NDCG@k, MRR@k — each with focal-K solid line + min–max band;
                AP (Score_AP vs pool); All@K=32 (hit rate vs pool).

Panels (micro): same seven slots as macro — P@k, R@k, F1@k with band; NDCG/MRR placeholders;
                AP; All@K=32.
  Micro JSON has no NDCG/MRR or All@K in per_k; the All@32 curve reads macro["per_k"]["32"]["All@K"]
  (query-level hit rate for the same ranking — not a second micro definition).

Outputs PNG + PDF. Default suptitle is paper-oriented; put min–max band / focal K in the caption.
X-axis default: “Candidate pool size” (override with --x_label_candidates).

Usage:
  python plot_scalability_composite_row.py --results_dir Ex-2_Scalibility_Results/scalability
  python plot_scalability_composite_row.py --figure_title_macro "…" --figure_title_micro "…"
  python plot_scalability_composite_row.py --x_label_candidates "Candidate size"
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from plot_scalability_vldb import (
    MARKERS,
    PALETTE,
    _get_k_val,
    _size_label,
    _size_sort_key,
    load_scalability_jsons,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _marker_for_model(model: str, models: Sequence[str]) -> str:
    """LOKI uses a star; other models cycle through MARKERS excluding '*'."""
    if str(model).upper() == "LOKI":
        return "*"
    others = [m for m in MARKERS if m != "*"]
    non_loki = [m for m in models if str(m).upper() != "LOKI"]
    try:
        idx = non_loki.index(model)
    except ValueError:
        idx = 0
    return others[idx % len(others)] if others else "o"


def _markersize_for_model(model: str, base_size: float, loki_star_scale: float) -> float:
    """Matplotlib's '*' marker reads smaller than o/s/D at the same markersize; scale LOKI up."""
    if str(model).upper() == "LOKI":
        return float(base_size * loki_star_scale)
    return float(base_size)


def _setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _infer_micro_k_values(all_data: Dict[str, Dict[int, Any]]) -> List[int]:
    for model in all_data:
        for sz in sorted(all_data[model].keys(), key=_size_sort_key):
            micro = all_data[model].get(sz, {}).get("micro") or {}
            pk = micro.get("per_k") or {}
            if pk:
                return sorted(int(k) for k in pk.keys())
    return []


def _minmax_series(
    all_data: Dict[str, Dict[int, Any]],
    sorted_sizes: List[int],
    metric_key: str,
    metric_type: str,
    focal_k: int,
    ks_band: Sequence[int],
    model: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mids, lows, highs = [], [], []
    for sz in sorted_sizes:
        block = all_data[model].get(sz, {}).get(metric_type)
        if block is None:
            mids.append(np.nan)
            lows.append(np.nan)
            highs.append(np.nan)
            continue
        vals = [_get_k_val(block, k, metric_key) for k in ks_band]
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            mids.append(np.nan)
            lows.append(np.nan)
            highs.append(np.nan)
            continue
        fk = _get_k_val(block, focal_k, metric_key)
        mids.append(fk if np.isfinite(fk) else float(np.nanmean(finite)))
        lows.append(min(finite))
        highs.append(max(finite))
    return np.asarray(mids, float), np.asarray(lows, float), np.asarray(highs, float)


def _scalar_series(
    all_data: Dict[str, Dict[int, Any]],
    sorted_sizes: List[int],
    metric_type: str,
    json_key: str,
    model: str,
) -> np.ndarray:
    ys = []
    for sz in sorted_sizes:
        block = all_data[model].get(sz, {}).get(metric_type)
        if block is None:
            ys.append(np.nan)
            continue
        v = block.get(json_key)
        if v is None:
            ys.append(np.nan)
        elif v == float("inf"):
            ys.append(np.nan)
        else:
            ys.append(float(v))
    return np.asarray(ys, float)


def _all_at_k_series(
    all_data: Dict[str, Dict[int, Any]],
    sorted_sizes: List[int],
    k: int,
    model: str,
) -> np.ndarray:
    """All@K is stored under macro per_k only."""
    ys = []
    for sz in sorted_sizes:
        macro = all_data[model].get(sz, {}).get("macro")
        if macro is None:
            ys.append(np.nan)
            continue
        ys.append(_get_k_val(macro, k, "All@K"))
    return np.asarray(ys, float)


def _timing_series(
    all_data: Dict[str, Dict[int, Any]],
    sorted_sizes: List[int],
    model: str,
) -> np.ndarray:
    """simulated_time_sec lives in the macro block (pool-size agnostic)."""
    ys = []
    for sz in sorted_sizes:
        mac = all_data[model].get(sz, {}).get("macro", {})
        v = mac.get("simulated_time_sec")
        ys.append(float(v) if v is not None else np.nan)
    return np.asarray(ys, float)


def _legend_handles(
    models: List[str],
    line_width: float,
    marker_size: float,
    loki_star_scale: float,
):
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color=PALETTE[i % len(PALETTE)],
            marker=_marker_for_model(m, models),
            markersize=_markersize_for_model(m, marker_size, loki_star_scale),
            linewidth=line_width,
            label=m,
        )
        for i, m in enumerate(models)
    ]


def _default_composite_title(metric_type: str) -> str:
    """Paper-friendly figure title for Document-Table Discovery scalability."""
    agg = "macro-averaged" if metric_type == "macro" else "micro-averaged"
    return f"Document\u2013table discovery scalability ({agg} metrics)"


def plot_composite_row(
    all_data: Dict[str, Dict[int, Any]],
    sizes: List[int],
    metric_type: str,
    k_values_for_band: List[int],
    focal_k: int,
    output_base: str,
    *,
    figure_title: Optional[str] = None,
    x_label_candidates: str = "Candidate (Table) Pool Size",
    line_width: float = 3.0,
    marker_size: float = 6.5,
    loki_star_scale: float = 1.5,
    title_fs: int = 15,
    label_fs: int = 15,
    tick_fs: int = 13,
    legend_fs: int = 16,
    all_k_hit: int = 32,
    metrics_to_plot: Optional[List[str]] = None,
):
    if metrics_to_plot is None:
        metrics_to_plot = ["P@K", "R@K", "F1@K", "NDCG@K", "MRR@K", "AP", "All@K", "Mean Rank"]

    plt = _setup_plt()
    models = list(all_data.keys())
    sorted_sizes = sorted(sizes, key=_size_sort_key)
    
    # X-axis uses a log₂ scale so that the equal-doubling steps (50→100→…→1600)
    # appear as equal visual intervals — each step is ×2, so Δlog₂ = 1.0 everywhere.
    # "Full" (pool=0) is placed at its actual pool size read from the JSON
    # (actual_avg_pool_size), falling ~half a doubling step beyond 1600 (2240 ≈ 1600×1.4).
    # Using the real size keeps the spacing honest; a synthetic ×2 position would overstate
    # the gap between 1600 and Full.
    # Fall back to inferring from any model block if the key is absent.
    full_pool_size: float = 0.0
    for _m in all_data:
        _mac = all_data[_m].get(0, {}).get("macro", {})
        _v = _mac.get("actual_avg_pool_size")
        if _v and float(_v) > 0:
            full_pool_size = float(_v)
            break
    if full_pool_size <= 0:
        full_pool_size = max([s for s in sorted_sizes if s > 0], default=1600) * 1.4

    numeric_sizes = []
    for s in sorted_sizes:
        num = s if s > 0 else full_pool_size
        numeric_sizes.append(num)
        
    x_pos = np.log2(np.array(numeric_sizes, dtype=float))
    x_labels = [_size_label(s) for s in sorted_sizes]

    ks_band = sorted(set(k_values_for_band))
    # Build a readable K-range string for the subtitle, e.g. {4, 8, 16}
    ks_str = ", ".join(str(k) for k in ks_band)
    band_subtitle = f"Solid line: K\u202f=\u202f{focal_k}  \u00b7  Shaded band: min\u2013max over K\u2009\u2208\u2009{{{ks_str}}}"

    if metric_type == "macro":
        full_band_specs = [
            ("P@K", "P@K"),
            ("R@K", "R@K"),
            ("F1@K", "F1@K"),
            ("NDCG@K", "NDCG@K"),
            ("MRR@K", "MRR@K"),
        ]
        full_stub_labels: List[Tuple[str, str]] = []
    else:
        full_band_specs = [
            ("P@K", "P@K"),
            ("R@K", "R@K"),
            ("F1@K", "F1@K"),
        ]
        full_stub_labels = [
            ("NDCG@K", "Not defined\n(micro P/R/F1)"),
            ("MRR@K", "Not defined\n(micro P/R/F1)"),
        ]
        
    band_specs = [(k, t) for (k, t) in full_band_specs if k in metrics_to_plot]
    stub_labels = [(k, t) for (k, t) in full_stub_labels if k in metrics_to_plot]

    has_ap = "AP" in metrics_to_plot
    has_allk = "All@K" in metrics_to_plot
    has_mr = "Mean Rank" in metrics_to_plot
    has_time = "Time" in metrics_to_plot

    total_plots = len(band_specs) + len(stub_labels) + sum([has_ap, has_allk, has_mr, has_time])
    if total_plots == 0:
        print(f"[WARN] No metrics to plot for {output_base}. Skipping.")
        return

    if total_plots <= 4:
        nrows = 1
        ncols = max(1, total_plots)
    else:
        nrows = 2
        ncols = int(np.ceil(total_plots / 2.0))
    
    # Width/height adapted for 2 rows
    fig_w = max(3.15 * ncols, 12)
    fig_h = 4.0 * nrows
    fig, axes_2d = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes_2d.flatten()

    for idx in range(total_plots, len(axes)):
        axes[idx].set_visible(False)

    i0 = len(band_specs)
    for ax, (mkey, short_title) in zip(axes[:i0], band_specs):
        for mi, model in enumerate(models):
            base = PALETTE[mi % len(PALETTE)]
            y_m, y_lo, y_hi = _minmax_series(
                all_data, sorted_sizes, mkey, metric_type, focal_k, ks_band, model
            )
            valid = np.isfinite(y_m) & np.isfinite(y_lo) & np.isfinite(y_hi)
            if np.any(valid):
                ax.fill_between(
                    x_pos[valid],
                    y_lo[valid],
                    y_hi[valid],
                    color=base,
                    alpha=0.26,
                    linewidth=0,
                    zorder=1,
                )
            ax.plot(
                x_pos,
                y_m,
                color=base,
                marker=_marker_for_model(model, models),
                linewidth=line_width,
                markersize=_markersize_for_model(model, marker_size, loki_star_scale),
                zorder=3,
            )
        display_title = short_title.replace("@K", f"@{focal_k}")
        ax.set_title(display_title, fontsize=title_fs, fontweight="bold", pad=6)
        ax.set_xticks(x_pos)
        _tl = ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=tick_fs)
        if _tl:
            _tl[-1].set_ha("center")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="y", labelsize=tick_fs)

    for j, (stub_title, stub_body) in enumerate(stub_labels):
        ax = axes[i0 + j]
        ax.set_axis_off()
        display_title = stub_title.replace("@K", f"@{focal_k}")
        ax.set_title(display_title, fontsize=title_fs, fontweight="bold", pad=6)
        ax.text(
            0.5,
            0.45,
            stub_body,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=label_fs,
            color="0.45",
        )

    current_idx = i0 + len(stub_labels)

    if has_ap:
        ax_ap = axes[current_idx]
        for mi, model in enumerate(models):
            base = PALETTE[mi % len(PALETTE)]
            ys = _scalar_series(all_data, sorted_sizes, metric_type, "Score_AP", model)
            if not np.any(np.isfinite(ys)):
                ys = _scalar_series(all_data, sorted_sizes, metric_type, "MAP", model)
            ax_ap.plot(
                x_pos,
                ys,
                color=base,
                marker=_marker_for_model(model, models),
                linewidth=line_width,
                markersize=_markersize_for_model(model, marker_size, loki_star_scale),
            )
        ax_ap.set_title("AP", fontsize=title_fs, fontweight="bold", pad=6)
        ax_ap.set_xticks(x_pos)
        _tl = ax_ap.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=tick_fs)
        if _tl:
            _tl[-1].set_ha("center")
        ax_ap.set_ylim(-0.05, 1.05)
        ax_ap.grid(True, alpha=0.35)
        ax_ap.tick_params(axis="y", labelsize=tick_fs)
        current_idx += 1

    if has_allk:
        ax_all = axes[current_idx]
        for mi, model in enumerate(models):
            base = PALETTE[mi % len(PALETTE)]
            ys = _all_at_k_series(all_data, sorted_sizes, all_k_hit, model)
            ax_all.plot(
                x_pos,
                ys,
                color=base,
                marker=_marker_for_model(model, models),
                linewidth=line_width,
                markersize=_markersize_for_model(model, marker_size, loki_star_scale),
            )
        ax_all.set_title(f"Hit Rate (All@{all_k_hit})", fontsize=title_fs, fontweight="bold", pad=6)
        ax_all.set_xticks(x_pos)
        _tl = ax_all.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=tick_fs)
        if _tl:
            _tl[-1].set_ha("center")
        ax_all.set_ylim(-0.05, 1.05)
        ax_all.grid(True, alpha=0.35)
        ax_all.tick_params(axis="y", labelsize=tick_fs)
        current_idx += 1

    if has_mr:
        ax_mr = axes[current_idx]
        for mi, model in enumerate(models):
            base = PALETTE[mi % len(PALETTE)]
            ys = _scalar_series(all_data, sorted_sizes, metric_type, "Mean_Rank", model)
            # If tracking under micro isn't present, try pulling from macro
            if not np.any(np.isfinite(ys)) and metric_type == "micro":
                ys = _scalar_series(all_data, sorted_sizes, "macro", "Mean_Rank", model)
                
            ax_mr.plot(
                x_pos,
                ys,
                color=base,
                marker=_marker_for_model(model, models),
                linewidth=line_width,
                markersize=_markersize_for_model(model, marker_size, loki_star_scale),
            )
        ax_mr.set_title("Mean Rank (lower is better)", fontsize=title_fs - 1, fontweight="bold", pad=6)
        ax_mr.set_xticks(x_pos)
        _tl = ax_mr.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=tick_fs)
        if _tl:
            _tl[-1].set_ha("center")
        
        ax_mr.set_ylim(bottom=0)
        
        ax_mr.grid(True, alpha=0.35)
        ax_mr.tick_params(axis="y", labelsize=tick_fs)
        current_idx += 1

    if has_time:
        import matplotlib.ticker as mticker
        ax_t = axes[current_idx]
        for mi, model in enumerate(models):
            base = PALETTE[mi % len(PALETTE)]
            ys = _timing_series(all_data, sorted_sizes, model)
            ax_t.plot(
                x_pos,
                ys,
                color=base,
                marker=_marker_for_model(model, models),
                linewidth=line_width,
                markersize=_markersize_for_model(model, marker_size, loki_star_scale),
            )
        ax_t.set_yscale("log")
        ax_t.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$s" if v > 0 else "")
        )
        ax_t.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax_t.set_ylabel("Time (seconds)", fontsize=label_fs)
        ax_t.set_title("Inference Time (log)", fontsize=title_fs, fontweight="bold", pad=6)
        ax_t.set_xticks(x_pos)
        _tl = ax_t.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=tick_fs)
        if _tl:
            _tl[-1].set_ha("center")
        ax_t.grid(True, which="both", alpha=0.3)
        ax_t.tick_params(axis="y", labelsize=tick_fs)
        current_idx += 1

    # Y-axis labels for the leftmost panels (first panel of each row)
    if total_plots > 0:
        if "Mean Rank" not in axes[0].get_title():
            axes[0].set_ylabel("Score", fontsize=label_fs)
        if nrows > 1 and ncols < total_plots:
            if "Mean Rank" not in axes[ncols].get_title():
                axes[ncols].set_ylabel("Score", fontsize=label_fs)
                if not axes[ncols].axison:
                    axes[ncols].set_axis_on()
                    axes[ncols].set_xticks([])
                    axes[ncols].set_yticks([])
                    for spine in axes[ncols].spines.values():
                        spine.set_visible(False)

    leg_y = 1.01 if nrows > 1 else 1.02
    title_y = 1.06 if nrows > 1 else 1.12
    sub_y = 1.03 if nrows > 1 else 1.065

    leg_handles = _legend_handles(models, line_width, marker_size, loki_star_scale)
    fig.legend(
        leg_handles,
        [h.get_label() for h in leg_handles],
        loc="upper center",
        ncol=min(4, len(models)),
        fontsize=legend_fs,
        frameon=True,
        bbox_to_anchor=(0.5, leg_y),
        columnspacing=1.2,
        handletextpad=0.6,
    )

    ft = _default_composite_title(metric_type)
    if figure_title is not None and str(figure_title).strip():
        ft = str(figure_title).strip()
    fig.suptitle(ft, fontsize=title_fs + 1, fontweight="bold", y=title_y)

    plt.tight_layout(rect=[0, 0, 1, 0.95] if nrows > 1 else [0, 0, 1, 0.94])

    # Single shared x-axis label for the entire figure
    fig.text(
        0.5, -0.02, x_label_candidates,
        ha="center", va="top",
        fontsize=title_fs, fontweight="normal",
        transform=fig.transFigure,
    )

    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{output_base}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", format=ext)
        print(f"[COMPOSITE] {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="One-row composite scalability figures (PNG+PDF).")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "Ex-2_Scalibility_Results", "scalability", "pharma_flipped_structured"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Default: <results_dir>/composite_row",
    )
    parser.add_argument("--focal_k", type=int, default=8)
    parser.add_argument(
        "--band_ks",
        type=str,
        default="8",
        help="Comma-separated K for min–max band (default: all K in JSON)",
    )
    parser.add_argument("--all_k_hit", type=int, default=32, help="K for All@K hit-rate panel")
    parser.add_argument(
        "--figure_title",
        type=str,
        default="",
        help="Suptitle for both figures if macro/micro-specific titles are empty (use neutral wording)",
    )
    parser.add_argument(
        "--figure_title_macro",
        type=str,
        default="",
        help="Suptitle for macro row only (overrides --figure_title for that file)",
    )
    parser.add_argument(
        "--figure_title_micro",
        type=str,
        default="",
        help="Suptitle for micro row only (overrides --figure_title for that file)",
    )
    parser.add_argument(
        "--x_label_candidates",
        type=str,
        default="Candidate (Table) Pool Size",
        help="Shared x-axis label (single label for the whole figure)",
    )
    parser.add_argument("--suffix", type=str, default="", help="Appended to output basename")
    parser.add_argument(
        "--marker_size",
        type=float,
        default=5.0,
        help="Base markersize for non-star markers (LOKI star uses marker_size * loki_star_scale)",
    )
    parser.add_argument(
        "--loki_star_scale",
        type=float,
        default=2.0,
        help="Multiply marker_size for LOKI '*' so it matches visual weight of other markers",
    )
    
    parser.add_argument(
        "--metrics_to_plot",
        type=str,
        nargs="+",
        default=["F1@K", "NDCG@K", "MRR@K", "AP", "Mean Rank", "Time"],
        choices=["P@K", "R@K", "F1@K", "NDCG@K", "MRR@K", "AP", "All@K", "Mean Rank", "Time"],
        help="List of metrics to include in the composite row. Add 'Time' for the log-scale inference time panel.",
    )
    
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(results_dir, "composite_row")
    os.makedirs(output_dir, exist_ok=True)

    all_data, sizes, k_macro = load_scalability_jsons(results_dir)
    band_ks: Optional[List[int]] = None
    if args.band_ks.strip():
        band_ks = [int(x.strip()) for x in args.band_ks.split(",") if x.strip()]
        band_ks = sorted({k for k in band_ks if k in k_macro})
        if not band_ks:
            band_ks = None

    k_band_eff = band_ks if band_ks is not None else k_macro

    def _title_for(metric_type: str) -> Optional[str]:
        if metric_type == "macro" and args.figure_title_macro.strip():
            return args.figure_title_macro.strip()
        if metric_type == "micro" and args.figure_title_micro.strip():
            return args.figure_title_micro.strip()
        if args.figure_title.strip():
            return args.figure_title.strip()
        return None

    suf = args.suffix or ""
    base_macro = os.path.join(output_dir, f"scalability_composite_macro_row{suf}")
    plot_composite_row(
        all_data,
        sizes,
        "macro",
        k_band_eff,
        args.focal_k,
        base_macro,
        all_k_hit=args.all_k_hit,
        marker_size=args.marker_size,
        loki_star_scale=args.loki_star_scale,
        figure_title=_title_for("macro"),
        x_label_candidates=args.x_label_candidates,
        metrics_to_plot=args.metrics_to_plot,
    )

    k_micro = _infer_micro_k_values(all_data)
    if not k_micro:
        k_micro = k_macro
    k_band_micro = [k for k in k_band_eff if k in k_micro] or k_micro

    base_micro = os.path.join(output_dir, f"scalability_composite_micro_row{suf}")
    plot_composite_row(
        all_data,
        sizes,
        "micro",
        k_band_micro,
        args.focal_k,
        base_micro,
        all_k_hit=args.all_k_hit,
        marker_size=args.marker_size,
        loki_star_scale=args.loki_star_scale,
        figure_title=_title_for("micro"),
        x_label_candidates=args.x_label_candidates,
        metrics_to_plot=args.metrics_to_plot,
    )


if __name__ == "__main__":
    main()
