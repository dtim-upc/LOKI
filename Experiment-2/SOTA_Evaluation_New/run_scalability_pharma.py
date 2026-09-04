"""
run_scalability_pharma.py — Search-space scalability study on Pharma.

Measures how well models find the ~8 ground-truth tables as the candidate
pool (search space) grows from 50 to Full (~2240 tables).

Approach (efficient):
  1. Run each model ONCE on the FULL dataset to get raw per-(query, table) scores.
  2. For each candidate pool size, restrict the search space per query:
       - Always include ALL ground-truth tables for that query.
       - Sample random negatives to fill to the target pool size.
  3. Re-rank within the restricted pool and recompute all metrics.

This avoids re-running expensive model inference for each pool size.

Quick examples:
  python run_scalability_pharma.py                                   # all 4 models
  python run_scalability_pharma.py --sizes 50 100 500                # custom sizes
  python run_scalability_pharma.py --skip_cmdl --skip_tabstar        # LOKI + TaBERT only
  python run_scalability_pharma.py --focal_k 8                       # main plot at K=8
"""

import os
import sys
import json
import argparse
import random
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    K_VALUES, OUTPUT_DIR, SEED, SCALABILITY_SIZES,
    LOKI_MODELS, LOKI_ACTIVE_MODEL, LOKI_USE_SCHEMA_AWARE_SCORER, LOKI_ARGS_PATH,
    LOKI_SCHEMA_AWARE_REPRESENTATION,
    LOKI_CELL_LEVEL_MATCHING_REPRESENTATION,
)
from metrics import evaluate_retrieval, evaluate_retrieval_micro
from plot_scalability import plot_scalability_main, plot_scalability_per_k
from evaluate_loki import load_loki_json


# Default test file for pharma flipped
DEFAULT_PHARMA_TEST_FILE = os.path.join(
    SCRIPT_DIR, "..", "Datasets", "pharma_flipped_structured", "test_row_level.json"
)


# ===========================================================================
# Extract hard negatives from the dataset
# ===========================================================================

def extract_hard_negatives(test_file, native_direction="DOC_TO_TABLE"):
    """
    Extract per-query hard negative IDs from the test JSON.

    For DOC_TO_TABLE (native): anchors are docs (queries), negatives are tables.
    For TABLE_TO_DOC (native): anchors are tables (queries), negatives are docs.

    Returns:
        hard_neg_map: {query_id: set(candidate_id, ...)}
    """
    examples = load_loki_json(test_file)

    hard_neg_map = {}
    for ex in examples:
        anchor_id = ex.get("anchor_id")
        if anchor_id is None:
            continue

        neg_ids = set()
        for neg in ex.get("negatives", []):
            neg_id = neg.get("id")
            if neg_id is not None:
                neg_ids.add(str(neg_id))

        if neg_ids:
            hard_neg_map[str(anchor_id)] = neg_ids

    total_hn = sum(len(v) for v in hard_neg_map.values())
    print(f"[SCALABILITY] Extracted hard negatives: {len(hard_neg_map)} queries, "
          f"{total_hn} total ({total_hn / max(len(hard_neg_map), 1):.1f} avg per query)")
    return hard_neg_map


# ===========================================================================
# Candidate pool restriction (core of the scalability study)
# ===========================================================================

def restrict_candidate_pool(
    scores_map: dict,
    gt_map: dict,
    pool_size: int,
    seed: int = 42,
    hard_neg_map: dict = None,
) -> tuple:
    """
    Restrict the candidate pool per query for a scalability experiment.

    For each query:
      - Always include ALL ground-truth tables.
      - Always include ALL hard negatives (from the dataset).
      - Sample random negatives to fill up to pool_size.
      - If pool_size == 0 or >= total tables: use the full pool.

    Args:
        scores_map:    {query_id: {table_id: score}} — full model scores
        gt_map:        {query_id: [gt_table_id, ...]}
        pool_size:     target number of candidate tables per query (0 = full)
        seed:          random seed for reproducible negative sampling
        hard_neg_map:  {query_id: set(table_id, ...)} — hard negatives to keep

    Returns:
        (restricted_scores_map, restricted_predictions_map, gt_map)
    """
    rng = random.Random(seed)
    restricted_scores = {}
    restricted_preds = {}
    hard_neg_map = hard_neg_map or {}

    for qid in scores_map:
        all_tables = scores_map[qid]
        gt_set = set(gt_map.get(qid, []))
        hn_set = hard_neg_map.get(qid, set())

        # Mandatory tables: GT + hard negatives (both always included)
        mandatory = gt_set | hn_set
        # Only keep mandatory tables that actually have scores
        mandatory_with_scores = {t for t in mandatory if t in all_tables}

        # If pool_size is 0 (full) or >= all available tables, no restriction
        if pool_size == 0 or pool_size >= len(all_tables):
            restricted_scores[qid] = dict(all_tables)
            ranked = sorted(all_tables.items(), key=lambda x: x[1], reverse=True)
            restricted_preds[qid] = [tid for tid, _ in ranked]
            continue

        # How many random negatives to add on top of mandatory tables
        n_mandatory = len(mandatory_with_scores)
        n_random = max(0, pool_size - n_mandatory)

        # Non-mandatory tables available for random sampling
        random_pool = [tid for tid in all_tables if tid not in mandatory]

        if n_random >= len(random_pool):
            sampled_random = random_pool
        else:
            sampled_random = rng.sample(random_pool, n_random)

        # Build restricted pool
        pool_tables = list(mandatory_with_scores) + sampled_random
        pool_scores = {tid: all_tables[tid] for tid in pool_tables}

        restricted_scores[qid] = pool_scores
        ranked = sorted(pool_scores.items(), key=lambda x: x[1], reverse=True)
        restricted_preds[qid] = [tid for tid, _ in ranked]

    return restricted_scores, restricted_preds, gt_map


# ===========================================================================
# Per-model full-dataset scoring (run ONCE)
# ===========================================================================

def score_cmdl_full(args):
    """Run CMDL on full dataset, return (scores_map, gt_map)."""
    from evaluate_cmdl import evaluate_cmdl
    _, scores_map, gt_map = evaluate_cmdl(
        test_file=args.test_file,
        max_test_examples=0,  # Full dataset
        seed=args.seed,
        task=args.task_direction,
        dataset_format=args.dataset_format,
        native_direction=args.native_direction,
        return_scores=True,
    )
    return scores_map, gt_map


def score_loki_full(args):
    """Run LOKI on full dataset, return (scores_map, gt_map)."""
    from evaluate_loki import evaluate_loki
    _, scores_map, gt_map = evaluate_loki(
        test_file=args.test_file,
        max_test_examples=0,
        seed=args.seed,
        loki_model_key=args.loki_model,
        aggregate_to_global_tables=False,
        task=args.task_direction,
        dataset_format=args.dataset_format,
        native_direction=args.native_direction,
        encode_batch_size=args.encode_batch_size,
        eval_row_chunk_size=args.eval_row_chunk_size,
        cache_table_embeddings=args.cache_table_embeddings,
        cache_doc_embeddings=args.cache_doc_embeddings,
        return_scores=True,
        use_schema_aware_loki=args.use_schema_aware_loki,
    )
    return scores_map, gt_map


def _resolve_loki_schema_mode(requested_use_schema_aware_loki) -> bool:
    if requested_use_schema_aware_loki is not None:
        return bool(requested_use_schema_aware_loki)

    try:
        with open(LOKI_ARGS_PATH, "r", encoding="utf-8") as f:
            loki_args = json.load(f)
        return bool(
            loki_args.get("use_header_conditioning", False)
            or loki_args.get("use_cell_level_matching", False)
        )
    except Exception:
        return False


def _resolve_loki_schema_representation(requested_use_schema_aware_loki) -> str:
    resolved_use_schema_aware_loki = _resolve_loki_schema_mode(requested_use_schema_aware_loki)
    try:
        with open(LOKI_ARGS_PATH, "r", encoding="utf-8") as f:
            loki_args = json.load(f)
        checkpoint_uses_header_conditioning = bool(loki_args.get("use_header_conditioning", False))
        checkpoint_uses_cell_level_matching = bool(loki_args.get("use_cell_level_matching", False))
    except Exception:
        checkpoint_uses_header_conditioning = False
        checkpoint_uses_cell_level_matching = False

    if resolved_use_schema_aware_loki:
        representation_parts = []
        if checkpoint_uses_header_conditioning:
            representation_parts.append(LOKI_SCHEMA_AWARE_REPRESENTATION)
        if checkpoint_uses_cell_level_matching:
            representation_parts.append(LOKI_CELL_LEVEL_MATCHING_REPRESENTATION)
        if representation_parts:
            return "+".join(representation_parts)
    return "legacy"


def _loki_cache_compatible(cache_data, use_schema_aware_loki: bool, schema_representation: str) -> bool:
    metadata = cache_data.get("metadata", {}) if isinstance(cache_data, dict) else {}
    cached_flag = bool(metadata.get("use_schema_aware_loki", False))
    cached_representation = metadata.get("loki_schema_representation", "legacy")
    return cached_flag == bool(use_schema_aware_loki) and cached_representation == schema_representation


def score_tabstar_full(args):
    """Run TabSTAR on full dataset, return (scores_map, gt_map)."""
    from evaluate_tabstar import evaluate_tabstar
    _, scores_map, gt_map = evaluate_tabstar(
        test_file=args.test_file,
        max_test_examples=0,
        seed=args.seed,
        task=args.task_direction,
        dataset_format=args.dataset_format,
        native_direction=args.native_direction,
        return_scores=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return scores_map, gt_map


def score_tabert_full(args):
    """Run TaBERT on full dataset, return (scores_map, gt_map)."""
    from evaluate_tabert import evaluate_tabert
    _, scores_map, gt_map = evaluate_tabert(
        test_file=args.test_file,
        max_test_examples=0,
        seed=args.seed,
        task=args.task_direction,
        dataset_format=args.dataset_format,
        native_direction=args.native_direction,
        return_scores=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        bf16=args.bf16,
        torch_compile=args.torch_compile,
    )
    return scores_map, gt_map


MODEL_SCORERS = {
    "CMDL": score_cmdl_full,
    "LOKI": score_loki_full,
    "TabSTAR": score_tabstar_full,
    "TaBERT": score_tabert_full,
}


# ===========================================================================
# Helpers
# ===========================================================================

def size_label(size):
    return "Full" if size == 0 else str(size)


# ===========================================================================
# Main
# ===========================================================================

def main():
    loki_model_names = ", ".join(LOKI_MODELS.keys())

    parser = argparse.ArgumentParser(
        description="Search-Space Scalability Study: vary candidate pool sizes"
    )

    # Core args
    parser.add_argument("--test_file", type=str, default=DEFAULT_PHARMA_TEST_FILE,
                        help="Path to test JSON (default: %s)" % DEFAULT_PHARMA_TEST_FILE)
    parser.add_argument("--sizes", type=int, nargs="+", default=SCALABILITY_SIZES,
                        help="Candidate pool sizes per query (0 = full). "
                             "Default: %s" % SCALABILITY_SIZES)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for negative sampling (default: %d)" % SEED)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(OUTPUT_DIR, "scalability"),
                        help="Output directory for scalability results")
    parser.add_argument("--focal_k", type=int, default=8,
                        help="K value for the main scalability figure (default: 8)")

    # Model selection
    parser.add_argument("--loki_model", type=str, default=LOKI_ACTIVE_MODEL,
                        choices=list(LOKI_MODELS.keys()),
                        help="Which LOKI checkpoint to use. Options: [%s]" % loki_model_names)
    parser.add_argument("--use_schema_aware_loki", action=argparse.BooleanOptionalAction,
                        default=LOKI_USE_SCHEMA_AWARE_SCORER,
                        help="Use the Rewind-compatible structured LOKI scorer path. Default auto-detects header conditioning and cell-level matching from the checkpoint args.json.")
    parser.add_argument("--skip_cmdl", action="store_true", help="Skip CMDL")
    parser.add_argument("--skip_loki", action="store_true", help="Skip LOKI")
    parser.add_argument("--skip_tabstar", action="store_true", help="Skip TabSTAR")
    parser.add_argument("--skip_tabert", action="store_true", help="Skip TaBERT")

    # LOKI-specific
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--eval_row_chunk_size", type=int, default=0)
    parser.add_argument("--cache_table_embeddings", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--cache_doc_embeddings", action=argparse.BooleanOptionalAction,
                        default=False)

    # TaBERT-specific
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=True)

    # Task direction args
    parser.add_argument("--task_direction", type=str, default="DOC_TO_TABLE",
                        choices=["DOC_TO_TABLE", "TABLE_TO_DOC"],
                        help="Task direction for evaluation.")
    parser.add_argument("--native_direction", type=str, default="DOC_TO_TABLE",
                        choices=["DOC_TO_TABLE", "TABLE_TO_DOC"],
                        help="Native direction of the dataset.")
    parser.add_argument("--dataset_format", type=str, default="other",
                        choices=["mimic", "other"],
                        help="Dataset format identifier.")

    # Result reuse
    parser.add_argument("--force_rerun", action="store_true",
                        help="Force re-scoring even if cached scores exist")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    resolved_loki_schema_mode = _resolve_loki_schema_mode(args.use_schema_aware_loki)
    resolved_loki_schema_representation = _resolve_loki_schema_representation(args.use_schema_aware_loki)

    suffix = ""

    # Determine which models to run
    active_models = []
    if not args.skip_cmdl:
        active_models.append("CMDL")
    if not args.skip_loki:
        active_models.append("LOKI")
    if not args.skip_tabstar:
        active_models.append("TabSTAR")
    if not args.skip_tabert:
        active_models.append("TaBERT")

    if not active_models:
        print("[ERROR] All models are skipped. Nothing to do.")
        return

    sizes = sorted(args.sizes, key=lambda s: 999_999 if s == 0 else s)
    k_values = K_VALUES

    print("\n" + "=" * 70)
    print("  Search-Space Scalability Study — Pharma Protocol")
    print("=" * 70)
    print(f"  Models:        {', '.join(active_models)}")
    print(f"  Pool sizes:    {[size_label(s) for s in sizes]}")
    print(f"  K values:      {k_values}")
    print(f"  Output:        {args.output_dir}")
    print("=" * 70 + "\n")

    # -----------------------------------------------------------------------
    # Phase 1: Score each model ONCE on the full dataset
    # -----------------------------------------------------------------------
    model_scores = {}  # {model_name: (scores_map, gt_map)}

    for model_name in active_models:
        cache_path = os.path.join(
            args.output_dir, f"{model_name}_pharma{suffix}_full_scores.json"
        )

        # Try loading cached scores
        if os.path.exists(cache_path) and not args.force_rerun:
            print(f"[CACHE] Loading {model_name} full scores from {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if model_name == "LOKI" and not _loki_cache_compatible(cached, resolved_loki_schema_mode, resolved_loki_schema_representation):
                print("  [CACHE] Ignoring cached LOKI scores because the schema-aware scorer configuration differs.")
            else:
                elapsed_time = cached.get("elapsed_time_sec", 0.0)
                model_scores[model_name] = (cached["scores_map"], cached["gt_map"], elapsed_time)
                
                # Report stats
                sm = cached["scores_map"]
                gm = cached["gt_map"]
                n_queries = len(sm)
                n_tables = len(set(t for s in sm.values() for t in s)) if sm else 0
                gt_sizes = [len(v) for v in gm.values()]
                avg_gt = np.mean(gt_sizes) if gt_sizes else 0
                if elapsed_time == 0.0:
                    print(f"  [WARN] Cache missing 'elapsed_time_sec'. Scalability simulated times will be 0.0.")
                print(f"  {model_name}: {n_queries} queries, {n_tables} tables, avg GT={avg_gt:.1f}, full_time={elapsed_time:.1f}s")
                continue

        print(f"\n{'=' * 65}")
        print(f"  Scoring {model_name} on FULL dataset (one-time)")
        print(f"{'=' * 65}")

        scorer = MODEL_SCORERS[model_name]
        t0 = time.time()
        try:
            scores_map, gt_map = scorer(args)
        except Exception as e:
            print(f"[ERROR] {model_name} scoring failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        elapsed = time.time() - t0

        model_scores[model_name] = (scores_map, gt_map, elapsed)

        # Cache scores for reuse
        cache_data = {
            "scores_map": scores_map,
            "gt_map": gt_map,
            "elapsed_time_sec": elapsed
        }
        if model_name == "LOKI":
            cache_data["metadata"] = {
                "use_schema_aware_loki": resolved_loki_schema_mode,
                "loki_schema_representation": resolved_loki_schema_representation,
            }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

        n_queries = len(scores_map)
        n_tables = len(set(t for s in scores_map.values() for t in s))
        gt_sizes = [len(v) for v in gt_map.values()]
        avg_gt = np.mean(gt_sizes) if gt_sizes else 0
        print(f"  [{model_name}] Scored in {elapsed:.1f}s — "
              f"{n_queries} queries × {n_tables} tables, avg GT={avg_gt:.1f}")
        print(f"  Cached to {cache_path}")

    # -----------------------------------------------------------------------
    # Phase 2: For each pool size, restrict candidates and compute metrics
    # -----------------------------------------------------------------------
    # Extract hard negatives from the dataset (once)
    hard_neg_map = extract_hard_negatives(args.test_file, native_direction=args.native_direction)

    # Structure: {model_name: {size: {macro: {...}, micro: {...}}}}
    all_scalability = {m: {} for m in active_models if m in model_scores}

    print("\n" + "=" * 70)
    print("  Phase 2: Computing metrics for each pool size")
    print("=" * 70)

    for model_name in active_models:
        if model_name not in model_scores:
            continue

        scores_map, gt_map, elapsed_full = model_scores[model_name]
        
        # Calculate full corpus avg pool size from scores_map for reliable time proportion scaling
        full_actual_sizes = [len(scores_map[q]) for q in scores_map] if scores_map else [1]
        avg_full_pool = float(np.mean(full_actual_sizes))
        
        cumulative_time = 0.0

        for pool_size in sizes:
            label = size_label(pool_size)

            # Restrict candidate pool (GT + hard negatives always kept)
            r_scores, r_preds, r_gt = restrict_candidate_pool(
                scores_map, gt_map, pool_size, seed=args.seed,
                hard_neg_map=hard_neg_map,
            )

            # Compute macro metrics
            macro = evaluate_retrieval(r_gt, r_preds, k_values, scores_map=r_scores)
            macro["num_examples"] = len(r_preds)
            macro["pool_size"] = pool_size
            macro["pool_label"] = label

            # Report actual pool sizes
            actual_sizes = [len(r_scores[q]) for q in r_scores]
            avg_pool = float(np.mean(actual_sizes))
            macro["actual_avg_pool_size"] = avg_pool

            # Simulate runtime: directly scaling the total true inference time linearly by target pool size
            simulated_time_sec = elapsed_full * (avg_pool / max(1.0, avg_full_pool))
            cumulative_time += simulated_time_sec
            macro["simulated_time_sec"] = simulated_time_sec
            macro["cumulative_time_sec"] = cumulative_time

            # Compute micro metrics
            micro = evaluate_retrieval_micro(r_gt, r_preds, k_values, scores_map=r_scores)
            micro["num_examples"] = len(r_preds)
            micro["pool_size"] = pool_size

            all_scalability[model_name][pool_size] = {
                "macro": macro,
                "micro": micro,
            }

            # Quick summary
            map_val = macro.get("MAP", 0)
            mean_rank = macro.get("Mean_Rank", 0)
            print(f"  {model_name} @ pool={label:>6s} "
                  f"(avg_actual={avg_pool:>6.0f}): "
                  f"MAP={map_val:.4f}  MeanRank={mean_rank:.2f}  Cum.Time={cumulative_time:.1f}s")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    for model_name in all_scalability:
        result_path = os.path.join(
            args.output_dir,
            f"{model_name}_pharma{suffix}_scalability.json"
        )
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in all_scalability[model_name].items()},
                f, indent=2
            )
        print(f"\n[SAVE] {model_name} scalability: {result_path}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  Search-Space Scalability Summary — MAP (macro)")
    print("=" * 100)
    header = f"  {'Pool Size':<12}"
    for m in all_scalability:
        header += f"  {m:<15}"
    print(header)
    print(f"  {'-' * (12 + 17 * len(all_scalability))}")

    for pool_size in sizes:
        row = f"  {size_label(pool_size):<12}"
        for m in all_scalability:
            res = all_scalability[m].get(pool_size, {})
            macro = res.get("macro", {})
            map_val = macro.get("MAP", 0)
            row += f"  {map_val:<15.4f}"
        print(row)

    print("=" * 100 + "\n")
    
    # -----------------------------------------------------------------------
    # Cumulative Timing Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  Search-Space Scalability Summary — Cumulative Inference Time (Seconds)")
    print("=" * 100)
    header = f"  {'Pool Size':<12}"
    for m in all_scalability:
        header += f"  {m:<15}"
    print(header)
    print(f"  {'-' * (12 + 17 * len(all_scalability))}")

    for pool_size in sizes:
        row = f"  {size_label(pool_size):<12}"
        for m in all_scalability:
            res = all_scalability[m].get(pool_size, {})
            macro = res.get("macro", {})
            cum_time = macro.get("cumulative_time_sec", 0.0)
            row += f"  {cum_time:<15.1f}"
        print(row)

    print("=" * 100 + "\n")

    # -----------------------------------------------------------------------
    # Generate plots
    # -----------------------------------------------------------------------
    try:
        plot_scalability_main(
            all_scalability, sizes,
            focal_k=args.focal_k,
            output_dir=args.output_dir,
            suffix=suffix,
            metric_type="macro",
        )
        plot_scalability_main(
            all_scalability, sizes,
            focal_k=args.focal_k,
            output_dir=args.output_dir,
            suffix=suffix,
            metric_type="micro",
        )
        plot_scalability_per_k(
            all_scalability, sizes, k_values,
            output_dir=args.output_dir,
            suffix=suffix,
            metric_type="macro",
        )
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")
        import traceback
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Excel export
    # -----------------------------------------------------------------------
    try:
        _export_scalability_excel(all_scalability, sizes, k_values, args.output_dir, suffix)
    except Exception as e:
        print(f"[WARN] Excel export failed: {e}")

    print("\nDone!")


def _export_scalability_excel(all_scalability, sizes, k_values, output_dir, suffix):
    """Export scalability results to Excel."""
    import pandas as pd

    rows = []
    for model_name, size_data in all_scalability.items():
        for size in sorted(sizes, key=lambda s: 999_999 if s == 0 else s):
            res = size_data.get(size, {})
            macro = res.get("macro", {})
            if not macro:
                continue
            for k in k_values:
                per_k = macro.get("per_k", {})
                kd = per_k.get(k, per_k.get(str(k), {}))
                rows.append({
                    "Model": model_name,
                    "Pool Size": size_label(size),
                    "K": k,
                    "P@K": kd.get("P@K", 0),
                    "R@K": kd.get("R@K", 0),
                    "F1@K": kd.get("F1@K", 0),
                    "NDCG@K": kd.get("NDCG@K", 0),
                    "MRR@K": kd.get("MRR@K", 0),
                    "All@K": kd.get("All@K", 0),
                    "MAP": macro.get("MAP", 0),
                    "Score_AP": macro.get("Score_AP", 0),
                    "Mean_Rank": macro.get("Mean_Rank", 0),
                    "Simulated_Time_sec": macro.get("simulated_time_sec", 0),
                    "Cumulative_Time_sec": macro.get("cumulative_time_sec", 0),
                })

    df = pd.DataFrame(rows)
    excel_path = os.path.join(output_dir, f"Scalability_Results{suffix}.xlsx")
    os.makedirs(output_dir, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="scalability", index=False)
    print(f"[EXCEL] Saved: {excel_path}")


if __name__ == "__main__":
    main()
