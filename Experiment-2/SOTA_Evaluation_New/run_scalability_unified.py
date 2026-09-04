"""
run_scalability_unified.py — Search-space scalability study Unified.

Measures how well models find the ground-truth tables as the candidate
pool (search space) grows using the unified dataset infrastructure.
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

from dataset_registry import get_dataset_info, get_split_path
from config import (
    K_VALUES, OUTPUT_DIR, SEED, SCALABILITY_SIZES,
    LOKI_MODELS, LOKI_ACTIVE_MODEL, LOKI_USE_SCHEMA_AWARE_SCORER, LOKI_ARGS_PATH,
    LOKI_SCHEMA_AWARE_REPRESENTATION, LOKI_CELL_LEVEL_MATCHING_REPRESENTATION,
    MAX_TEST_EXAMPLES, MAX_QUERIES
)
from metrics import evaluate_retrieval, evaluate_retrieval_micro
from plot_scalability import plot_scalability_main, plot_scalability_per_k
from evaluate_loki import load_loki_json


# ===========================================================================
# Extract hard negatives from the dataset
# ===========================================================================

def extract_hard_negatives(test_file):
    """
    Extract per-query hard negative IDs from the test JSON.

    Anchors are queries; negatives are candidate items to always keep
    in restricted pools so the scalability study includes the dataset's
    hard confusors at every pool size.

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


def _log_pool_composition(restricted_scores, gt_map, pool_size):
    """Log summary of pool composition for verification."""
    gt_counts, rand_counts = [], []
    for qid in restricted_scores:
        pool_ids = set(restricted_scores[qid].keys())
        gt_set = set(gt_map.get(qid, []))
        gt_in = len(pool_ids & gt_set)
        rand_in = len(pool_ids) - gt_in
        gt_counts.append(gt_in)
        rand_counts.append(rand_in)
    label = "Full" if pool_size == 0 else str(pool_size)
    print(f"    Pool composition @ {label}: "
          f"avg GT={np.mean(gt_counts):.1f}, "
          f"avg random={np.mean(rand_counts):.1f}, "
          f"avg total={np.mean([len(restricted_scores[q]) for q in restricted_scores]):.0f}")


# ===========================================================================
# Per-model full-dataset scoring (run ONCE) passing task parameters
# ===========================================================================

def score_cmdl_full(args, dataset_format, test_file, native_direction="TABLE_TO_DOC"):
    from evaluate_cmdl import evaluate_cmdl
    _, scores_map, gt_map = evaluate_cmdl(
        test_file=test_file,
        max_test_examples=args.max_test_examples,
        max_queries=args.max_queries,
        seed=args.seed,
        task=args.task,
        dataset_format=dataset_format,
        native_direction=native_direction,
        return_scores=True,
    )
    return scores_map, gt_map

def score_loki_full(args, dataset_format, test_file, native_direction="TABLE_TO_DOC"):
    from evaluate_loki import evaluate_loki
    _, scores_map, gt_map = evaluate_loki(
        test_file=test_file,
        max_test_examples=args.max_test_examples,
        max_queries=args.max_queries,
        seed=args.seed,
        loki_model_key=args.loki_model,
        task=args.task,
        dataset_format=dataset_format,
        native_direction=native_direction,
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

def score_tabstar_full(args, dataset_format, test_file, native_direction="TABLE_TO_DOC"):
    from evaluate_tabstar import evaluate_tabstar
    _, scores_map, gt_map = evaluate_tabstar(
        test_file=test_file,
        max_test_examples=args.max_test_examples,
        max_queries=args.max_queries,
        seed=args.seed,
        task=args.task,
        dataset_format=dataset_format,
        native_direction=native_direction,
        return_scores=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return scores_map, gt_map

def score_tabert_full(args, dataset_format, test_file, native_direction="TABLE_TO_DOC"):
    from evaluate_tabert import evaluate_tabert
    _, scores_map, gt_map = evaluate_tabert(
        test_file=test_file,
        max_test_examples=args.max_test_examples,
        max_queries=args.max_queries,
        seed=args.seed,
        task=args.task,
        dataset_format=dataset_format,
        native_direction=native_direction,
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

def size_label(size):
    return "Full" if size == 0 else str(size)

# ===========================================================================
# Main
# ===========================================================================

def main():
    loki_model_names = ", ".join(LOKI_MODELS.keys())

    parser = argparse.ArgumentParser(description="Unified Search-Space Scalability Study")

    parser.add_argument("--dataset", type=str, default="pharma_flipped_structured",
                        help="Dataset name (e.g., pharma, pharma_flipped_structured, protrix, totto, mimic, multihiertt)")
    parser.add_argument("--task", type=str, default="DOC_TO_TABLE", choices=["DOC_TO_TABLE", "TABLE_TO_DOC"],
                        help="Task direction (use DOC_TO_TABLE only for pharma_flipped)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Data split to evaluate on")
                        
    parser.add_argument("--sizes", type=int, nargs="+", default=SCALABILITY_SIZES,
                        help="Candidate pool sizes per query (0 = full). Default: %s" % SCALABILITY_SIZES)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for negative sampling (default: %d)" % SEED)
    parser.add_argument("--focal_k", type=int, default=8,
                        help="K value for the main scalability figure (default: 8)")
    parser.add_argument("--max_test_examples", type=int, default=MAX_TEST_EXAMPLES, 
                        help="Max test examples to extract candidates from (0=all)")
    parser.add_argument("--max_queries", type=int, default=MAX_QUERIES, 
                        help="Max queries to evaluate (0=all)")

    parser.add_argument("--loki_model", type=str, default=LOKI_ACTIVE_MODEL,
                        choices=list(LOKI_MODELS.keys()))
    parser.add_argument("--use_schema_aware_loki", action=argparse.BooleanOptionalAction,
                        default=LOKI_USE_SCHEMA_AWARE_SCORER,
                        help="Use the Rewind-compatible structured LOKI scorer path. Default auto-detects header conditioning and cell-level matching from the checkpoint args.json.")
    parser.add_argument("--skip_cmdl", action="store_true")
    parser.add_argument("--skip_loki", action="store_true")
    parser.add_argument("--skip_tabstar", action="store_true")
    parser.add_argument("--skip_tabert", action="store_true")

    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--eval_row_chunk_size", type=int, default=0)
    parser.add_argument("--cache_table_embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_doc_embeddings", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--force_rerun", action="store_true",
                        help="Force re-scoring even if cached scores exist")

    args = parser.parse_args()
    resolved_loki_schema_mode = _resolve_loki_schema_mode(args.use_schema_aware_loki)
    resolved_loki_schema_representation = _resolve_loki_schema_representation(args.use_schema_aware_loki)

    info = get_dataset_info(args.dataset)
    test_file = get_split_path(args.dataset, args.split)
    dataset_format = info["format"]
    native_direction = info.get("native_direction", "TABLE_TO_DOC")
    
    # We create a nested subfolder for scalability per dataset
    out_dir = os.path.join(OUTPUT_DIR, "scalability", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    
    output_prefix = f"{args.dataset}_{args.task}_{args.split}"
    suffix = ""

    active_models = []
    if not args.skip_cmdl: active_models.append("CMDL")
    if not args.skip_loki: active_models.append("LOKI")
    if not args.skip_tabstar: active_models.append("TabSTAR")
    if not args.skip_tabert: active_models.append("TaBERT")

    if not active_models:
        print("[ERROR] All models are skipped. Nothing to do.")
        return

    # Dynamically compute scalability sizes based on candidate pool size
    # Only for unified script; other scripts use static config
    probe_scores = None  # may hold (model_name, scores_map, gt_map, elapsed) from probe
    if args.sizes == SCALABILITY_SIZES:
        # Use the first model to discover the candidate pool size,
        # then reuse its scores in the main loop (avoid double-scoring).
        first_model = None
        for m in ["CMDL", "LOKI", "TabSTAR", "TaBERT"]:
            if m in active_models:
                first_model = m
                break
        probe_scores = None  # will hold (model_name, scores_map, gt_map, elapsed)
        if first_model is not None:
            scorer = MODEL_SCORERS[first_model]
            try:
                print(f"\n[{first_model}] Probing full dataset to compute dynamic pool sizes ...")
                t0_probe = time.time()
                scores_map, gt_map = scorer(args, dataset_format, test_file, native_direction)
                elapsed_probe = time.time() - t0_probe
                probe_scores = (first_model, scores_map, gt_map, elapsed_probe)
                # Compute pool sizes from the actual per-query candidate counts
                pool_sizes = [len(scores_map[q]) for q in scores_map]
                max_pool = int(np.mean(pool_sizes))
                actual_max = max(pool_sizes) if pool_sizes else 0
                # Start at 50, double until >= max_pool
                dyn_sizes = []
                s = 50
                while s < max_pool:
                    dyn_sizes.append(s)
                    s *= 2
                if max_pool not in dyn_sizes:
                    dyn_sizes.append(max_pool)
                # Only append 0 (Full) if it would differ from max_pool
                # (i.e. some queries have more candidates than the average)
                if actual_max > max_pool:
                    dyn_sizes.append(0)
                sizes = dyn_sizes
            except Exception as e:
                print(f"[WARN] Could not compute dynamic pool sizes: {e}")
                # Fallback to static sizes
                sizes = sorted(args.sizes, key=lambda s: 999_999 if s == 0 else s)
        else:
            sizes = sorted(args.sizes, key=lambda s: 999_999 if s == 0 else s)
    else:
        sizes = sorted(args.sizes, key=lambda s: 999_999 if s == 0 else s)

    k_values = K_VALUES

    print("\n" + "=" * 70)
    print(f"  Unified Scalability Study: {args.dataset.upper()} ({args.task})")
    print("=" * 70)
    print(f"  Models:        {', '.join(active_models)}")
    print(f"  Pool sizes:    {[size_label(s) for s in sizes]}")
    print(f"  Output:        {out_dir}")
    print("=" * 70 + "\n")

    model_scores = {}

    # Reuse probe scores so the first model is not scored twice
    if probe_scores is not None:
        pm_name, pm_scores, pm_gt, pm_elapsed = probe_scores
        model_scores[pm_name] = (pm_scores, pm_gt, pm_elapsed)
        cache_path = os.path.join(out_dir, f"{output_prefix}_{pm_name}{suffix}_full_scores.json")
        cache_data = {"scores_map": pm_scores, "gt_map": pm_gt, "elapsed_time_sec": pm_elapsed}
        if pm_name == "LOKI":
            cache_data["metadata"] = {
                "use_schema_aware_loki": resolved_loki_schema_mode,
                "loki_schema_representation": resolved_loki_schema_representation,
            }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

    for model_name in active_models:
        if model_name in model_scores:
            continue  # already scored during probe

        cache_path = os.path.join(out_dir, f"{output_prefix}_{model_name}{suffix}_full_scores.json")

        if os.path.exists(cache_path) and not args.force_rerun:
            print(f"[CACHE] Loading {model_name} full scores from {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if model_name == "LOKI" and not _loki_cache_compatible(cached, resolved_loki_schema_mode, resolved_loki_schema_representation):
                print("  [CACHE] Ignoring cached LOKI scores because the schema-aware scorer configuration differs.")
            else:
                elapsed_time = cached.get("elapsed_time_sec", 0.0)
                model_scores[model_name] = (cached["scores_map"], cached["gt_map"], elapsed_time)
                continue

        print(f"\n[{model_name}] Scoring full dataset (one-time)")
        scorer = MODEL_SCORERS[model_name]
        t0 = time.time()
        try:
            scores_map, gt_map = scorer(args, dataset_format, test_file, native_direction)
        except Exception as e:
            print(f"[ERROR] {model_name} scoring failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        elapsed = time.time() - t0

        model_scores[model_name] = (scores_map, gt_map, elapsed)

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

    # Extract hard negatives from the dataset (once)
    hard_neg_map = extract_hard_negatives(test_file)

    all_scalability = {m: {} for m in active_models if m in model_scores}

    print("\n" + "=" * 70)
    print("  Phase 2: Computing metrics for each pool size")
    print("=" * 70)

    for model_name in active_models:
        if model_name not in model_scores: continue
        scores_map, gt_map, elapsed_full = model_scores[model_name]
        full_actual_sizes = [len(scores_map[q]) for q in scores_map] if scores_map else [1]
        avg_full_pool = float(np.mean(full_actual_sizes))
        cumulative_time = 0.0

        for pool_size in sizes:
            label = size_label(pool_size)
            r_scores, r_preds, r_gt = restrict_candidate_pool(
                scores_map, gt_map, pool_size, seed=args.seed,
                hard_neg_map=hard_neg_map,
            )

            # Log pool composition for first model only (same pool structure for all)
            if model_name == active_models[0]:
                _log_pool_composition(r_scores, gt_map, pool_size)

            macro = evaluate_retrieval(r_gt, r_preds, k_values, scores_map=r_scores)
            macro["pool_size"] = pool_size
            
            actual_sizes = [len(r_scores[q]) for q in r_scores]
            avg_pool = float(np.mean(actual_sizes))
            macro["actual_avg_pool_size"] = avg_pool

            simulated_time_sec = elapsed_full * (avg_pool / max(1.0, avg_full_pool))
            cumulative_time += simulated_time_sec
            macro["simulated_time_sec"] = simulated_time_sec
            macro["cumulative_time_sec"] = cumulative_time

            micro = evaluate_retrieval_micro(r_gt, r_preds, k_values, scores_map=r_scores)
            all_scalability[model_name][pool_size] = {"macro": macro, "micro": micro}
            print(f"  {model_name} @ pool={label:>6s} MAP={macro.get('MAP', 0):.4f}")

    for model_name in all_scalability:
        result_path = os.path.join(out_dir, f"{output_prefix}_{model_name}{suffix}_scalability.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in all_scalability[model_name].items()}, f, indent=2)

    try:
        plot_scalability_main(all_scalability, sizes, focal_k=args.focal_k, output_dir=out_dir, suffix=suffix, metric_type="macro")
    except Exception as e:
        pass

    try:
        _export_scalability_excel(all_scalability, sizes, k_values, out_dir, suffix)
    except Exception as e:
        pass

    print("\nDone!")

def _export_scalability_excel(all_scalability, sizes, k_values, output_dir, suffix):
    import pandas as pd
    rows = []
    for model_name, size_data in all_scalability.items():
        for size in sorted(sizes, key=lambda s: 999_999 if s == 0 else s):
            res = size_data.get(size, {})
            macro = res.get("macro", {})
            if not macro: continue
            for k in k_values:
                per_k = macro.get("per_k", {})
                kd = per_k.get(k, per_k.get(str(k), {}))
                rows.append({"Model": model_name, "Pool Size": size_label(size), "K": k, "MAP": macro.get("MAP", 0)})
    df = pd.DataFrame(rows)
    excel_path = os.path.join(output_dir, f"Scalability_Results{suffix}.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="scalability", index=False)
    print(f"[EXCEL] Saved: {excel_path}")

if __name__ == "__main__":
    main()
