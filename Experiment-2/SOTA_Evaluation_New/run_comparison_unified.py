"""
run_comparison_unified.py — Unified Multi-Model, Multi-Dataset Evaluation

Evaluates CMDL, LOKI, TabSTAR, and TaBERT on a specified dataset and task direction.
Relies on the `dataset_registry.py` for canonical data paths and formats.

Usage:
  python run_comparison_unified.py --dataset pharma --task DOC_TO_TABLE
  python run_comparison_unified.py --dataset protrix --task TABLE_TO_DOC --skip_cmdl
"""

import os
import sys
import json
import argparse
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dataset_registry import get_dataset_info, get_split_path
from config import K_VALUES, OUTPUT_DIR, MAX_TEST_EXAMPLES, SEED, LOKI_ACTIVE_MODEL, LOKI_USE_SCHEMA_AWARE_SCORER

# Import evaluators (they must be in the same directory)
from evaluate_loki import evaluate_loki
from evaluate_tabstar import evaluate_tabstar
from evaluate_tabert import evaluate_tabert
from evaluate_cmdl import evaluate_cmdl
from metrics import print_results_table, print_results_table_micro

# ===========================================================================
# Helpers
# ===========================================================================

def load_results_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_k_metric(results, k, metric):
    per_k = results.get("per_k", {})
    k_data = per_k.get(k, per_k.get(str(k), {}))
    return k_data.get(metric, 0)

def print_macro_comparison(models_results: dict, k_values, baseline_model_key="LOKI"):
    """Side-by-side comparison of MACRO metrics."""
    active_models = [m for m, res in models_results.items() if res is not None and "macro" in res]
    if not active_models: return

    baseline_match = [m for m in active_models if m.startswith(baseline_model_key)]
    baseline_key = baseline_match[0] if baseline_match else active_models[0]
    other_models = [m for m in active_models if m != baseline_key]
    ordered_models = other_models + [baseline_key]

    print("\n" + "=" * 130)
    print("  Unified Comparison (MACRO)")
    print("=" * 130)

    header = f"\n  {'Metric':<12}  {'K':<5}"
    for m in ordered_models:
        header += f"  {m:<18}"
    
    delta_cols = [f"Diff(B-{m})" for m in other_models]
    if delta_cols:
        header += "  " + "  ".join(f"{d:<15}" for d in delta_cols)
    print(header)
    print(f"  {'-' * (len(header) - 3)}")

    for metric_name in ["P@K", "R@K", "F1@K", "NDCG@K", "MRR@K"]:
        for k in k_values:
            row_str = f"  {metric_name:<12}  {k:<5}"
            vals = {}
            for m in ordered_models:
                val = _get_k_metric(models_results[m]["macro"], k, metric_name)
                vals[m] = val
                row_str += f"  {val:<18.4f}"
            
            baseline_val = vals[baseline_key]
            for m in other_models:
                delta = baseline_val - vals[m]
                marker = "^" if delta > 0 else "v" if delta < 0 else "="
                row_str += f"  {marker} {delta:+.4f}       "
            print(row_str)
        print()

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Model Table-Text Evaluation")
    
    # Core unified parameters
    parser.add_argument("--dataset", type=str, default="pharma_flipped_structured", 
                        help="Dataset name (e.g., pharma, pharma_flipped_structured, protrix, totto, mimic, multihiertt)")
    parser.add_argument("--task", type=str, default="DOC_TO_TABLE", choices=["DOC_TO_TABLE", "TABLE_TO_DOC"],
                        help="Task direction")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Data split to evaluate on")
                        
    # Standard overrides
    parser.add_argument("--max_test_examples", type=int, default=MAX_TEST_EXAMPLES,
                        help="Max test queries. 0=all (default: %d)" % MAX_TEST_EXAMPLES)
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    
    # Skips
    parser.add_argument("--skip_loki", action="store_true")
    parser.add_argument("--skip_tabstar", action="store_true")
    parser.add_argument("--skip_tabert", action="store_true")
    parser.add_argument("--skip_cmdl", action="store_true")
    
    # LOKI specific
    parser.add_argument("--loki_model", type=str, default=LOKI_ACTIVE_MODEL)
    parser.add_argument("--use_schema_aware_loki", action=argparse.BooleanOptionalAction,
                        default=LOKI_USE_SCHEMA_AWARE_SCORER,
                        help="Use the Rewind-compatible structured LOKI scorer path. Default auto-detects header conditioning and cell-level matching from the checkpoint args.json.")
    
    args = parser.parse_args()
    
    # 1. Resolve Dataset Configuration
    info = get_dataset_info(args.dataset)
    test_file = get_split_path(args.dataset, args.split)
    dataset_format = info["format"]
    native_direction = info.get("native_direction", "TABLE_TO_DOC")
    
    print("\n" + "="*80)
    print(f"  UNIFIED PIPELINE EVALUATION")
    print("="*80)
    print(f"  Dataset: {args.dataset.upper()} ({info['description']})")
    print(f"  Format : {dataset_format}")
    print(f"  NatDir : {native_direction}")
    print(f"  Task   : {args.task}")
    print(f"  Split  : {args.split} -> {test_file}")
    print(f"  Max Ex : {args.max_test_examples}")
    print("="*80 + "\n")
    
    output_prefix = f"{args.dataset}_{args.task}_{args.split}"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine outputs dictionary
    models_out = {
        "CMDL": os.path.join(OUTPUT_DIR, f"{output_prefix}_CMDL.json") if not args.skip_cmdl else None,
        "LOKI": os.path.join(OUTPUT_DIR, f"{output_prefix}_LOKI.json") if not args.skip_loki else None,
        "TabSTAR": os.path.join(OUTPUT_DIR, f"{output_prefix}_TabSTAR.json") if not args.skip_tabstar else None,
        "TaBERT": os.path.join(OUTPUT_DIR, f"{output_prefix}_TaBERT.json") if not args.skip_tabert else None,
    }
    
    models_results = {m: None for m in models_out.keys()}
    
    # 2. Run Evaluators
    # CMDL
    if not args.skip_cmdl:
        print("\n\n>>> Running CMDL ...")
        t0 = time.time()
        macro, micro = evaluate_cmdl(
            test_file=test_file,
            max_test_examples=args.max_test_examples,
            seed=args.seed,
            task=args.task,
            dataset_format=dataset_format,
            native_direction=native_direction,
            return_micro=True
        )
        res = {"macro": macro, "micro": micro}
        with open(models_out["CMDL"], "w") as f: json.dump(res, f, indent=2)
        models_results["CMDL"] = res
        print(f"    Done in {time.time()-t0:.1f}s")
        
    # TabSTAR
    if not args.skip_tabstar:
        print("\n\n>>> Running TabSTAR ...")
        t0 = time.time()
        macro, micro = evaluate_tabstar(
            test_file=test_file,
            max_test_examples=args.max_test_examples,
            seed=args.seed,
            task=args.task,
            dataset_format=dataset_format,
            native_direction=native_direction,
            return_micro=True
        )
        res = {"macro": macro, "micro": micro}
        with open(models_out["TabSTAR"], "w") as f: json.dump(res, f, indent=2)
        models_results["TabSTAR"] = res
        print(f"    Done in {time.time()-t0:.1f}s")

    # TaBERT
    if not args.skip_tabert:
        print("\n\n>>> Running TaBERT ...")
        t0 = time.time()
        macro, micro = evaluate_tabert(
            test_file=test_file,
            max_test_examples=args.max_test_examples,
            seed=args.seed,
            task=args.task,
            dataset_format=dataset_format,
            native_direction=native_direction,
            return_micro=True
        )
        res = {"macro": macro, "micro": micro}
        with open(models_out["TaBERT"], "w") as f: json.dump(res, f, indent=2)
        models_results["TaBERT"] = res
        print(f"    Done in {time.time()-t0:.1f}s")

    # LOKI
    if not args.skip_loki:
        print("\n\n>>> Running LOKI ...")
        t0 = time.time()
        macro, micro = evaluate_loki(
            test_file=test_file,
            max_test_examples=args.max_test_examples,
            seed=args.seed,
            loki_model_key=args.loki_model,
            task=args.task,
            dataset_format=dataset_format,
            native_direction=native_direction,
            return_micro=True,
            use_schema_aware_loki=args.use_schema_aware_loki,
        )
        res = {"macro": macro, "micro": micro}
        with open(models_out["LOKI"], "w") as f: json.dump(res, f, indent=2)
        models_results["LOKI"] = res
        print(f"    Done in {time.time()-t0:.1f}s")
        
    # 3. Print combined table
    print_macro_comparison(models_results, K_VALUES)
    print("\n[✔] Unified Evaluation Complete!")


if __name__ == "__main__":
    main()
