"""
evaluate_tabstar.py — Evaluate the frozen TabSTAR model on Table-Text discovery.

Matches the LOKI and CMDL evaluation pipeline exactly, supporting `combined_tables`
aggregation and macro/micro metrics.

Usage:
  python evaluate_tabstar.py
  python evaluate_tabstar.py --max_test_examples 100
  python evaluate_tabstar.py --combined_tables
"""

import os
import sys
import json
import argparse
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup  (same pattern as evaluate_loki.py / evaluate_cmdl.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABSTAR_DIR = os.path.join(SCRIPT_DIR, "..", "TabSTAR")
TABSTAR_SRC_DIR = os.path.join(TABSTAR_DIR, "src")
for _p in (TABSTAR_DIR, TABSTAR_SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    TEST_DATA_FILE, K_VALUES, OUTPUT_DIR,
    MAX_TEST_EXAMPLES, MAX_QUERIES, SEED,
    TABSTAR_MODEL_PATH, TABSTAR_E5_PATH,
)
from metrics import evaluate_retrieval, evaluate_retrieval_micro, print_results_table, print_results_table_micro
from evaluate_loki import load_loki_json, subsample_deterministic
from unified_data import extract_tables_and_docs_unified as _extract_tables_and_docs_unified, subsample_queries

def extract_tables_and_docs(examples, task="DOC_TO_TABLE", dataset_format="other", native_direction="DOC_TO_TABLE"):
    return _extract_tables_and_docs_unified(examples, task=task, dataset_format=dataset_format, native_direction=native_direction)

from run_tabstar_retrieval import TabStarRowEncoder, SentenceEncoder, score_table_document, verbalize_row


def evaluate_tabstar(
    test_file=None,
    max_test_examples=None,
    max_queries=None,
    seed=None,
    k_values=None,
    aggregation="mean_max",
    top_k=5,
    encode_batch_size=64,
    task="DOC_TO_TABLE",
    dataset_format="other",
    native_direction="DOC_TO_TABLE",
    return_predictions=False,
    return_micro=False,
    return_scores=False,
    device="cuda",
    tabstar_model_path=TABSTAR_MODEL_PATH,
    e5_model_path=TABSTAR_E5_PATH,
):
    """
    Run TabSTAR table-level discovery evaluation.
    Matches the pipeline semantics of `evaluate_loki.py`.
    """
    test_file = test_file or TEST_DATA_FILE
    max_test_examples = max_test_examples if max_test_examples is not None else MAX_TEST_EXAMPLES
    max_queries = max_queries if max_queries is not None else MAX_QUERIES
    seed = seed if seed is not None else SEED
    k_values = k_values or K_VALUES

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    # ------------------------------------------------------------------
    # 1. Load & subsample
    # ------------------------------------------------------------------
    examples = load_loki_json(test_file)
    examples = subsample_deterministic(examples, max_test_examples, seed)

    # ------------------------------------------------------------------
    # 2. Extract tables, documents, ground truth
    # ------------------------------------------------------------------
    tables_dict, docs_dict, gt_map = extract_tables_and_docs(examples, task=task, dataset_format=dataset_format, native_direction=native_direction)

    if max_queries and max_queries > 0:
        gt_map = subsample_queries(gt_map, max_queries, seed)

    # No source aggregation performed.

    print("  Unique tables:    %d" % len(tables_dict))
    print("  Unique documents: %d" % len(docs_dict))
    print("  GT queries:       %d" % len(gt_map))

    # ------------------------------------------------------------------
    # 3. Load Encoders
    # ------------------------------------------------------------------
    print("\n[+] Building TabSTAR row encoder (frozen, penultimate layer) ...")
    row_encoder = TabStarRowEncoder(
        device=torch_device,
        tabstar_model_path=tabstar_model_path,
        e5_model_path=e5_model_path,
    )
    print("\n[+] Building sentence encoder (E5-small, frozen) ...")
    sent_encoder = SentenceEncoder(device=torch_device, e5_model_path=e5_model_path)

    # ------------------------------------------------------------------
    # 4. Pre-encode tables and documents
    # ------------------------------------------------------------------
    table_ids = list(tables_dict.keys())
    # Determine queries and candidate pools based on task
    if task.upper() == "DOC_TO_TABLE":
        query_doc_ids = [did for did in docs_dict.keys() if did in gt_map]
        candidate_doc_ids = query_doc_ids
    else:
        query_doc_ids = None
        candidate_doc_ids = list(docs_dict.keys())

    table_embeddings = {}
    print("\n[+] Encoding all tables (rows) ...")
    for tid in tqdm(table_ids, desc="Encoding tables"):
        rows = tables_dict[tid]
        verbalized_rows = [verbalize_row(r) for r in rows]
        if not verbalized_rows:
            table_embeddings[tid] = None
            continue
        try:
            row_embs = row_encoder.encode_rows_batch(verbalized_rows, batch_size=4)
            table_embeddings[tid] = row_embs
        except Exception as e:
            print(f"  [!] Skipping table {tid}: {e}")
            table_embeddings[tid] = None

    doc_embeddings = {}
    print("\n[+] Encoding all documents (sentences) ...")
    for did in tqdm(candidate_doc_ids, desc="Encoding documents"):
        sents = docs_dict[did]
        try:
            doc_embs = sent_encoder.encode_sentences(sents, batch_size=encode_batch_size)
            doc_embeddings[did] = doc_embs
        except Exception as e:
            print(f"  [!] Skipping doc {did}: {e}")
            doc_embeddings[did] = None

    # ------------------------------------------------------------------
    # 5. Score each document against every table
    # ------------------------------------------------------------------
    predictions_map = {}
    scores_map = {}

    # Score depending on task direction
    if task.upper() == "DOC_TO_TABLE":
        print(f"\n[*] TabSTAR cross-scoring {len(query_doc_ids)} docs vs {len(table_ids)} tables ...")
        for did in tqdm(query_doc_ids, desc="Scoring"):
            sent_emb = doc_embeddings.get(did)
            if sent_emb is None:
                continue
            
            doc_scores = {}
            for tid in table_ids:
                row_emb = table_embeddings.get(tid)
                if row_emb is None:
                    continue
                score = score_table_document(row_emb, sent_emb, aggregation, top_k)
                doc_scores[tid] = score

            ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            predictions_map[did] = [tid for tid, _ in ranked]
            scores_map[did] = doc_scores
    else:
        # TABLE_TO_DOC: queries are tables, candidates are all docs
        query_table_ids = [tid for tid in table_ids if tid in gt_map]
        print(f"\n[*] TabSTAR cross-scoring {len(query_table_ids)} tables vs {len(candidate_doc_ids)} docs ...")
        for tid in tqdm(query_table_ids, desc="Scoring"):
            row_emb = table_embeddings.get(tid)
            if row_emb is None:
                continue
            table_scores = {}
            for did in candidate_doc_ids:
                sent_emb = doc_embeddings.get(did)
                if sent_emb is None:
                    continue
                score = score_table_document(row_emb, sent_emb, aggregation, top_k)
                table_scores[did] = score

            ranked = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
            predictions_map[tid] = [did for did, _ in ranked]
            scores_map[tid] = table_scores

    # Removed Combine Tables logic.

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    macro_results = evaluate_retrieval(gt_map, predictions_map, k_values, scores_map=scores_map)
    macro_results["num_examples"] = len(examples)
    macro_results["max_test_examples"] = max_test_examples
    macro_results["model"] = "TabSTAR-penultimate-frozen"

    if return_scores:
        return macro_results, scores_map, gt_map

    if return_micro:
        micro_results = evaluate_retrieval_micro(gt_map, predictions_map, k_values, scores_map=scores_map)
        micro_results["num_examples"] = len(query_doc_ids)
        micro_results["model"] = "TabSTAR-penultimate-frozen"
        
        if return_predictions:
            return macro_results, micro_results, predictions_map, gt_map
        return macro_results, micro_results

    if return_predictions:
        return macro_results, predictions_map, gt_map
    return macro_results


def main():
    parser = argparse.ArgumentParser(description="TabSTAR Table-Text Discovery Evaluation")

    parser.add_argument("--test_file", type=str, default=TEST_DATA_FILE,
                        help="Path to test JSON (default: %s)" % TEST_DATA_FILE)
    parser.add_argument("--max_test_examples", type=int, default=MAX_TEST_EXAMPLES,
                        help="Max test examples (pool subset), 0=all (default: %d)" % MAX_TEST_EXAMPLES)
    parser.add_argument("--max_queries", type=int, default=MAX_QUERIES,
                        help="Max queries to evaluate, 0=all (default: %d)" % MAX_QUERIES)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for subsampling (default: %d)" % SEED)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory (default: %s)" % OUTPUT_DIR)
                        
    parser.add_argument("--task", type=str, default="DOC_TO_TABLE",
                        help="Task direction (DOC_TO_TABLE or TABLE_TO_DOC)")
    parser.add_argument("--dataset_format", type=str, default="protrix",
                        help="Schema format (protrix or mimic)")
                        
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--aggregation", type=str, default="mean_max",
                   choices=["mean_max", "max", "mean", "top_k_mean"],
                   help="Row×sentence aggregation strategy")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "TabSTAR_pharma_results.json")

    print("\n" + "=" * 65)
    print("  Running TabSTAR Evaluation")
    print("=" * 65)

    macro, micro = evaluate_tabstar(
        test_file=args.test_file,
        max_test_examples=args.max_test_examples,
        max_queries=args.max_queries,
        seed=args.seed,
        task=args.task,
        dataset_format=args.dataset_format,
        aggregation=args.aggregation,
        device=args.device,
        return_micro=True
    )

    print_results_table(macro, "TabSTAR (Macro)")
    print_results_table_micro(micro, "TabSTAR (Micro)")

    results = {
        "macro": macro,
        "micro": micro
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[TabSTAR] Results saved to %s" % out_path)

if __name__ == "__main__":
    main()
