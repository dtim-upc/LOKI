"""
evaluate_cmdl.py — Evaluate the CMDL model on Table-Text discovery.

Matches the LOKI evaluation pipeline exactly, supporting `combined_tables`
aggregation and macro/micro metrics, but running CMDL's dual-encoder
(text_enet and col_enet) on top of WEM-featurized inputs.
"""

import os
import sys
import json
import argparse
import time
import re
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CMDL_DIR = os.path.join(SCRIPT_DIR, "..", "CMDL")
if CMDL_DIR not in sys.path:
    sys.path.insert(0, CMDL_DIR)

from config import (
    TEST_DATA_FILE, K_VALUES, OUTPUT_DIR,
    MAX_TEST_EXAMPLES, MAX_QUERIES, SEED,
    CMDL_MODEL_DIR, WEM_MODEL_PATH, WEM_DIM,
)
from metrics import evaluate_retrieval, evaluate_retrieval_micro, print_results_table, print_results_table_micro
from evaluate_loki import load_loki_json, subsample_deterministic
from unified_data import extract_tables_and_docs_unified, extract_structured_tables, subsample_queries

# Import CMDL components
from indexer.wem import WEM
from run_pharma_cmdl import EncoderNet

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
def tokenize(text: str):
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) > 1]

def evaluate_cmdl(
    test_file=None,
    max_test_examples=None,
    max_queries=None,
    seed=None,
    k_values=None,
    encode_batch_size=64,
    task="DOC_TO_TABLE",
    dataset_format="other",
    native_direction="DOC_TO_TABLE",
    return_predictions=False,
    return_micro=False,
    return_scores=False,
    device="cuda",
    cmdl_model_dir=CMDL_MODEL_DIR,
    wem_model_path=WEM_MODEL_PATH,
    wem_dim=WEM_DIM,
    hidden_size=200,
    output_size=100
):
    """
    Run CMDL table-level discovery evaluation.
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
    tables_dict, docs_dict, gt_map = extract_tables_and_docs_unified(examples, task=task, dataset_format=dataset_format, native_direction=native_direction)

    if max_queries and max_queries > 0:
        gt_map = subsample_queries(gt_map, max_queries, seed)

    # No source aggregation performed.

    print("  Unique tables/columns: %d" % len(tables_dict))
    print("  Unique documents:      %d" % len(docs_dict))
    print("  GT queries:            %d" % len(gt_map))

    # ------------------------------------------------------------------
    # 3. Load WEM and CMDL Encoders
    # ------------------------------------------------------------------
    print("\n[+] Loading FastText WEM...")
    wem = WEM(wem_model_path, wem_dim)
    
    print("[+] Loading CMDL EncoderNet models...")
    text_enet = EncoderNet(ip=wem_dim, op=output_size, hidden1=hidden_size, hidden2=hidden_size)
    col_enet = EncoderNet(ip=wem_dim, op=output_size, hidden1=hidden_size, hidden2=hidden_size)
    
    t_path = os.path.join(cmdl_model_dir, "text_enet_best.pt")
    c_path = os.path.join(cmdl_model_dir, "col_enet_best.pt")
    if not os.path.exists(t_path) or not os.path.exists(c_path):
        print(f"[ERROR] CMDL models not found in {cmdl_model_dir}. Please run run_pharma_cmdl.py first.")
        sys.exit(1)
        
    text_enet.load_state_dict(torch.load(t_path, map_location=torch_device, weights_only=True))
    col_enet.load_state_dict(torch.load(c_path, map_location=torch_device, weights_only=True))
    text_enet.to(torch_device)
    col_enet.to(torch_device)
    text_enet.eval()
    col_enet.eval()

    # ------------------------------------------------------------------
    # 4. Extract structured table data for column-level featurization
    # ------------------------------------------------------------------
    # CMDL's col_enet was trained on column-level embeddings (one per column),
    # NOT whole-table embeddings. We must featurize at the column level
    # and aggregate column scores to table-level, matching the training pipeline.
    structured_tables = extract_structured_tables(
        examples, task=task, native_direction=native_direction
    )

    table_ids = list(tables_dict.keys())
    # Determine queries and candidate pools based on task
    if task.upper() == "DOC_TO_TABLE":
        query_doc_ids = [did for did in docs_dict.keys() if did in gt_map]
        candidate_doc_ids = query_doc_ids
    else:  # TABLE_TO_DOC
        query_doc_ids = None
        candidate_doc_ids = list(docs_dict.keys())

    # ------------------------------------------------------------------
    # 4a. Featurize tables at COLUMN level (matching training pipeline)
    # ------------------------------------------------------------------
    # col_embs: list of (table_id, col_embedding_tensor)
    # This mirrors how run_pharma_cmdl.py builds per-column features:
    #   for each table -> for each column -> tokenize all cell values -> avg WEM -> col_enet
    print("\n[+] Featurizing and embedding table COLUMNS ...")
    col_emb_list = []       # [(table_id, col_embedding), ...]
    col_parent_table = []   # parallel list of parent table IDs
    fallback_count = 0

    with torch.no_grad():
        for tid in tqdm(table_ids, desc="Encoding Table Columns"):
            struct = structured_tables.get(tid)
            if struct and struct["headers"] and struct["rows"]:
                # Column-level featurization (matches training)
                headers = struct["headers"]
                rows = struct["rows"]
                n_cols = len(headers)
                for col_idx, col_hdr in enumerate(headers):
                    # Collect all cell values for this column across rows
                    col_values = []
                    for row in rows:
                        if col_idx < len(row):
                            val = str(row[col_idx]).strip()
                            if val:
                                col_values.append(val)
                    if not col_values:
                        continue
                    # Tokenize all column values and average WEM
                    col_tokens = []
                    for v in col_values:
                        col_tokens.extend(tokenize(v))
                    word_embs = [wem.get_vector(w) for w in col_tokens if wem.get_vector(w) is not None]
                    if not word_embs:
                        feat = np.zeros(wem_dim)
                    else:
                        feat = np.mean(word_embs, axis=0)
                    feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(torch_device)
                    col_emb = col_enet(feat_t).squeeze(0)
                    col_emb_list.append(col_emb)
                    col_parent_table.append(tid)
            else:
                # Fallback: no structured data, treat whole table text as one "column"
                fallback_count += 1
                full_text = " ".join([r["formatted"] if isinstance(r, dict) else str(r) for r in tables_dict[tid]])
                tokens = tokenize(full_text)
                word_embs = [wem.get_vector(w) for w in tokens if wem.get_vector(w) is not None]
                if not word_embs:
                    feat = np.zeros(wem_dim)
                else:
                    feat = np.mean(word_embs, axis=0)
                feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(torch_device)
                col_emb = col_enet(feat_t).squeeze(0)
                col_emb_list.append(col_emb)
                col_parent_table.append(tid)

    print(f"  Total column embeddings: {len(col_emb_list)} (from {len(table_ids)} tables, {fallback_count} fallback)")

    # ------------------------------------------------------------------
    # 4b. Featurize documents
    # ------------------------------------------------------------------
    print("\n[+] Featurizing and embedding documents ...")
    doc_embs = {}
    with torch.no_grad():
        for did in tqdm(candidate_doc_ids, desc="Encoding Docs"):
            full_text = " ".join(docs_dict[did])
            tokens = tokenize(full_text)
            word_embs = [wem.get_vector(w) for w in tokens if wem.get_vector(w) is not None]
            if not word_embs:
                feat = np.zeros(wem_dim)
            else:
                feat = np.mean(word_embs, axis=0)
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(torch_device)
            doc_embs[did] = text_enet(feat_t).squeeze(0)

    # ------------------------------------------------------------------
    # 5. Score: column-level distances -> table-level aggregation
    #    Matches TrainedEmbeddingsIndexer.search() from training:
    #      score(doc, table) = sum over columns of 1/(1+dist)
    # ------------------------------------------------------------------
    predictions_map = {}
    scores_map = {}

    doc_ids_list = list(doc_embs.keys())

    if len(doc_ids_list) > 0 and len(col_emb_list) > 0:
        D = torch.stack([doc_embs[d] for d in doc_ids_list])                # [N_docs, dim]
        C = torch.stack(col_emb_list)                                        # [N_cols, dim]

        # Pairwise Euclidean distances [N_docs, N_cols]
        D_exp = D.unsqueeze(1).expand(D.size(0), C.size(0), D.size(1))
        C_exp = C.unsqueeze(0).expand(D.size(0), C.size(0), C.size(1))
        col_dists = torch.sqrt(torch.pow(D_exp - C_exp, 2).sum(2) + 1e-12)  # [N_docs, N_cols]
        # Convert to similarity: 1/(1+dist)
        col_sims = 1.0 / (1.0 + col_dists)                                   # [N_docs, N_cols]

        # Aggregate column similarities to table level
        # Build mapping: table_id -> list of column indices
        table_to_col_indices = defaultdict(list)
        for idx, parent_tid in enumerate(col_parent_table):
            table_to_col_indices[parent_tid].append(idx)

        unique_table_ids = list(table_to_col_indices.keys())

        if task.upper() == "DOC_TO_TABLE":
            print(f"\n[*] CMDL column-level scoring: {len(doc_ids_list)} docs vs {len(col_emb_list)} columns -> {len(unique_table_ids)} tables ...")
            for i, did in enumerate(doc_ids_list):
                table_scores = {}
                for tid in unique_table_ids:
                    col_indices = table_to_col_indices[tid]
                    # Sum of column similarities (matching training's search aggregation)
                    table_scores[tid] = sum(col_sims[i, ci].item() for ci in col_indices)
                ranked = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                predictions_map[did] = [tid for tid, _ in ranked]
                scores_map[did] = table_scores
        else:
            print(f"\n[*] CMDL column-level scoring: {len(unique_table_ids)} tables vs {len(doc_ids_list)} docs ...")
            for tid in unique_table_ids:
                col_indices = table_to_col_indices[tid]
                doc_scores = {}
                for i, did in enumerate(doc_ids_list):
                    doc_scores[did] = sum(col_sims[i, ci].item() for ci in col_indices)
                ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                predictions_map[tid] = [did for did, _ in ranked]
                scores_map[tid] = doc_scores

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    macro_results = evaluate_retrieval(gt_map, predictions_map, k_values, scores_map=scores_map)
    macro_results["num_examples"] = len(examples)
    macro_results["max_test_examples"] = max_test_examples
    macro_results["model"] = "CMDL"

    if return_scores:
        return macro_results, scores_map, gt_map

    if return_micro:
        micro_results = evaluate_retrieval_micro(gt_map, predictions_map, k_values, scores_map=scores_map)
        micro_results["num_examples"] = len(query_doc_ids)
        micro_results["model"] = "CMDL"
        
        if return_predictions:
            return macro_results, micro_results, predictions_map, gt_map
        return macro_results, micro_results

    if return_predictions:
        return macro_results, predictions_map, gt_map
    return macro_results


def main():
    parser = argparse.ArgumentParser(description="CMDL Table-Text Discovery Evaluation")

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
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "CMDL_pharma_results.json")

    print("\n" + "=" * 65)
    print("  Running CMDL Evaluation")
    print("=" * 65)

    macro, micro = evaluate_cmdl(
        test_file=args.test_file,
        max_test_examples=args.max_test_examples,
        max_queries=args.max_queries,
        seed=args.seed,
        task=args.task,
        dataset_format=args.dataset_format,
        device=args.device,
        return_micro=True
    )

    print_results_table(macro, "CMDL (Macro)")
    print_results_table_micro(micro, "CMDL (Micro)")

    results = {
        "macro": macro,
        "micro": micro
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[CMDL] Results saved to %s" % out_path)

if __name__ == "__main__":
    main()
