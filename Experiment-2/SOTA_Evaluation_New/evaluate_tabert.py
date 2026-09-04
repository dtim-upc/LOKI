"""
evaluate_tabert.py — Evaluate the TaBERT model on Table-Text discovery.

Matches the LOKI and CMDL evaluation pipeline exactly, supporting `combined_tables`
aggregation and macro/micro metrics.

Usage:
  python evaluate_tabert.py
  python evaluate_tabert.py --max_test_examples 100
  python evaluate_tabert.py --combined_tables
"""

import os
import sys
import re
import json
import argparse
import logging
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup  (same pattern as evaluate_loki.py / evaluate_cmdl.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABERT_DIR = os.path.join(SCRIPT_DIR, "..", "TaBERT")
if TABERT_DIR not in sys.path:
    sys.path.insert(0, TABERT_DIR)

from config import (
    TEST_DATA_FILE, K_VALUES, OUTPUT_DIR,
    MAX_TEST_EXAMPLES, MAX_QUERIES, SEED,
    TABERT_MODEL_PATH,
)
from metrics import evaluate_retrieval, evaluate_retrieval_micro, print_results_table, print_results_table_micro
from evaluate_loki import load_loki_json, subsample_deterministic
from unified_data import (
    extract_tables_and_docs_unified as _extract_tables_and_docs_unified,
    extract_structured_tables as _extract_structured_tables,
    subsample_queries,
)

def extract_tables_and_docs(examples, task="DOC_TO_TABLE", dataset_format="other", native_direction="DOC_TO_TABLE"):
    return _extract_tables_and_docs_unified(examples, task=task, dataset_format=dataset_format, native_direction=native_direction)

from table_bert import TableBertModel
from table_bert.table import Table, Column


def structured_to_tabert_table(tid: str, headers: list, rows: list) -> Table:
    """Build a TaBERT ``Table`` from structured headers + content rows.

    ``headers`` is a list of column names.  ``rows`` is a list of lists,
    each inner list containing cell values aligned with *headers*.
    Falls back to ``strings_to_tabert_table_legacy`` for unstructured data.
    """
    if not headers or not rows:
        return None

    data = [[str(c) for c in row] for row in rows[:3]]

    columns = []
    for i, name in enumerate(headers):
        sample = data[0][i] if data and i < len(data[0]) and data[0][i] else name
        columns.append(Column(name, 'text', sample_value=sample))

    return Table(id=str(tid), header=columns, data=data)


_KV_PATTERN = re.compile(r'(?:^|; )([\w][\w-]*): ')


def strings_to_tabert_table_legacy(tid: str, rows: list) -> Table:
    """Fallback: convert formatted string rows to a TaBERT Table.

    Used for tables that lack structured data (non-pharma_flipped_structured).
    """
    if not rows:
        return None

    data = []
    header_names = []

    if isinstance(rows[0], dict):
        for r in rows:
            for k in r.keys():
                if k not in header_names:
                    header_names.append(k)
        for r in rows:
            data.append([str(r.get(c, "")) for c in header_names])
    else:
        parsed = [_parse_kv(r) for r in rows]
        if all(p is not None for p in parsed):
            for p in parsed:
                for k in p.keys():
                    if k not in header_names:
                        header_names.append(k)
            data = [[p.get(c, "") for c in header_names] for p in parsed]
        else:
            header_names = ["Row_Text"]
            data = [[str(r)] for r in rows]

    data = data[:3]

    columns = []
    for i, name in enumerate(header_names):
        sample = data[0][i] if data and data[0][i] else name
        columns.append(Column(name, 'text', sample_value=sample))

    return Table(id=str(tid), header=columns, data=data)


def _parse_kv(sent: str):
    """Parse ``'key: val; key: val; ...'`` into a dict, or None."""
    s = sent.rstrip('.')
    matches = list(_KV_PATTERN.finditer(s))
    if not matches:
        return None
    result = {}
    for i, m in enumerate(matches):
        val_start = m.end()
        val_end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        result[m.group(1)] = s[val_start:val_end]
    return result


def evaluate_tabert(
    test_file=None,
    max_test_examples=None,
    max_queries=None,
    seed=None,
    k_values=None,
    aggregation="max",
    encode_batch_size=32,
    task="DOC_TO_TABLE",
    dataset_format="other",
    native_direction="DOC_TO_TABLE",
    return_predictions=False,
    return_micro=False,
    return_scores=False,
    device="cuda",
    tabert_model_path=TABERT_MODEL_PATH,
    bf16=True,
    torch_compile=True,
):
    """
    Run TaBERT table-level discovery evaluation (cross-encoder).
    Matches the pipeline semantics of ``evaluate_loki.py``.

    TaBERT is a cross-encoder: every (doc-sentence, table) pair goes
    through a full joint BERT forward pass.  Unlike bi-encoder models
    (LOKI, TabSTAR), table and document representations cannot be cached
    independently without changing the scoring semantics.

    ``bf16`` and ``torch_compile`` speed up each forward pass.
    """
    test_file = test_file or TEST_DATA_FILE
    max_test_examples = max_test_examples if max_test_examples is not None else MAX_TEST_EXAMPLES
    max_queries = max_queries if max_queries is not None else MAX_QUERIES
    seed = seed if seed is not None else SEED
    k_values = k_values or K_VALUES

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    # 1. Load & subsample
    examples = load_loki_json(test_file)
    examples = subsample_deterministic(examples, max_test_examples, seed)

    # 2. Extract tables, documents, ground truth
    tables_dict, docs_dict, gt_map = extract_tables_and_docs(examples, task=task, dataset_format=dataset_format, native_direction=native_direction)

    # 2b. Extract structured table data (headers + content arrays) for proper
    #     multi-column TaBERT Table construction when available.
    tables_structured = _extract_structured_tables(examples, task=task, native_direction=native_direction)

    if max_queries and max_queries > 0:
        gt_map = subsample_queries(gt_map, max_queries, seed)

    # No source aggregation performed.

    print("  Unique tables:    %d" % len(tables_dict))
    print("  Unique documents: %d" % len(docs_dict))
    print("  GT queries:       %d" % len(gt_map))

    # 3. Load TaBERT Encoder
    print(f"\n[+] Building TaBERT model from {tabert_model_path} ...")
    model_path_abs = os.path.abspath(tabert_model_path)
    # Suppress the "Some weights … were not used" warning from HuggingFace.
    # It fires during the transient BertForMaskedLM.from_pretrained() scaffold
    # inside __init__, which is immediately overwritten by load_state_dict().
    _hf_logger = logging.getLogger("transformers.modeling_utils")
    _prev_level = _hf_logger.level
    _hf_logger.setLevel(logging.ERROR)
    model = TableBertModel.from_pretrained(model_path_abs)
    _hf_logger.setLevel(_prev_level)
    model.to(torch_device)
    model.eval()

    if bf16 and device == "cuda":
        # Wrap model.encode() with autocast instead of casting model weights
        # directly.  TaBERT's to_tensor_dict() creates Float32 masks; blanket
        # model.to(bfloat16) causes "mat1 and mat2 must have the same dtype"
        # in the linear layers.  Autocast handles per-op casting correctly and
        # keeps numerically sensitive ops (softmax, layernorm) in FP32.
        _original_encode = model.encode
        def _bf16_encode(*args, **kwargs):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                return _original_encode(*args, **kwargs)
        model.encode = _bf16_encode
        print("[TaBERT] Using BFloat16 inference (autocast)")

    if torch_compile:
        try:
            # Compile the BERT backbone directly since we call model.encode(),
            # not model.forward() — top-level torch.compile(model) would not
            # trace through encode() → encode_context_and_table() → bert().
            if hasattr(model, '_bert_model') and hasattr(model._bert_model, 'bert'):
                model._bert_model.bert = torch.compile(model._bert_model.bert)
            elif hasattr(model, '_bert_model'):
                model._bert_model = torch.compile(model._bert_model)
            print("[TaBERT] BERT backbone compiled with torch.compile()")
        except Exception as e:
            print("[TaBERT] torch.compile() unavailable: %s" % e)

    # 4. Prepare & Tokenize Tables (and pre-tokenize docs for TABLE_TO_DOC)
    table_ids = list(tables_dict.keys())

    print("\n[+] Preparing and tokenizing tables ...")
    tokenized_tables = {}
    for tid in tqdm(table_ids, desc="Tokenizing tables"):
        # Prefer structured data when available; fall back to legacy string parsing
        s = tables_structured.get(tid)
        if s:
            table_obj = structured_to_tabert_table(tid, s["headers"], s["rows"])
        else:
            table_obj = strings_to_tabert_table_legacy(tid, tables_dict[tid])
        if table_obj:
            table_obj.tokenize(model.tokenizer)
            tokenized_tables[tid] = table_obj
        else:
            tokenized_tables[tid] = None

    # Pre-tokenize all document sentences (used for TABLE_TO_DOC to avoid re-tokenizing)
    tokenized_docs = {did: [model.tokenizer.tokenize(s) for s in sents] for did, sents in docs_dict.items()}

    # ------------------------------------------------------------------
    # 5. Score (cross-encoder). Support both DOC_TO_TABLE and TABLE_TO_DOC
    # ------------------------------------------------------------------
    predictions_map = {}
    scores_map = {}

    if task.upper() == "DOC_TO_TABLE":
        query_ids = [did for did in docs_dict.keys() if did in gt_map]
        print(f"\n[*] TaBERT cross-scoring {len(query_ids)} docs vs {len(table_ids)} tables ...")

        for did in tqdm(query_ids, desc="Scoring Docs"):
            tokenized_sents = tokenized_docs.get(did, [])

            doc_scores = {}
            for tid in table_ids:
                table = tokenized_tables.get(tid)
                if table is None:
                    continue

                contexts = tokenized_sents
                tables = [table] * len(tokenized_sents)

                sent_scores = []
                for i in range(0, len(contexts), encode_batch_size):
                    batch_contexts = contexts[i:i+encode_batch_size]
                    batch_tables = tables[i:i+encode_batch_size]

                    with torch.no_grad():
                        try:
                            context_encoding, column_encoding, _ = model.encode(
                                contexts=batch_contexts,
                                tables=batch_tables
                            )
                            ctx_emb = context_encoding.mean(dim=1)
                            col_emb = column_encoding.mean(dim=1)

                            sims = F.cosine_similarity(ctx_emb, col_emb, dim=-1)
                            sent_scores.append(sims.cpu())
                        except Exception:
                            pass

                if not sent_scores:
                    doc_scores[tid] = -float('inf')
                    continue

                sent_scores = torch.cat(sent_scores, dim=0)

                if aggregation == "max" or aggregation == "mean_max":
                    score = sent_scores.max().item()
                elif aggregation == "mean":
                    score = sent_scores.mean().item()
                else:
                    score = sent_scores.max().item()

                doc_scores[tid] = score

            ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            predictions_map[did] = [tid for tid, _ in ranked]
            scores_map[did] = doc_scores

    else:
        # TABLE_TO_DOC: tables are queries, docs are candidates
        query_ids = [tid for tid in table_ids if tid in gt_map]
        print(f"\n[*] TaBERT cross-scoring {len(query_ids)} tables vs {len(tokenized_docs)} docs ...")

        for tid in tqdm(query_ids, desc="Scoring Tables"):
            table = tokenized_tables.get(tid)
            if table is None:
                continue

            table_scores = {}
            # iterate over all documents as candidates
            for did, contexts in tokenized_docs.items():
                if not contexts:
                    table_scores[did] = -float('inf')
                    continue

                tables = [table] * len(contexts)

                sent_scores = []
                for i in range(0, len(contexts), encode_batch_size):
                    batch_contexts = contexts[i:i+encode_batch_size]
                    batch_tables = tables[i:i+encode_batch_size]

                    with torch.no_grad():
                        try:
                            context_encoding, column_encoding, _ = model.encode(
                                contexts=batch_contexts,
                                tables=batch_tables
                            )
                            ctx_emb = context_encoding.mean(dim=1)
                            col_emb = column_encoding.mean(dim=1)

                            sims = F.cosine_similarity(ctx_emb, col_emb, dim=-1)
                            sent_scores.append(sims.cpu())
                        except Exception:
                            pass

                if not sent_scores:
                    table_scores[did] = -float('inf')
                    continue

                sent_scores = torch.cat(sent_scores, dim=0)

                if aggregation == "max" or aggregation == "mean_max":
                    score = sent_scores.max().item()
                elif aggregation == "mean":
                    score = sent_scores.mean().item()
                else:
                    score = sent_scores.max().item()

                table_scores[did] = score

            ranked = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
            predictions_map[tid] = [did for did, _ in ranked]
            scores_map[tid] = table_scores

    # Removed Combine Tables logic.

    # 6. Evaluate
    model_label = "TaBERT-large"
    macro_results = evaluate_retrieval(gt_map, predictions_map, k_values, scores_map=scores_map)
    macro_results["num_examples"] = len(examples)
    macro_results["max_test_examples"] = max_test_examples
    macro_results["model"] = model_label

    if return_scores:
        return macro_results, scores_map, gt_map

    if return_micro:
        micro_results = evaluate_retrieval_micro(gt_map, predictions_map, k_values, scores_map=scores_map)
        micro_results["num_examples"] = len(query_ids)
        micro_results["model"] = model_label
        
        if return_predictions:
            return macro_results, micro_results, predictions_map, gt_map
        return macro_results, micro_results

    if return_predictions:
        return macro_results, predictions_map, gt_map
    return macro_results


def main():
    parser = argparse.ArgumentParser(description="TaBERT Table-Text Discovery Evaluation")

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
    parser.add_argument("--aggregation", type=str, default="max",
                   choices=["max", "mean"],
                   help="Row/sentence aggregation strategy")
    parser.add_argument("--encode_batch_size", type=int, default=16, help="Cross encoder batch size")

    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use BFloat16 inference on CUDA (default: True). "
                             "Matches LOKI's bfloat16 precision.")
    parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Apply torch.compile() for optimized inference (default: True). "
                             "Requires PyTorch >= 2.0.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "TaBERT_pharma_results.json")

    print("\n" + "=" * 65)
    print("  Running TaBERT Evaluation")
    print("=" * 65)

    macro, micro = evaluate_tabert(
        test_file=args.test_file,
        max_test_examples=args.max_test_examples,
        max_queries=args.max_queries,
        seed=args.seed,
        task=args.task,
        dataset_format=args.dataset_format,
        aggregation=args.aggregation,
        encode_batch_size=args.encode_batch_size,
        device=args.device,
        bf16=args.bf16,
        torch_compile=args.torch_compile,
        return_micro=True,
    )

    print_results_table(macro, "TaBERT (Macro)")
    print_results_table_micro(micro, "TaBERT (Micro)")

    results = {
        "macro": macro,
        "micro": micro
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[TaBERT] Results saved to %s" % out_path)

if __name__ == "__main__":
    main()
