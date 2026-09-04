"""
TabSTAR Penultimate-Layer Retrieval
====================================

Uses the original TabSTAR architecture (frozen, pretrained) for cross-modal
table-text retrieval.  The key idea:

    Row embedding  = mean-pool over InteractionEncoder output tokens  (384-d)
    Sent embedding = E5-small CLS token with "passage:" prefix        (384-d)
    Score          = cosine similarity between row and sentence embeddings

No LOKI code is used.  The evaluation protocol (pos > neg → correct) matches
LOKI Rewind's format so the results can be compared directly.

Author : auto-generated (TabSTAR-native retrieval bridge)
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Import SOTA evaluation modules directly
SOTA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SOTA_Evaluation_New"))
if SOTA_DIR not in sys.path:
    sys.path.insert(0, SOTA_DIR)

from evaluate_loki import extract_tables_and_docs, subsample_deterministic
from metrics import evaluate_retrieval, evaluate_retrieval_micro, print_results_table, print_results_table_micro

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


# ── TabSTAR imports ──────────────────────────────────────────────────────────
# We import TabSTAR architecture components directly — no wrapper model needed.
from tabstar.arch.config import TabStarConfig, D_MODEL, E5_SMALL
from tabstar.arch.interaction import InteractionEncoder
from tabstar.arch.fusion import NumericalFusion
from tabstar.constants import E5_SMALL_LOCAL_PATH


# ═══════════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING  (pharma row-level JSON – same format used by Rewind / LOKI)
# ═══════════════════════════════════════════════════════════════════════════════

def load_examples(path: str) -> List[Dict[str, Any]]:
    """Load row-level dataset from JSON."""
    print(f"Loading dataset from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  -> {len(data)} examples loaded.")
    return data


# extract_rows and extract_sentences have been replaced by SOTA's extract_tables_and_docs


# ═══════════════════════════════════════════════════════════════════════════════
#  2. TABSTAR ROW ENCODER  (frozen — uses InteractionEncoder penultimate layer)
# ═══════════════════════════════════════════════════════════════════════════════

class TabStarRowEncoder(torch.nn.Module):
    """
    Encodes a table row using TabSTAR's native pipeline:

        verbalized col-val pairs
            → E5-small  [CLS]  embedding per column   (n_cols, 384)
            → NumericalFusion   (text ⊕ zeros)         (n_cols, 384)
            → InteractionEncoder  (6-layer Transformer) (n_cols, 384)   ← penultimate
            → mean-pool  →  L2-normalise               (384,)

    The PredictionHead is **never** instantiated — we stop at the
    InteractionEncoder output, which is the penultimate representation.
    """

    def __init__(self, device: torch.device = None,
                 tabstar_model_path: str = "alana89/TabSTAR",
                 e5_model_path: str = None):
        super().__init__()
        from tabstar.arch.arch import TabStarModel
        from tabstar.arch.config import TabStarConfig

        print(f"Loading TabSTAR pre-trained weights from {tabstar_model_path} ...")
        if e5_model_path:
            config = TabStarConfig.from_pretrained(tabstar_model_path)
            config._e5_local_path = e5_model_path
            tabstar = TabStarModel.from_pretrained(tabstar_model_path, config=config)
        else:
            tabstar = TabStarModel.from_pretrained(tabstar_model_path)

        self.text_encoder = tabstar.text_encoder
        self.tokenizer = tabstar.tokenizer
        self.numerical_fusion = tabstar.numerical_fusion
        self.tabular_encoder = tabstar.tabular_encoder

        if device is not None:
            self.to(device)

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_row(self, col_val_strings: List[str]) -> torch.Tensor:
        """
        Encode a single row (list of verbalized column-value strings) into
        a 384-d embedding vector.

        Args:
            col_val_strings:  e.g. ["col1: val1", "col2: val2", ...]

        Returns:
            Tensor of shape (384,), L2-normalised.
        """
        n_cols = len(col_val_strings)

        # ── E5-small: CLS embedding for each column value ──
        inputs = self.tokenizer(
            col_val_strings,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        # CLS token = index 0
        cls_embs = outputs.last_hidden_state[:, 0, :]  # (n_cols, 384)

        # Reshape to (1, n_cols, 384) for batch-dim expected by fusion/encoder
        cls_embs = cls_embs.unsqueeze(0)

        # ── NumericalFusion (pure-text: x_num = zeros) ──
        x_num = torch.zeros(1, n_cols, dtype=cls_embs.dtype, device=self.device)
        fused = self.numerical_fusion(textual_embeddings=cls_embs, x_num=x_num)
        # fused: (1, n_cols, 384)

        # ── InteractionEncoder (6-layer Transformer) ──
        encoded = self.tabular_encoder(fused)  # (1, n_cols, 384)

        # ── Mean-pool over columns → single 384-d row vector ──
        row_emb = encoded.mean(dim=1)  # (1, 384)
        row_emb = F.normalize(row_emb, p=2, dim=-1)
        return row_emb.squeeze(0)  # (384,)

    @torch.no_grad()
    def encode_rows_batch(self, rows: List[List[str]], batch_size: int = 8) -> torch.Tensor:
        """
        Encode multiple rows.  Each row is a list of column-value strings.

        Returns:  Tensor of shape (num_rows, 384), L2-normalised.
        """
        all_embs = []
        for i in range(0, len(rows), batch_size):
            batch = rows[i: i + batch_size]
            embs = torch.stack([self.encode_row(r) for r in batch])
            all_embs.append(embs)
        return torch.cat(all_embs, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. SENTENCE ENCODER  (E5-small with "passage:" prefix — same backbone)
# ═══════════════════════════════════════════════════════════════════════════════

class SentenceEncoder(torch.nn.Module):
    """
    Encodes free-text sentences using E5-small.
    Uses the "passage:" prefix recommended by E5 for passage-level encoding.
    """

    def __init__(self, device: torch.device = None, e5_model_path: str = None):
        super().__init__()
        e5_path = e5_model_path or E5_SMALL_LOCAL_PATH or E5_SMALL
        self.text_encoder = AutoModel.from_pretrained(e5_path)
        self.tokenizer = AutoTokenizer.from_pretrained(e5_path)

        if device is not None:
            self.to(device)

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_sentences(self, sentences: List[str], batch_size: int = 64) -> torch.Tensor:
        """
        Encode sentences into 384-d embeddings with "passage:" prefix.

        Returns:  Tensor of shape (num_sentences, 384), L2-normalised.
        """
        # E5-small: prefix with "passage:" for passage-level encoding
        prefixed = [f"passage: {s}" for s in sentences]

        all_embs = []
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i: i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.text_encoder(**inputs)
            cls_embs = outputs.last_hidden_state[:, 0, :]  # (B, 384)
            cls_embs = F.normalize(cls_embs, p=2, dim=-1)
            all_embs.append(cls_embs)
        return torch.cat(all_embs, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. SCORING  (cosine similarity + aggregation)
# ═══════════════════════════════════════════════════════════════════════════════

def score_table_document(
    row_embeddings: torch.Tensor,      # (num_rows, 384)
    sentence_embeddings: torch.Tensor,  # (num_sents, 384)
    aggregation: str = "mean_max",
    top_k: int = 5,
) -> float:
    """
    Compute a global table–document similarity score from row and sentence
    embeddings using cosine similarity.

    Aggregation methods:
        mean_max  :  For each row, take the max similarity to any sentence,
                     then average over rows.  (default)
        max       :  Global maximum over the full row×sentence sim matrix.
        mean      :  Global mean over the full row×sentence sim matrix.
        top_k_mean:  Average the top-k pair similarities.
    """
    # Cosine sim matrix  (num_rows, num_sents)
    sim = torch.mm(row_embeddings, sentence_embeddings.t())

    if aggregation == "mean_max":
        # For each row: best-matching sentence → then mean over rows
        return sim.max(dim=1).values.mean().item()

    elif aggregation == "max":
        return sim.max().item()

    elif aggregation == "mean":
        return sim.mean().item()

    elif aggregation == "top_k_mean":
        flat = sim.flatten()
        k = min(top_k, flat.numel())
        return flat.topk(k).values.mean().item()

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


# ═══════════════════════════════════════════════════════════════════════════════
#  5. ROW VERBALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def verbalize_row(row_text: str) -> List[str]:
    """
    Convert a pre-formatted row string into individual column-value text
    chunks suitable for TabSTAR's per-column E5 encoding.

    Input format (pharma data):
        "col1: val1; col2: val2; col3: val3"

    Output:
        ["col1: val1", "col2: val2", "col3: val3"]

    If the row is already a single string without separators, we fall back
    to treating the entire row as a single "column".
    """
    if ";" in row_text:
        parts = [p.strip() for p in row_text.split(";") if p.strip()]
        return parts if parts else [row_text]
    elif "\t" in row_text:
        parts = [p.strip() for p in row_text.split("\t") if p.strip()]
        return parts if parts else [row_text]
    else:
        # Treat the whole row as a single token
        return [row_text]


# ═══════════════════════════════════════════════════════════════════════════════
#  6. EVALUATION  (matches LOKI Rewind output format)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    row_encoder: TabStarRowEncoder,
    sent_encoder: SentenceEncoder,
    examples: List[Dict[str, Any]],
    aggregation: str = "mean_max",
    top_k: int = 5,
    max_examples: int = 0,
    seed: int = 42,
    task: str = "DOC_TO_TABLE",
    dataset_format: str = "other",
    native_direction: str = "DOC_TO_TABLE",
) -> Dict[str, Any]:
    """
    Evaluate TabSTAR retrieval on the dataset-wide discovery task.
    
    1. Subsamples using identical SOTA seed/logic
    2. Extracts all tables/docs
    3. Scores every query doc vs every table
    4. Computes MACRO and MICRO metrics perfectly aligned with LOKI
    """
    examples = subsample_deterministic(examples, max_examples, seed)
    tables_dict, docs_dict, gt_map = extract_tables_and_docs(
        examples, task=task, dataset_format=dataset_format, native_direction=native_direction
    )

    query_doc_ids = [did for did in docs_dict.keys() if did in gt_map]
    table_ids = list(tables_dict.keys())

    print(f"\n  Unique tables:    {len(tables_dict)}")
    print(f"  Unique documents: {len(docs_dict)}")
    print(f"  GT queries:       {len(gt_map)}")

    table_embeddings = {}
    print("\n[+] Encoding all tables (rows) ...")
    for tid in tqdm(table_ids, desc="Encoding tables"):
        rows = tables_dict[tid]
        verbalized_rows = [verbalize_row(r) for r in rows]
        if not verbalized_rows:
            table_embeddings[tid] = None
            continue
        try:
            # We use small internal batch size for rows
            row_embs = row_encoder.encode_rows_batch(verbalized_rows, batch_size=4)
            table_embeddings[tid] = row_embs
        except Exception as e:
            print(f"  [!] Skipping table {tid}: {e}")
            table_embeddings[tid] = None

    doc_embeddings = {}
    print("\n[+] Encoding all documents (sentences) ...")
    for did in tqdm(query_doc_ids, desc="Encoding documents"):
        sents = docs_dict[did]
        try:
            doc_embs = sent_encoder.encode_sentences(sents, batch_size=64)
            doc_embeddings[did] = doc_embs
        except Exception as e:
            print(f"  [!] Skipping doc {did}: {e}")
            doc_embeddings[did] = None

    predictions_map = {}
    scores_map = {}
    
    print("\n[*] Cross-scoring all docs vs all tables ...")
    for did in tqdm(query_doc_ids, desc="Scoring"):
        sent_emb = doc_embeddings.get(did)
        if sent_emb is None:
            continue
            
        doc_scores = {}
        for tid in table_ids:
            row_emb = table_embeddings.get(tid)
            if row_emb is None:
                continue
            
            # NxM cosine similarity + aggregation
            score = score_table_document(row_emb, sent_emb, aggregation, top_k)
            doc_scores[tid] = score

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        predictions_map[did] = [tid for tid, _ in ranked]
        scores_map[did] = doc_scores

    k_values = [1, 5, 10, 20, 50, 100]
    
    macro_results = evaluate_retrieval(gt_map, predictions_map, k_values, scores_map=scores_map)
    macro_results["num_examples"] = len(examples)
    macro_results["max_test_examples"] = max_examples

    micro_results = evaluate_retrieval_micro(gt_map, predictions_map, k_values, scores_map=scores_map)
    micro_results["num_examples"] = len(query_doc_ids)

    print_results_table(macro_results, "TabSTAR-penultimate-frozen (Macro)")
    print_results_table_micro(micro_results, "TabSTAR-penultimate-frozen (Micro)")
    
    return {"macro": macro_results, "micro": micro_results}


# ═══════════════════════════════════════════════════════════════════════════════
#  7. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="TabSTAR penultimate-layer retrieval (frozen, no LOKI code)"
    )

    # Data
    p.add_argument("--eval_file", type=str,
                   default="pharma_data/test_row_level.json",
                   help="Path to row-level JSON (val or test split)")

    p.add_argument("--max_examples", type=int, default=0,
                   help="Limit evaluation to first N examples (0 = all)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for subsampling (matches CMDL/LOKI)")
    p.add_argument("--task", type=str, default="DOC_TO_TABLE",
                   help="Task direction (DOC_TO_TABLE or TABLE_TO_DOC)")
    p.add_argument("--dataset_format", type=str, default="other",
                   help="Schema format (protrix, mimic, or other)")
    p.add_argument("--native_direction", type=str, default="DOC_TO_TABLE",
                   help="Native direction of the dataset file")

    # Scoring
    p.add_argument("--aggregation", type=str, default="mean_max",
                   choices=["mean_max", "max", "mean", "top_k_mean"],
                   help="Row×sentence aggregation strategy")
    p.add_argument("--top_k", type=int, default=5,
                   help="k for top_k_mean aggregation")

    # Hardware
    p.add_argument("--device", type=str, default="auto",
                   help="'cuda', 'cpu', or 'auto'")

    # Output
    p.add_argument("--output", type=str, default="output/retrieval_results.json",
                   help="Path to save results JSON (default: print only)")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ── Build encoders ──
    print("\n[+] Building TabSTAR row encoder (frozen, penultimate layer) ...")
    row_encoder = TabStarRowEncoder(device=device)
    print(f"   InteractionEncoder: {sum(p.numel() for p in row_encoder.tabular_encoder.parameters()):,} params")
    print(f"   NumericalFusion:    {sum(p.numel() for p in row_encoder.numerical_fusion.parameters()):,} params")

    print("\n[+] Building sentence encoder (E5-small, frozen) ...")
    sent_encoder = SentenceEncoder(device=device)

    # ── Load data ──
    examples = load_examples(args.eval_file)

    # ── Evaluate ──
    print(f"\n[*] Evaluating with aggregation='{args.aggregation}' ...")
    t0 = time.time()
    metrics = evaluate(
        row_encoder=row_encoder,
        sent_encoder=sent_encoder,
        examples=examples,
        aggregation=args.aggregation,
        top_k=args.top_k,
        max_examples=args.max_examples,
        seed=args.seed,
        task=args.task,
        dataset_format=args.dataset_format,
        native_direction=args.native_direction,
    )
    elapsed = time.time() - t0

    metrics["elapsed_seconds"] = round(elapsed, 2)
    metrics["model"] = "TabSTAR-penultimate-frozen"
    metrics["backbone"] = E5_SMALL

    # ── Save ──
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n[*] Results saved to {out}")

    # ── TODO (if results poor) ──
    # Fine-tuning ablation:
    #   - Add a small trainable projection head on top of the frozen embeddings
    #   - Use contrastive loss (e.g., InfoNCE) to align row↔sentence representations
    #   - Keep InteractionEncoder frozen, only train the projection head
    #   - This would be reported as a separate row in the paper


if __name__ == "__main__":
    main()
