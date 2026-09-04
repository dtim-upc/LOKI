"""
Evaluation script for fine-tuned TaBERT on contrastive datasets.

Supports two evaluation levels:
  1. Table-level contrastive accuracy (positive score > negative score)
  2. Row-sentence grounding (P/R/F1 matching LOKI's protocol)

Supports both flipped and non-flipped dataset formats via --is_flipped.

Usage:
    python TaBERT/evaluate_finetuned.py --model_dir finetuned_tabert_pharma/best \
                                  --data_dir Datasets/pharma_flipped_structured --is_flipped True
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from data_loader import (
    load_row_level_dataset,
    example_to_tabert_table,
    example_to_single_row_tables,
    structured_to_tabert_table,
    _extract_table_structured,
    _extract_sentences,
    load_annotations,
)
from model_wrapper import TaBERTForContrastive, _resolve_path, _resolve_model_path


def evaluate_table_level(
    model: TaBERTForContrastive,
    examples: List[Dict[str, Any]],
    is_flipped: bool = True,
    sample_row_num: int = 3,
    max_context_len: int = 128,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Table-level contrastive evaluation.

    is_flipped=True:  anchor=doc sentences (context), pos/neg=structured tables
    is_flipped=False: anchor=table rows (table), pos/neg=doc sentences (context)
    """
    model.eval()
    device = model.device
    tokenizer = model.tokenizer

    correct = 0
    total = 0
    pos_scores_all = []
    neg_scores_all = []

    for example in tqdm(examples, desc="Table-level eval"):
        with torch.no_grad():
            if is_flipped:
                # Anchor = doc sentences (context), pos/neg = structured tables
                anchor_doc_sents = example.get("anchor_sentences", [])
                if not anchor_doc_sents:
                    continue
                context_tokens = tokenizer.tokenize(anchor_doc_sents[0])[:max_context_len]

                # Build positive table from structured data
                primary = example.get("primary_positive", {})
                pos_data = _extract_table_structured(primary)
                if pos_data is None:
                    continue
                pos_headers, pos_rows = pos_data
                pos_table = structured_to_tabert_table(
                    table_id=str(primary.get("id", "pos")),
                    headers=pos_headers,
                    row_content_lists=pos_rows,
                    tokenizer=tokenizer,
                    max_rows=sample_row_num,
                )
                if pos_table is None:
                    continue
                try:
                    with autocast(device_type='cuda', enabled=use_amp):
                        best_pos_score = model.score_single(context_tokens, pos_table).item()
                except Exception:
                    continue

                # Score each negative table
                best_neg_score = float('-inf')
                for neg in example.get("negatives", []):
                    neg_data = _extract_table_structured(neg)
                    if neg_data is None:
                        continue
                    neg_headers, neg_rows = neg_data
                    neg_table = structured_to_tabert_table(
                        table_id=str(neg.get("id", "neg")),
                        headers=neg_headers,
                        row_content_lists=neg_rows,
                        tokenizer=tokenizer,
                        max_rows=sample_row_num,
                    )
                    if neg_table is None:
                        continue
                    try:
                        with autocast(device_type='cuda', enabled=use_amp):
                            s = model.score_single(context_tokens, neg_table).item()
                        best_neg_score = max(best_neg_score, s)
                    except Exception:
                        continue
            else:
                # Anchor = table, pos/neg = doc sentences (context)
                table = example_to_tabert_table(example, tokenizer=tokenizer)
                if table is None:
                    continue
                if sample_row_num > 0 and len(table.data) > sample_row_num:
                    table = table.with_rows(table.data[:sample_row_num])
                    table.tokenize(tokenizer)

                # Extract positive and negative sentences
                primary = example.get("primary_positive", {})
                pos_sents = _extract_sentences(primary.get("sentences", []))
                if not pos_sents:
                    continue

                # Score each positive sentence
                pos_sentence_scores = []
                for sent in pos_sents:
                    tokens = tokenizer.tokenize(sent)[:max_context_len]
                    try:
                        with autocast(device_type='cuda', enabled=use_amp):
                            s = model.score_single(tokens, table)
                        pos_sentence_scores.append(s.item())
                    except Exception:
                        continue
                if not pos_sentence_scores:
                    continue
                best_pos_score = max(pos_sentence_scores)

                # Score each negative sentence
                best_neg_score = float('-inf')
                for neg in example.get("negatives", []):
                    neg_sents = _extract_sentences(neg.get("sentences", []))
                    for sent in neg_sents:
                        tokens = tokenizer.tokenize(sent)[:max_context_len]
                        try:
                            with autocast(device_type='cuda', enabled=use_amp):
                                s = model.score_single(tokens, table)
                            best_neg_score = max(best_neg_score, s.item())
                        except Exception:
                            continue

        if best_neg_score == float('-inf'):
            continue

        pos_scores_all.append(best_pos_score)
        neg_scores_all.append(best_neg_score)

        if best_pos_score > best_neg_score:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    mean_pos = np.mean(pos_scores_all) if pos_scores_all else 0.0
    mean_neg = np.mean(neg_scores_all) if neg_scores_all else 0.0
    separation = mean_pos - mean_neg

    results = {
        'table_level_accuracy': accuracy,
        'total_examples': total,
        'correct': correct,
        'mean_positive_score': float(mean_pos),
        'mean_negative_score': float(mean_neg),
        'score_separation': float(separation),
    }

    print(f"\n  Table-Level Results:")
    print(f"    Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"    Mean pos score: {mean_pos:.4f}")
    print(f"    Mean neg score: {mean_neg:.4f}")
    print(f"    Separation: {separation:.4f}")

    return results


def evaluate_row_sentence(
    model: TaBERTForContrastive,
    examples: List[Dict[str, Any]],
    annotations: Dict[int, List[List[int]]],
    is_flipped: bool = True,
    max_context_len: int = 128,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Row-sentence level grounding evaluation.

    is_flipped=True:  rows = anchor doc sentences, sentences = positive table rows
    is_flipped=False: rows = anchor table rows, sentences = positive doc sentences
    """
    model.eval()
    device = model.device
    tokenizer = model.tokenizer

    all_f1 = []
    all_precision = []
    all_recall = []

    annotated_count = 0

    for example in tqdm(examples, desc="Row-sentence eval"):
        anchor_id = example.get("anchor_id")
        if anchor_id is None or anchor_id not in annotations:
            continue

        highlighted_cells = annotations[anchor_id]
        if not highlighted_cells:
            continue

        primary = example.get("primary_positive", {})

        if is_flipped:
            # Rows = anchor doc sentences (each scored as context)
            # Sentences = positive table rows (each → single-row structured Table)
            anchor_sents = example.get("anchor_sentences", [])
            if not anchor_sents:
                continue
            pos_data = _extract_table_structured(primary)
            if pos_data is None:
                continue
            pos_headers, pos_rows = pos_data
            num_rows = len(anchor_sents)
            num_sents = len(pos_rows)

            score_matrix = np.zeros((num_rows, num_sents))
            with torch.no_grad():
                for r_idx, doc_sent in enumerate(anchor_sents):
                    context_tokens = tokenizer.tokenize(doc_sent)[:max_context_len]
                    for s_idx, row_content in enumerate(pos_rows):
                        row_table = structured_to_tabert_table(
                            table_id="eval",
                            headers=pos_headers,
                            row_content_lists=[row_content],
                            tokenizer=tokenizer,
                            max_rows=1,
                        )
                        if row_table is None:
                            continue
                        try:
                            with autocast(device_type='cuda', enabled=use_amp):
                                s = model.score_single(context_tokens, row_table)
                            score_matrix[r_idx, s_idx] = s.item()
                        except Exception:
                            score_matrix[r_idx, s_idx] = 0.0
        else:
            # Rows = anchor table rows (each → single-row Table)
            # Sentences = positive doc sentences (each scored as context)
            sentences = _extract_sentences(primary.get("sentences", []))
            if not sentences:
                continue
            row_tables = example_to_single_row_tables(example, tokenizer=tokenizer)
            if not row_tables:
                continue
            num_rows = len(row_tables)
            num_sents = len(sentences)

            score_matrix = np.zeros((num_rows, num_sents))
            with torch.no_grad():
                for r_idx, row_table in enumerate(row_tables):
                    for s_idx, sent in enumerate(sentences):
                        tokens = tokenizer.tokenize(sent)[:max_context_len]
                        try:
                            with autocast(device_type='cuda', enabled=use_amp):
                                s = model.score_single(tokens, row_table)
                            score_matrix[r_idx, s_idx] = s.item()
                        except Exception:
                            score_matrix[r_idx, s_idx] = 0.0

        # Convert highlighted_cells to 0-based (row_idx, sent_idx) pairs
        gt_pairs = set()
        for cell in highlighted_cells:
            if len(cell) >= 2:
                row_idx = cell[0] - 1  # 1-based -> 0-based
                sent_idx = cell[1]     # already 0-based
                if 0 <= row_idx < num_rows and 0 <= sent_idx < num_sents:
                    gt_pairs.add((row_idx, sent_idx))

        if not gt_pairs:
            continue

        # Compute F1 using optimal threshold
        gt_scores = [score_matrix[r, s] for r, s in gt_pairs]
        non_gt_scores = [
            score_matrix[r, s]
            for r in range(num_rows)
            for s in range(num_sents)
            if (r, s) not in gt_pairs
        ]

        if gt_scores and non_gt_scores:
            threshold = (np.mean(gt_scores) + np.mean(non_gt_scores)) / 2.0
        else:
            threshold = np.median(score_matrix.flatten())

        predicted_pairs = set()
        for r in range(num_rows):
            for s in range(num_sents):
                if score_matrix[r, s] >= threshold:
                    predicted_pairs.add((r, s))

        tp = len(gt_pairs & predicted_pairs)
        fp = len(predicted_pairs - gt_pairs)
        fn = len(gt_pairs - predicted_pairs)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        annotated_count += 1

    if not all_f1:
        print("  No annotated examples found for row-sentence evaluation.")
        return {}

    results = {
        'row_sent_precision': float(np.mean(all_precision)),
        'row_sent_recall': float(np.mean(all_recall)),
        'row_sent_f1': float(np.mean(all_f1)),
        'annotated_examples': annotated_count,
    }

    print(f"\n  Row-Sentence Results ({annotated_count} annotated examples):")
    print(f"    Precision: {results['row_sent_precision']:.4f}")
    print(f"    Recall:    {results['row_sent_recall']:.4f}")
    print(f"    F1:        {results['row_sent_f1']:.4f}")

    return results


def evaluate_frozen_baseline(
    model: TaBERTForContrastive,
    examples: List[Dict[str, Any]],
    is_flipped: bool = True,
    sample_row_num: int = 3,
    max_context_len: int = 128,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Stage-0 frozen baseline: evaluate without any fine-tuning.
    Same as table-level eval but explicitly labeled as frozen baseline.
    """
    print("\nRunning frozen (Stage-0) baseline evaluation...")
    results = evaluate_table_level(
        model, examples, is_flipped, sample_row_num, max_context_len, use_amp,
    )
    return {f"frozen_{k}": v for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned TaBERT")
    parser.add_argument("--model_dir", type=str, default="finetuned_tabert_pharma/best",
                        help="Path to fine-tuned model directory")
    parser.add_argument("--base_model_path", type=str, default="pretrained/tabert_large_k3/model.bin",
                        help="Path to base pretrained model")
    parser.add_argument("--data_dir", type=str, default="../Datasets/pharma_flipped_structured",
                        help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate (test/val)")
    parser.add_argument("--sample_row_num", type=int, default=3)
    parser.add_argument("--max_context_len", type=int, default=512)
    parser.add_argument("--use_amp", type=bool, default=True)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--run_frozen_baseline", action="store_true",
                        help="Also run frozen (no fine-tuning) baseline")
    parser.add_argument("--run_row_sentence", default=True,
                        help="Also run row-sentence level evaluation")
    parser.add_argument("--is_flipped", type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=True,
                        help="True if dataset is flipped (anchors=docs, pos/neg=tables)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # All paths resolve relative to the TaBERT script directory
    args.data_dir = _resolve_model_path(args.data_dir)
    args.base_model_path = _resolve_model_path(args.base_model_path)
    args.model_dir = _resolve_model_path(args.model_dir)

    # Load model
    print(f"Loading fine-tuned model from {args.model_dir}...")
    model = TaBERTForContrastive.load_finetuned(
        args.model_dir,
        base_model_path=args.base_model_path,
    )
    model.to(device)
    model.eval()

    # Load test data
    test_path = os.path.join(args.data_dir, f"{args.split}_row_level.json")
    print(f"Loading {args.split} data from {test_path}...")
    test_data = load_row_level_dataset(test_path)
    print(f"Loaded {len(test_data)} examples")

    all_results = {}

    # Table-level evaluation
    print("\n" + "=" * 60)
    print("Table-Level Contrastive Evaluation")
    print("=" * 60)
    table_results = evaluate_table_level(
        model, test_data, args.is_flipped, args.sample_row_num, args.max_context_len, args.use_amp,
    )
    all_results.update(table_results)

    # Row-sentence evaluation
    if args.run_row_sentence:
        annotation_file = os.path.join(args.data_dir, "Annotated_Test.json")
        annotations = load_annotations(annotation_file)
        if annotations:
            print("\n" + "=" * 60)
            print("Row-Sentence Grounding Evaluation")
            print("=" * 60)
            rs_results = evaluate_row_sentence(
                model, test_data, annotations, args.is_flipped, args.max_context_len, args.use_amp,
            )
            all_results.update(rs_results)

    # Frozen baseline
    if args.run_frozen_baseline:
        print("\n" + "=" * 60)
        print("Frozen Baseline (Stage-0)")
        print("=" * 60)
        frozen_model = TaBERTForContrastive(
            model_path=args.base_model_path,
            use_lora=False,
            gradient_checkpointing=False,
        )
        frozen_model.to(device)
        frozen_results = evaluate_frozen_baseline(
            frozen_model, test_data, args.is_flipped, args.sample_row_num, args.max_context_len, args.use_amp,
        )
        all_results.update(frozen_results)
        del frozen_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    _tabert_dir = str(Path(__file__).resolve().parent)
    output_file = args.output_file or os.path.join(_tabert_dir, "eval_results.json")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 60)
    print("Summary:")
    for k, v in all_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
