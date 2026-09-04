"""
Preview what TaBERT actually sees during training — using the real data_loader.py
pipeline so this also serves as a validation tool.

Loads examples via data_loader functions, builds TaBERT Table objects, and
prints the actual context tokens, table headers, and table data that get
passed to model.encode().

Usage:
    python preview_dataset.py [--data_dir pharma_flipped_structured] [--split train]
                              [--index 0] [--max_rows 3] [--is_flipped True]
"""

import os
import sys
import json
import argparse
import textwrap
from pathlib import Path

# Ensure TaBERT package is importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import (
    load_row_level_dataset,
    _extract_sentences,
    _extract_table_structured,
    structured_to_tabert_table,
    example_to_tabert_table,
    generate_triplets_for_example,
)
from transformers import BertTokenizer


def wrap(text, width=100, indent="    "):
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def print_tabert_table(table, indent="    ", max_col_width=60):
    """Pretty-print an actual TaBERT Table object."""
    if table is None:
        print(f"{indent}(None)")
        return

    header_names = [col.name for col in table.header]
    col_widths = [min(len(h), max_col_width) for h in header_names]
    for row in table.data:
        for i, cell in enumerate(row):
            col_widths[i] = min(max(col_widths[i], len(str(cell))), max_col_width)

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header_names))
    print(f"{indent}{header_line}")
    print(f"{indent}{'-+-'.join('-' * w for w in col_widths)}")

    for row in table.data:
        cells = []
        for i, cell in enumerate(row):
            s = str(cell)
            display = s[:max_col_width - 3] + "..." if len(s) > max_col_width else s
            cells.append(display.ljust(col_widths[i]))
        print(f"{indent}{' | '.join(cells)}")


def preview_example(example, tokenizer, is_flipped, sample_row_num=3):
    anchor_id = example.get("anchor_id")

    print("=" * 100)
    print(f"  EXAMPLE  anchor_id={anchor_id}  (is_flipped={is_flipped})")
    print("=" * 100)

    # Generate one triplet
    triplets = generate_triplets_for_example(example, strategy="primary_only", max_triplets=1)
    if not triplets:
        print("  (no triplets could be generated)\n")
        return

    triplet = triplets[0]

    if is_flipped:
        # ── Context: anchor doc sentences ──
        doc_sents = example.get("anchor_sentences", [])
        print(f"\n  CONTEXT  (anchor doc sentences → TaBERT NL input)  [{len(doc_sents)} sentences]")
        print("  " + "-" * 96)
        for i, s in enumerate(doc_sents):
            print(f"  [{i}]")
            print(wrap(s))

        # Tokenize first sentence as TaBERT would
        ctx_tokens = tokenizer.tokenize(doc_sents[0])[:128] if doc_sents else []
        print(f"\n  Tokenized context (first sentence, max 128 tokens): [{len(ctx_tokens)} tokens]")
        print(f"    {' '.join(ctx_tokens[:30])}{'...' if len(ctx_tokens) > 30 else ''}")

        # ── Positive table: built from structured headers + rows ──
        pos_headers = triplet['positive_headers']
        pos_rows = triplet['positive_rows']
        pos_table = structured_to_tabert_table(
            table_id=str(triplet['positive_id']),
            headers=pos_headers,
            row_content_lists=pos_rows,
            tokenizer=tokenizer,
            max_rows=sample_row_num,
        )
        print(f"\n  POSITIVE TABLE  id={triplet['positive_id']}  "
              f"[{len(pos_rows)} rows total → {len(pos_table.data) if pos_table else 0} after truncation]")
        print("  " + "-" * 96)
        print_tabert_table(pos_table)
        if pos_rows and len(pos_rows) > sample_row_num > 0:
            print(f"    ... ({len(pos_rows) - sample_row_num} more rows truncated by sample_row_num={sample_row_num})")

        # ── Negative table ──
        neg_headers = triplet['negative_headers']
        neg_rows = triplet['negative_rows']
        neg_table = structured_to_tabert_table(
            table_id=str(triplet['negative_id']),
            headers=neg_headers,
            row_content_lists=neg_rows,
            tokenizer=tokenizer,
            max_rows=sample_row_num,
        )
        print(f"\n  NEGATIVE TABLE  id={triplet['negative_id']}  "
              f"[{len(neg_rows)} rows total → {len(neg_table.data) if neg_table else 0} after truncation]")
        print("  " + "-" * 96)
        print_tabert_table(neg_table)
        if neg_rows and len(neg_rows) > sample_row_num > 0:
            print(f"    ... ({len(neg_rows) - sample_row_num} more rows truncated by sample_row_num={sample_row_num})")

    else:
        # ── Table: anchor rows via example_to_tabert_table ──
        table = example_to_tabert_table(example, tokenizer=tokenizer)
        if table and sample_row_num > 0 and len(table.data) > sample_row_num:
            table = table.with_rows(table.data[:sample_row_num])
            table.tokenize(tokenizer)
        print(f"\n  ANCHOR TABLE  (anchor rows → TaBERT Table)  "
              f"[{len(table.data) if table else 0} rows, "
              f"{len(table.header) if table else 0} columns]")
        print("  " + "-" * 96)
        print_tabert_table(table)

        # ── Positive table ──
        pos_headers = triplet['positive_headers']
        pos_rows = triplet['positive_rows']
        print(f"\n  POSITIVE TABLE  id={triplet['positive_id']}  "
              f"[headers={pos_headers}, {len(pos_rows)} rows]")
        print("  " + "-" * 96)
        for i, row in enumerate(pos_rows[:3]):
            print(f"  [{i}] {row}")
        if len(pos_rows) > 3:
            print(f"    ... ({len(pos_rows) - 3} more)")

        # ── Negative table ──
        neg_headers = triplet['negative_headers']
        neg_rows = triplet['negative_rows']
        print(f"\n  NEGATIVE TABLE  id={triplet['negative_id']}  "
              f"[headers={neg_headers}, {len(neg_rows)} rows]")
        print("  " + "-" * 96)
        for i, row in enumerate(neg_rows[:3]):
            print(f"  [{i}] {row}")
        if len(neg_rows) > 3:
            print(f"    ... ({len(neg_rows) - 3} more)")

    # ── Summary ──
    print(f"\n  WHAT model.score() RECEIVES")
    print("  " + "-" * 96)
    if is_flipped:
        print(f"  contexts  = [tokenizer.tokenize(anchor_sentence)]  → {len(ctx_tokens)} tokens")
        print(f"  pos_table = Table(header={[c.name for c in pos_table.header]}, "
              f"rows={len(pos_table.data)})")
        print(f"  neg_table = Table(header={[c.name for c in neg_table.header]}, "
              f"rows={len(neg_table.data)})")
    else:
        print(f"  table      = Table(header={[c.name for c in table.header]}, "
              f"rows={len(table.data)})")
        print(f"  pos_table  = structured({pos_headers}, {len(pos_rows)} rows)")
        print(f"  neg_table  = structured({neg_headers}, {len(neg_rows)} rows)")
    print(f"  → model.score(context, pos) should be > model.score(context, neg)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Preview TaBERT training data (uses real data_loader.py)")
    parser.add_argument("--data_dir", default="pharma_flipped_structured")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=None,
                        help="Show a specific example index (default: show first 3)")
    parser.add_argument("--max_rows", type=int, default=3,
                        help="Max table rows TaBERT sees (matches sample_row_num)")
    parser.add_argument("--count", type=int, default=3,
                        help="Number of examples to show (ignored if --index is set)")
    parser.add_argument("--is_flipped", type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=True,
                        help="True for flipped format (anchors=docs, default), False for standard")
    parser.add_argument("--model_path", default="bert-large-uncased",
                        help="BERT model name or path (for tokenizer)")
    args = parser.parse_args()

    data_path = SCRIPT_DIR / args.data_dir / f"{args.split}_row_level.json"
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    # Load tokenizer directly (same as TaBERT uses internally)
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    print(f"Loading tokenizer ({args.model_path})...")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    data = load_row_level_dataset(str(data_path))
    print(f"Loaded {len(data)} examples from {data_path.name}")
    print(f"Mode: is_flipped={args.is_flipped}\n")

    if args.index is not None:
        if args.index >= len(data):
            print(f"Index {args.index} out of range (max {len(data)-1})")
            return
        preview_example(data[args.index], tokenizer, args.is_flipped, args.max_rows)
    else:
        for i in range(min(args.count, len(data))):
            preview_example(data[i], tokenizer, args.is_flipped, args.max_rows)


if __name__ == "__main__":
    main()
