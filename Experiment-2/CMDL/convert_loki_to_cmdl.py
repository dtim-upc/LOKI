"""
convert_loki_to_cmdl.py - Convert LOKI's MIMIC JSON format into CMDL's native format.

Produces:
  - tables/  : One CSV per table-example (columns = table headers, rows = table rows)
  - texts/   : One TXT per text-document  (all sentences concatenated)
  - <datalake>-text-tables.gt : Ground truth mapping  text_id -> table_id

Usage:
  python convert_loki_to_cmdl.py \
      --input_file mimic_data/train_row_level.json \
      --output_dir cmdl_mimic_data/train \
      --max_examples 0 \
      --seed 42
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path


def load_loki_json(path):
    """Load a LOKI JSON dataset file."""
    print("Loading LOKI JSON from %s ..." % path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("  Loaded %d examples" % len(data))
    return data


def subsample_deterministic(examples, max_examples, seed):
    """
    Deterministic subsampling matching LOKI's run_cross_attention.py approach:
    sort by stable key (example_id), then sample with an isolated RNG.
    """
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples

    def get_stable_key(ex):
        return ex.get("example_id", "") or str(ex.get("anchor_id", ""))

    rng = random.Random(seed)
    sorted_examples = sorted(examples, key=get_stable_key)
    sampled = rng.sample(sorted_examples, max_examples)
    print("  -> Subsampled %d examples (seed=%d)" % (len(sampled), seed))
    return sampled


def extract_sentences_text(sentences_data):
    """
    Extract all sentence texts from LOKI's sentences format (dict keyed by index)
    and concatenate them into a single document string.
    """
    texts = []
    if isinstance(sentences_data, dict):
        try:
            sorted_keys = sorted(sentences_data.keys(), key=lambda k: int(k))
        except ValueError:
            sorted_keys = sorted(sentences_data.keys())
        for k in sorted_keys:
            item = sentences_data[k]
            if isinstance(item, dict):
                t = item.get("text", "")
            elif isinstance(item, str):
                t = item
            else:
                t = ""
            if t:
                texts.append(t)
    elif isinstance(sentences_data, list):
        for item in sentences_data:
            if isinstance(item, dict):
                t = item.get("text", "")
            elif isinstance(item, str):
                t = item
            else:
                t = ""
            if t:
                texts.append(t)
    return " ".join(texts)


def convert_example(example, tables_dir, texts_dir):
    """
    Convert a single LOKI example into CMDL-format files.

    Returns:
        list of (text_id, table_id) ground-truth pairs produced, or empty list on skip.
    """
    example_id = example.get("example_id")
    if example_id is None:
        example_id = example.get("anchor_id", "")
    example_id = str(example_id)

    if not example_id:
        return []

    # ---- Export table as CSV ----
    tables = example.get("tables", {})
    if not tables and "anchor_rows" in example:
        tables = {
            "main": {
                "headers": example.get("anchor_headers", []),
                "rows": example.get("anchor_rows", [])
            }
        }

    if not tables:
        return []

    # Each example may have multiple table types (e.g. diagnosis, medication).
    # CMDL expects one CSV per table, so we create one CSV per table-type within
    # the example. The table_id includes the example_id to keep things unique.
    table_ids = []
    for table_type, table_data in tables.items():
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        if not headers or not rows:
            continue

        # Table ID = example_id (already includes table type)
        table_id = example_id
        csv_path = tables_dir / ("%s.csv" % table_id)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                if isinstance(row, dict):
                    content = row.get("content", [])
                    if not content:
                        content = [row.get("formatted", "")]
                elif isinstance(row, str):
                    content = [row]
                else:
                    content = []
                    
                # content may have leading empty strings (padding); strip to match header count
                if len(content) > len(headers):
                    content = content[len(content) - len(headers):]
                elif len(content) < len(headers):
                    content = content + [""] * (len(headers) - len(content))
                writer.writerow(content)

        table_ids.append(table_id)

    if not table_ids:
        return []

    # ---- Export text documents ----
    gt_pairs = []

    # Primary positive
    primary = example.get("primary_positive", {})
    if primary and primary.get("id") is not None:
        text_id = str(primary["id"])
        sentences = primary.get("sentences", {})
        doc_text = extract_sentences_text(sentences)
        if doc_text:
            txt_path = texts_dir / ("%s.txt" % text_id)
            if not txt_path.exists():  # avoid overwriting (same text may appear in multiple examples)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(doc_text)
            for tid in table_ids:
                gt_pairs.append((text_id, tid))

    # Additional positives
    for add_pos in example.get("additional_positives", []):
        if add_pos and add_pos.get("id") is not None:
            text_id = str(add_pos["id"])
            sentences = add_pos.get("sentences", {})
            doc_text = extract_sentences_text(sentences)
            if doc_text:
                txt_path = texts_dir / ("%s.txt" % text_id)
                if not txt_path.exists():
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(doc_text)
                for tid in table_ids:
                    gt_pairs.append((text_id, tid))

    return gt_pairs


def convert_dataset(examples, output_dir, datalake="mimic"):
    """
    Convert a full list of LOKI examples into CMDL's native directory structure.
    """
    out = Path(output_dir)
    tables_dir = out / "tables"
    texts_dir = out / "texts"
    tables_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    all_gt_pairs = []
    skipped = 0

    for example in examples:
        pairs = convert_example(example, tables_dir, texts_dir)
        if pairs:
            all_gt_pairs.extend(pairs)
        else:
            skipped += 1

    # Write ground truth file
    gt_path = out / ("%s-text-tables.gt" % datalake)
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for text_id, table_id in all_gt_pairs:
            writer.writerow([text_id, table_id])

    # Write metadata
    meta = {
        "source_format": "loki_mimic_v2",
        "total_examples": len(examples),
        "skipped_examples": skipped,
        "total_gt_pairs": len(all_gt_pairs),
        "unique_tables": len(list(tables_dir.glob("*.csv"))),
        "unique_texts": len(list(texts_dir.glob("*.txt"))),
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("")
    print("[OK] Conversion complete:")
    print("   Tables:  %d CSV files in %s" % (meta["unique_tables"], tables_dir))
    print("   Texts:   %d TXT files in %s" % (meta["unique_texts"], texts_dir))
    print("   GT:      %d pairs in %s" % (meta["total_gt_pairs"], gt_path))
    print("   Skipped: %d examples (missing table/text data)" % skipped)

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Convert LOKI MIMIC JSON to CMDL native format (CSV tables + text files + ground truth)"
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to LOKI JSON file (e.g. train_row_level.json)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for CMDL-format data"
    )
    parser.add_argument(
        "--max_examples", type=int, default=0,
        help="Maximum number of examples to convert (0 = all). Uses deterministic subsampling."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic subsampling (matches LOKI's approach)"
    )
    parser.add_argument(
        "--datalake", type=str, default="mimic",
        help="Datalake name prefix for output files (default: mimic)"
    )

    args = parser.parse_args()

    # Load
    examples = load_loki_json(args.input_file)

    # Subsample
    examples = subsample_deterministic(examples, args.max_examples, args.seed)

    # Convert
    convert_dataset(examples, args.output_dir, args.datalake)


if __name__ == "__main__":
    main()
