import argparse
import csv
import os
import sys
import random
import shutil

def read_text_to_table_gt(gt_file, sep=","):
    """Lightweight GT reader: returns {text_id: [table_id, ...]}."""
    gt_map = {}
    with open(gt_file, "r") as f:
        csvf = csv.reader(f, delimiter=sep)
        for row in csvf:
            if len(row) < 2:
                continue
            key = row[0].strip()
            value = row[1].strip()
            values = gt_map.get(key, [])
            values.append(value)
            gt_map[key] = values
    return gt_map

def _read_abstract_only(filepath):
    """Read a PubMed target file and return only the first non-empty line (abstract).
    Strips MeSH keyword lines that follow the abstract."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:  # first non-empty line is the abstract
                    return line
    except Exception:
        pass
    return ""

def phase_split_and_copy(args):
    """
    Read the master GT file, split text IDs 70/15/15, and copy text files
    and table files into pharma_data/{train,val,test}/ directories.
    """
    data_root = args.data_dir

    # Check if all splits already exist
    splits_ready = True
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_root, split)
        gt_path = os.path.join(split_dir, "pharma-text-tables.gt")
        if not os.path.exists(gt_path):
            splits_ready = False
            break

    if splits_ready and not args.force_split:
        print(f"[Phase 0] Split data already exists in {data_root} -- skipping")
        print("          (use --force_split to re-create)")
        return

    print("[Phase 0] Splitting Pharma dataset into train/val/test...")
    if args.abstract_only:
        print("  [ABSTRACT-ONLY MODE] Keywords will be stripped from text files.")

    # Read master GT
    gt_map = read_text_to_table_gt(args.gt_file)
    all_text_ids = sorted(gt_map.keys())
    print(f"  Master GT: {len(all_text_ids)} unique text IDs")

    # Deterministic shuffle & split
    rng = random.Random(args.seed)
    shuffled = list(all_text_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    # test gets the remainder
    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train:n_train + n_val])
    test_ids = set(shuffled[n_train + n_val:])

    print(
        "  Split: %d train, %d val, %d test"
        % (len(train_ids), len(val_ids), len(test_ids))
    )

    # For each split: create dirs, copy texts, copy tables, write GT subset
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_dir = os.path.join(data_root, split_name)
        texts_dir = os.path.join(split_dir, "texts")
        tables_dir = os.path.join(split_dir, "tables")
        gt_path = os.path.join(split_dir, "pharma-text-tables.gt")

        # Clean and recreate
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(texts_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        # Copy text files
        copied_texts = 0
        for tid in split_ids:
            # Text files in pubmed-targets are directories named Target-XXXXX
            # containing files, OR they are files directly. Let's handle both.
            src = os.path.join(args.text_input_dir, tid)
            if os.path.isfile(src):
                if args.abstract_only:
                    # Keep only the first line (abstract), strip keyword lines
                    content = _read_abstract_only(src)
                    if content:
                        with open(os.path.join(texts_dir, tid), "w", encoding="utf-8") as f:
                            f.write(content)
                        copied_texts += 1
                else:
                    shutil.copy2(src, os.path.join(texts_dir, tid))
                    copied_texts += 1
            elif os.path.isdir(src):
                # Copy the directory contents as a single text file
                # (merge all files inside into one)
                content = ""
                for fname in sorted(os.listdir(src)):
                    fpath = os.path.join(src, fname)
                    if os.path.isfile(fpath):
                        try:
                            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                                content += f.read() + "\n"
                        except Exception:
                            pass
                if content:
                    if args.abstract_only:
                        # Keep only the first line of the merged content
                        content = content.split("\n")[0].strip()
                    if content:
                        with open(os.path.join(texts_dir, tid), "w", encoding="utf-8") as f:
                            f.write(content)
                        copied_texts += 1

        # Copy ALL table files (including .zip files as-is)
        copied_tables = 0
        for fname in os.listdir(args.table_input_dir):
            src = os.path.join(args.table_input_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tables_dir, fname))
                copied_tables += 1

        # Write GT subset
        gt_rows = 0
        with open(gt_path, "w", newline="") as f:
            writer = csv.writer(f)
            for tid in sorted(split_ids):
                if tid in gt_map:
                    for table_id in gt_map[tid]:
                        writer.writerow([tid, table_id])
                        gt_rows += 1

        print(
            "  %s: %d texts, %d tables, %d GT rows"
            % (split_name, copied_texts, copied_tables, gt_rows)
        )

    print("[Phase 0] Split complete.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legacy Splitting for CMDL Pharma dataset")
    parser.add_argument(
        "--gt_file", type=str, default="inputs/pubmed-drugbank-tables.gt",
        help="Master ground-truth file (text-to-table mapping)"
    )
    parser.add_argument(
        "--text_input_dir", type=str, default="inputs/pubmed-targets",
        help="Directory containing PubMed target text files"
    )
    parser.add_argument(
        "--table_input_dir", type=str, default="inputs/drugbank-tables",
        help="Directory containing DrugBank table CSVs"
    )
    parser.add_argument("--data_dir", type=str, default="pharma_data", help="Root directory for split data")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_split", action="store_true", help="Force re-splitting")
    parser.add_argument("--abstract_only", action="store_true", help="Strip MeSH keywords")

    args = parser.parse_args()
    if args.abstract_only:
        if args.data_dir == "pharma_data":
            args.data_dir = "pharma_data_abstract_only"
            
    phase_split_and_copy(args)
