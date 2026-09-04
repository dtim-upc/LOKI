"""
run_mimic_cmdl_original.py - CMDL-style Doc-to-Table training on MIMIC.

This runner keeps CMDL's original column-text joint training mechanics
from `trainer/column-text-joint-training.ipynb`:
  - independent text and column mini-batches
  - label sub-matrix per step
  - TripletLoss over text/column encoder outputs

Adaptation for MIMIC:
  - uses gold MIMIC ground-truth pairs (mimic-text-tables.gt) instead of
    Snorkel weak labels
  - keeps query modality as documents (MIMIC notes) for Doc-to-Table eval
"""

import argparse
import csv
import json
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from convert_loki_to_cmdl import load_loki_json, subsample_deterministic, convert_dataset
from profiler.text_profiler import text_profiler
from profiler.table_profiler import table_profiler
from trainer.text_featurizer import text_featurizer
from trainer.table_featurizer import table_featurizer
from indexer.wem import WEM
from indexer.trained_embeddings import TrainedEmbeddings, TrainedEmbeddingsIndexer


class DataCombiner(Dataset):
    """Minimal dataset wrapper used by CMDL notebook training."""

    def __init__(self, ids, features):
        self.ids = ids
        self.features = features

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return (i, self.ids[i], self.features[i])


def ensure_minibatch_size_at_least_two(idx, batch, loader_iter, loader):
    """
    BatchNorm layers require batch size > 1 during training.
    When full-coverage mode reaches the final remainder batch of size 1,
    append one extra sample from the next batch (or duplicate if necessary).
    """
    if batch.shape[0] > 1:
        return idx, batch, loader_iter

    try:
        next_idx, _, next_batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        next_idx, _, next_batch = next(loader_iter)

    if next_batch.shape[0] >= 1:
        idx = torch.cat([idx, next_idx[:1]], dim=0)
        batch = torch.cat([batch, next_batch[:1]], dim=0)
    else:
        # Defensive fallback: duplicate the current sample.
        idx = torch.cat([idx, idx], dim=0)
        batch = torch.cat([batch, batch], dim=0)
    return idx, batch, loader_iter


def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.sqrt(torch.pow(x - y, 2).sum(2) + 1e-12)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, neg_weight=1.0, normalize_feature=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.neg_weight = neg_weight
        self.normalize_feature = normalize_feature

    def forward(self, emb1, emb2, label_mat, mask1):
        if self.normalize_feature:
            emb1 = emb1.sigmoid()
            emb2 = emb2.sigmoid()
        mat_dist = euclidean_dist(emb1, emb2)
        n_rows = mat_dist.size(0)
        eligible = torch.count_nonzero(mask1)
        if eligible == 0:
            return (torch.tensor(0.0, device=emb1.device), torch.tensor(1.0, device=emb1.device))

        positives = mask1 * mat_dist * label_mat
        dist_ap = torch.sum(positives, dim=1)
        negatives, _ = torch.sort(
            mat_dist + 100000.0 * (label_mat + 1 - mask1), dim=1, descending=False
        )
        dist_an = negatives[:, 0]

        loss = torch.sum(
            dist_ap + self.neg_weight * torch.max(
                torch.zeros_like(dist_an), self.margin - dist_an
            )
        ) / eligible

        pos_counts = (mask1 * label_mat).sum(dim=1).clamp(min=1)
        dist_ap_avg = dist_ap / pos_counts
        prec = (dist_an.data > dist_ap_avg.data).sum() * 1.0 / n_rows
        return (loss, prec)


class EncoderNet(nn.Module):
    def __init__(self, ip=1000, op=100, hidden1=200, hidden2=200):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(ip, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Linear(hidden2, op),
        )

    def forward(self, x):
        return self.net(x)


def write_csv_list(fp, rows):
    with open(fp, "w", newline="") as f:
        csvf = csv.writer(f)
        for r in rows:
            csvf.writerow([r])


def read_gt_ids(file_path):
    tid = set()
    cid = set()
    with open(file_path, "r") as f:
        csvf = csv.reader(f)
        for r in csvf:
            if len(r) < 2:
                continue
            tid.add(r[0])
            cid.add(r[1])
    return sorted(list(tid)), sorted(list(cid))


def build_label_matrix(text_ids, col_ids, gt_map):
    col_to_table = {}
    for j, cid in enumerate(col_ids):
        table_name = cid.split(",")[0] if "," in cid else cid
        col_to_table[j] = table_name

    text_to_tables = {}
    for text_id, table_ids in gt_map.items():
        table_set = set()
        for t in table_ids:
            table_set.add(t)
            table_set.add(t + ".csv")
        text_to_tables[text_id] = table_set

    n_text = len(text_ids)
    n_col = len(col_ids)
    label_mat = torch.zeros(n_text, n_col)
    mask1 = torch.zeros(n_text, 1)

    for i, tid in enumerate(text_ids):
        tid_clean = tid.replace(".txt", "") if tid.endswith(".txt") else tid
        target_tables = text_to_tables.get(tid_clean, set())
        if not target_tables:
            continue
        has_match = False
        for j in range(n_col):
            table_name = col_to_table[j]
            if table_name in target_tables:
                label_mat[i][j] = 1.0
                has_match = True
        if has_match:
            mask1[i] = 1.0

    active_texts = int(mask1.sum().item())
    active_pairs = int(label_mat.sum().item())
    print(
        "  Label matrix: %d texts x %d columns, %d active texts, %d positive pairs"
        % (n_text, n_col, active_texts, active_pairs)
    )
    return label_mat, mask1


def phase_convert(args, split_name, input_file, max_examples):
    output_dir = os.path.join(args.output_dir, "data", split_name)
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(meta_path) and not args.force_conversion:
        with open(meta_path) as f:
            meta = json.load(f)
        print(
            "[Phase 1] Found existing %s data (%d tables, %d texts) -- skipping conversion"
            % (split_name, meta["unique_tables"], meta["unique_texts"])
        )
        print("          (use --force_conversion to re-convert)")
        return output_dir

    print("[Phase 1] Converting %s: %s" % (split_name, input_file))
    examples = load_loki_json(input_file)
    examples = subsample_deterministic(examples, max_examples, args.seed)
    convert_dataset(examples, output_dir, "mimic")
    return output_dir


def phase_build_features(args, data_dir, split_name):
    table_path = os.path.join(data_dir, "tables")
    text_path = os.path.join(data_dir, "texts")
    gt_path = os.path.join(data_dir, "mimic-text-tables.gt")
    feature_dir = os.path.join(args.output_dir, "features", split_name)
    os.makedirs(feature_dir, exist_ok=True)

    text_feat_path = os.path.join(feature_dir, "mimic-textfeatures.pt")
    col_feat_path = os.path.join(feature_dir, "mimic-columnfeatures.pt")
    if args.skip_features and os.path.exists(text_feat_path) and os.path.exists(col_feat_path):
        print("[Phase 2] Skipping feature building for %s (cached)" % split_name)
        with open(os.path.join(feature_dir, "mimic-textids.list")) as f:
            text_ids = [r.strip() for r in f if r.strip()]
        with open(os.path.join(feature_dir, "mimic-colids.list")) as f:
            col_ids = [r.strip() for r in f if r.strip()]
        text_f = torch.load(text_feat_path, weights_only=True)
        table_f = torch.load(col_feat_path, weights_only=True)
        return text_ids, col_ids, text_f, table_f

    print("[Phase 2] Building features for %s ..." % split_name)
    text_ids_gt, table_ids_gt = read_gt_ids(gt_path)
    print("  GT: %d texts, %d tables" % (len(text_ids_gt), len(table_ids_gt)))

    text_p = text_profiler("en_core_web_sm")
    table_p = table_profiler(None, None)
    wem = WEM(args.wem_model_path, args.wem_dim)

    text_ids, text_f = text_featurizer(wem, None).featurize(text_p, text_path)
    col_ids, table_f = table_featurizer(wem, None).featurize(table_p, table_path, ",")
    text_ids = list(text_ids)
    col_ids = list(col_ids)

    write_csv_list(os.path.join(feature_dir, "mimic-textids.list"), text_ids)
    write_csv_list(os.path.join(feature_dir, "mimic-colids.list"), col_ids)
    torch.save(text_f, text_feat_path)
    torch.save(table_f, col_feat_path)
    print(
        "  Saved: %d text features (%s), %d col features (%s)"
        % (len(text_ids), text_f.shape, len(col_ids), table_f.shape)
    )
    return text_ids, col_ids, text_f, table_f


def read_text_to_table_gt(gt_file, sep=","):
    """Lightweight GT reader to avoid heavy compare_gt imports."""
    gt_map = {}
    with open(gt_file, "r") as f:
        csvf = csv.reader(f, delimiter=sep)
        for row in csvf:
            if len(row) < 2:
                continue
            key = row[0]
            value = row[1]
            values = gt_map.get(key, [])
            values.append(value)
            gt_map[key] = values
    return gt_map


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_matches(gt_map, predictions):
    tp = 0
    fp = 0
    fn = 0
    for idx in predictions:
        gt_matches = set(gt_map.get(idx, []))
        if len(gt_matches) == 0:
            continue
        pred_matches = set(predictions.get(idx, []))
        true_matches = len(gt_matches.intersection(pred_matches))
        false_matches = len(pred_matches) - true_matches
        non_matches = len(gt_matches) - true_matches
        tp += true_matches
        fp += false_matches
        fn += non_matches
    fp = 1 if (tp + fp) == 0 else fp
    prec = 1.0 * tp / (tp + fp)
    rec = 1.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0 if tp == 0 else 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1


def evaluate(gt_map, ip_fn, search_fn, topn):
    predictions = {}
    for idx in gt_map:
        input_vec = ip_fn(idx)
        results = search_fn(input_vec, topn)
        predictions[idx] = [tid for (tid, score) in results]
    return eval_matches(gt_map, predictions)


def phase_evaluate(args, text_enet, col_enet, text_ids, col_ids, text_f, col_f, gt_path, split_name):
    device = next(text_enet.parameters()).device
    feature_dir = os.path.join(args.output_dir, "features", split_name)
    os.makedirs(feature_dir, exist_ok=True)

    text_enet.eval()
    col_enet.eval()
    with torch.no_grad():
        text_emb = text_enet(text_f.to(device)).cpu().numpy()
        col_emb = col_enet(col_f.to(device)).cpu().numpy()

    text_emb_path = os.path.join(feature_dir, "mimic-0-trainedtext.npy")
    col_emb_path = os.path.join(feature_dir, "mimic-0-trainedcolumns.npy")
    np.save(text_emb_path, text_emb)
    np.save(col_emb_path, col_emb)

    text_emb_obj = TrainedEmbeddings(text_ids, text_emb_path)
    col_emb_obj = TrainedEmbeddings(col_ids, col_emb_path)
    col_emb_ind = TrainedEmbeddingsIndexer("mimic-trained", col_emb_obj, "table")
    col_emb_ind.create_index()
    for cid in col_ids:
        col_emb_ind.index_doc(cid)
    col_emb_ind.commit_index()

    raw_gt_map = read_text_to_table_gt(gt_path)
    text_id_set = set(text_ids)
    gt_map = {}
    skipped = 0
    for raw_text_id, raw_table_ids in raw_gt_map.items():
        if raw_text_id in text_id_set:
            mapped_text_id = raw_text_id
        elif raw_text_id + ".txt" in text_id_set:
            mapped_text_id = raw_text_id + ".txt"
        else:
            skipped += 1
            continue
        mapped_tables = [t + ".csv" for t in raw_table_ids]
        gt_map[mapped_text_id] = mapped_tables

    print("\n[Phase 4] Evaluation on %s:" % split_name)
    print("  GT remapped: %d text queries (%d skipped -- not in profiled data)" % (len(gt_map), skipped))
    if len(gt_map) == 0:
        print("  [WARNING] No matching text IDs found! Skipping evaluation.")
        return []

    k_values = [1, 3, 5, 10]
    results = []
    print("  %-5s  %-8s  %-8s  %-8s" % ("K", "P@K", "R@K", "F1@K"))
    print("  " + "-" * 35)
    for k in k_values:
        p, r, f1 = evaluate(gt_map, text_emb_obj, col_emb_ind.search, k)
        results.append({"k": k, "precision": p, "recall": r, "f1": f1})
        print("  %-5d  %-8.4f  %-8.4f  %-8.4f" % (k, p, r, f1))

    results_path = os.path.join(args.output_dir, "%s_results.json" % split_name)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to %s" % results_path)
    return results


def phase_train_cmdl_style(
    args,
    train_text_ids,
    train_col_ids,
    train_text_f,
    train_col_f,
    train_gt_path,
):
    """
    Train with CMDL notebook mechanics:
      - text and column data loaders are iterated independently
      - each step slices label matrix to current text/column batch
      - mask keeps only rows with both positive and negative examples
    """
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Phase 3] CMDL-style joint training on device: %s" % device)

    # Build full label matrix using MIMIC GT -> text-to-table mapping
    train_gt_map = read_text_to_table_gt(train_gt_path)
    label_mat_full, _ = build_label_matrix(train_text_ids, train_col_ids, train_gt_map)

    # Optional fractioning (mirrors notebook behavior)
    n_text = max(1, int(len(train_text_ids) * args.train_text_fraction))
    n_col = max(1, int(len(train_col_ids) * args.train_col_fraction))
    train_text_ids = train_text_ids[:n_text]
    train_col_ids = train_col_ids[:n_col]
    train_text_f = train_text_f[:n_text]
    train_col_f = train_col_f[:n_col]
    label_mat = label_mat_full[:n_text, :n_col]

    print("  Training subset: %d texts, %d columns" % (n_text, n_col))

    # Optional paper-style mini-batch sizing: m and n as a fraction of DEs.
    # This can create very large batches on big datasets, so keep it opt-in.
    text_batch_size = args.text_batch_size
    col_batch_size = args.col_batch_size
    if args.batch_fraction > 0.0:
        text_batch_size = max(1, int(n_text * args.batch_fraction))
        col_batch_size = max(1, int(n_col * args.batch_fraction))
        print(
            "  Using batch_fraction=%.4f -> text_bsz=%d col_bsz=%d"
            % (args.batch_fraction, text_batch_size, col_batch_size)
        )

    text_dim = train_text_f.shape[1]
    col_dim = train_col_f.shape[1]
    print("  Text dim: %d, Col dim: %d, Output dim: %d" % (text_dim, col_dim, args.output_size))

    text_enet = EncoderNet(
        ip=text_dim, op=args.output_size, hidden1=args.hidden_size, hidden2=args.hidden_size
    ).to(device)
    col_enet = EncoderNet(
        ip=col_dim, op=args.output_size, hidden1=args.hidden_size, hidden2=args.hidden_size
    ).to(device)

    criterion = TripletLoss(margin=args.margin, neg_weight=args.neg_weight)
    optimizer = torch.optim.Adam(
        list(text_enet.parameters()) + list(col_enet.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )

    train_text_ds = DataCombiner(train_text_ids, train_text_f)
    train_col_ds = DataCombiner(train_col_ids, train_col_f)
    train_text_loader = DataLoader(
        train_text_ds, batch_size=text_batch_size, shuffle=True, drop_last=False
    )
    train_col_loader = DataLoader(
        train_col_ds, batch_size=col_batch_size, shuffle=True, drop_last=False
    )

    # Notebook used a fixed value (26). Keep it configurable by default.
    # If ensure_all_examples_per_epoch is enabled, run enough steps so both
    # loaders are fully consumed at least once (the shorter one may repeat).
    if args.ensure_all_examples_per_epoch:
        steps_per_epoch = max(len(train_text_loader), len(train_col_loader))
    elif args.steps_per_epoch <= 0:
        steps_per_epoch = max(len(train_text_loader), len(train_col_loader))
    else:
        steps_per_epoch = args.steps_per_epoch

    print(
        "  epochs=%d text_bsz=%d col_bsz=%d steps_per_epoch=%s"
        % (args.epochs, text_batch_size, col_batch_size, str(steps_per_epoch))
    )

    best_loss = float("inf")
    best_epoch = 0
    label_mat = label_mat.to(device)

    for epoch in range(args.epochs):
        text_enet.train()
        col_enet.train()

        epoch_loss = 0.0
        epoch_prec = 0.0
        step_count = 0

        text_it = iter(train_text_loader)
        col_it = iter(train_col_loader)

        while step_count < steps_per_epoch:
            try:
                t_idx, _, t_batch = next(text_it)
            except StopIteration:
                text_it = iter(train_text_loader)
                t_idx, _, t_batch = next(text_it)

            try:
                c_idx, _, c_batch = next(col_it)
            except StopIteration:
                col_it = iter(train_col_loader)
                c_idx, _, c_batch = next(col_it)

            # Full-coverage mode may reach remainder batches of size 1.
            # Keep BatchNorm-compatible batches without changing model design.
            t_idx, t_batch, text_it = ensure_minibatch_size_at_least_two(
                t_idx, t_batch, text_it, train_text_loader
            )
            c_idx, c_batch, col_it = ensure_minibatch_size_at_least_two(
                c_idx, c_batch, col_it, train_col_loader
            )

            t_idx = t_idx.to(device)
            c_idx = c_idx.to(device)
            t_batch = t_batch.to(device)
            c_batch = c_batch.to(device)

            # Build sub-label matrix and mask for this text-column mini-batch
            sub_label = label_mat[t_idx][:, c_idx]
            col_bsz = sub_label.shape[1]
            mask = torch.count_nonzero(sub_label, dim=1)
            mask = torch.where((mask > 0) & (mask < col_bsz), 1, 0).unsqueeze(-1).float()

            text_emb = text_enet(t_batch)
            col_emb = col_enet(c_batch)
            loss, prec = criterion(text_emb, col_emb, sub_label, mask)

            if loss.item() > 0.0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_prec += prec.item()
            step_count += 1

        avg_loss = epoch_loss / max(step_count, 1)
        avg_prec = epoch_prec / max(step_count, 1)
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print("  Epoch %d/%d  loss=%.4f  prec=%.4f" % (epoch + 1, args.epochs, avg_loss, avg_prec))

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(text_enet.state_dict(), os.path.join(model_dir, "text_enet_best.pt"))
            torch.save(col_enet.state_dict(), os.path.join(model_dir, "col_enet_best.pt"))

    # Reload best checkpoint for downstream eval
    text_enet.load_state_dict(torch.load(os.path.join(model_dir, "text_enet_best.pt"), weights_only=True))
    col_enet.load_state_dict(torch.load(os.path.join(model_dir, "col_enet_best.pt"), weights_only=True))
    print("  Best checkpoint from epoch %d (train_loss=%.4f)" % (best_epoch, best_loss))
    return text_enet, col_enet


def load_saved_model(args, text_dim, col_dim):
    model_dir = os.path.join(args.output_dir, "models")
    text_enet = EncoderNet(
        ip=text_dim, op=args.output_size, hidden1=args.hidden_size, hidden2=args.hidden_size
    )
    col_enet = EncoderNet(
        ip=col_dim, op=args.output_size, hidden1=args.hidden_size, hidden2=args.hidden_size
    )
    text_path = os.path.join(model_dir, "text_enet_best.pt")
    col_path = os.path.join(model_dir, "col_enet_best.pt")
    if not os.path.exists(text_path) or not os.path.exists(col_path):
        print("[ERROR] No saved model found at %s" % model_dir)
        print("        Run training first (without --eval_only or --test_only)")
        sys.exit(1)
    text_enet.load_state_dict(torch.load(text_path, weights_only=True))
    col_enet.load_state_dict(torch.load(col_path, weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_enet.to(device)
    col_enet.to(device)
    print("  Loaded saved model from %s" % model_dir)
    return text_enet, col_enet


def main():
    parser = argparse.ArgumentParser(
        description="CMDL-style Doc-to-Table training/evaluation on MIMIC"
    )

    # Data paths
    parser.add_argument("--loki_train_file", type=str, default="mimic_data/train_row_level.json")
    parser.add_argument("--loki_eval_file", type=str, default="mimic_data/val_row_level.json")
    parser.add_argument("--loki_test_file", type=str, default="mimic_data/test_row_level.json")

    # Subsampling
    parser.add_argument("--max_train_examples", type=int, default=0)
    parser.add_argument("--max_eval_examples", type=int, default=0)
    parser.add_argument("--max_test_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # WEM model and features
    parser.add_argument(
        "--wem_model_path", type=str, default="resources/fasttext/cc/cc.en.300.bin"
    )
    parser.add_argument("--wem_dim", type=int, default=300)

    # CMDL-style training parameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--text_batch_size", type=int, default=32)
    parser.add_argument("--col_batch_size", type=int, default=32)
    parser.add_argument(
        "--batch_fraction",
        type=float,
        default=0.08,
        help="If >0, set text/col batch sizes as this fraction of train DEs (paper-style m,n).",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=26,
        help="CMDL notebook default is 26. Set <=0 to auto-derive from loaders.",
    )
    parser.add_argument(
        "--ensure_all_examples_per_epoch",
        action="store_true",
        help="Run enough steps so each text/column example is seen at least once per epoch.",
    )
    parser.add_argument(
        "--disable_ensure_all_examples_per_epoch",
        dest="ensure_all_examples_per_epoch",
        action="store_false",
        help="Disable full per-epoch example coverage and use steps_per_epoch behavior.",
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--neg_weight", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--output_size", type=int, default=100)
    parser.add_argument("--train_text_fraction", type=float, default=1.0)
    parser.add_argument("--train_col_fraction", type=float, default=1.0)

    # Output and control flags
    parser.add_argument("--output_dir", type=str, default="output_cmdl_mimic_original")
    parser.add_argument("--force_conversion", action="store_true")
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")

    parser.set_defaults(ensure_all_examples_per_epoch=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    print("=" * 60)
    print("CMDL-Style Pipeline on MIMIC (Doc-to-Table)")
    if args.test_only:
        print("  Mode: TEST ONLY (loading saved model)")
    elif args.eval_only:
        print("  Mode: EVAL ONLY (loading saved model, val + test)")
    else:
        print("  Mode: FULL PIPELINE (train + val + test)")
    print("=" * 60)

    # TEST-ONLY mode
    if args.test_only:
        if not os.path.exists(args.loki_test_file):
            print("[ERROR] Test file not found: %s" % args.loki_test_file)
            sys.exit(1)

        test_data_dir = phase_convert(args, "test", args.loki_test_file, args.max_test_examples)
        test_text_ids, test_col_ids, test_text_f, test_col_f = phase_build_features(args, test_data_dir, "test")
        test_gt_path = os.path.join(test_data_dir, "mimic-text-tables.gt")

        text_enet, col_enet = load_saved_model(args, test_text_f.shape[1], test_col_f.shape[1])
        phase_evaluate(
            args,
            text_enet,
            col_enet,
            test_text_ids,
            test_col_ids,
            test_text_f,
            test_col_f,
            test_gt_path,
            "test",
        )
        print("\n" + "=" * 60)
        print("Test evaluation complete! Results saved to %s" % args.output_dir)
        print("=" * 60)
        return

    # Convert + build features for train/val
    train_data_dir = phase_convert(args, "train", args.loki_train_file, args.max_train_examples)
    val_data_dir = phase_convert(args, "val", args.loki_eval_file, args.max_eval_examples)
    train_text_ids, train_col_ids, train_text_f, train_col_f = phase_build_features(args, train_data_dir, "train")
    val_text_ids, val_col_ids, val_text_f, val_col_f = phase_build_features(args, val_data_dir, "val")
    train_gt_path = os.path.join(train_data_dir, "mimic-text-tables.gt")
    val_gt_path = os.path.join(val_data_dir, "mimic-text-tables.gt")

    # Train or load
    if not args.eval_only:
        text_enet, col_enet = phase_train_cmdl_style(
            args,
            train_text_ids,
            train_col_ids,
            train_text_f,
            train_col_f,
            train_gt_path,
        )
    else:
        text_enet, col_enet = load_saved_model(args, train_text_f.shape[1], train_col_f.shape[1])

    # Evaluate on val
    phase_evaluate(
        args,
        text_enet,
        col_enet,
        val_text_ids,
        val_col_ids,
        val_text_f,
        val_col_f,
        val_gt_path,
        "val",
    )

    # Evaluate on test if present
    if os.path.exists(args.loki_test_file):
        test_data_dir = phase_convert(args, "test", args.loki_test_file, args.max_test_examples)
        test_text_ids, test_col_ids, test_text_f, test_col_f = phase_build_features(args, test_data_dir, "test")
        test_gt_path = os.path.join(test_data_dir, "mimic-text-tables.gt")
        phase_evaluate(
            args,
            text_enet,
            col_enet,
            test_text_ids,
            test_col_ids,
            test_text_f,
            test_col_f,
            test_gt_path,
            "test",
        )

    print("\n" + "=" * 60)
    print("Pipeline complete! Results saved to %s" % args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

