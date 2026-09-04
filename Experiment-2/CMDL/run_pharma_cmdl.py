"""
run_pharma_cmdl.py - CMDL-style Doc-to-Table training on Pharma dataset.

Trains CMDL on the Pharma (PubMed Documents + DrugBank Tables) dataset,
corresponding to Table 2 row 1B (Doc_to_Table) in the CMDL paper.

Pipeline:
  Phase 0: Split & Copy -- read GT, split text IDs 70/15/15,
           copy text/table files into pharma_data/{split}/ folders
  Phase 1: Build Features -- profile + WEM featurize texts and tables
  Phase 2: Train -- CMDL-style column-text joint training with TripletLoss
  Phase 3: Evaluate -- P@K / R@K / F1@K on val and test
"""

import argparse
import csv
import json
import os
import sys
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from indexer.wem import WEM
from indexer.trained_embeddings import TrainedEmbeddings, TrainedEmbeddingsIndexer


# ---------------------------------------------------------------------------
#  Dataset & Model (same as CMDL notebook / MIMIC script)
# ---------------------------------------------------------------------------
class DataCombiner(Dataset):
    """Minimal dataset wrapper used by CMDL notebook training."""

    def __init__(self, ids, features):
        self.ids = ids
        self.features = features

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return (i, self.ids[i], self.features[i])


class PairedCmdlSampler:
    """Samplers text queries and perfectly paired positive/negative columns from anchor_groups json"""
    def __init__(self, text_ids, col_ids, anchor_groups, batch_size, shuffle=True):
        self.text_ids = text_ids
        self.col_ids = col_ids
        
        # Build mapping from col_id to integer index for fast lookup
        self.col_id_to_idx = {cid: i for i, cid in enumerate(self.col_ids)}
        
        # Only keep anchors that have at least one column we actually featurized
        self.valid_anchors = []
        for i, tid in enumerate(self.text_ids):
            tid_clean = tid.replace(".txt", "") if tid.endswith(".txt") else tid
            group_cids = anchor_groups.get(tid_clean, [])
            valid_c_indices = [self.col_id_to_idx[c] for c in group_cids if c in self.col_id_to_idx]
            if len(valid_c_indices) > 0:
                self.valid_anchors.append((i, valid_c_indices))
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        anchors = list(self.valid_anchors)
        if self.shuffle:
            random.shuffle(anchors)
            
        for i in range(0, len(anchors), self.batch_size):
            batch_anchors = anchors[i:i+self.batch_size]
            
            # ensure batch size at least 2 for batchnorm
            if len(batch_anchors) == 1 and len(anchors) > 1:
                batch_anchors.append(anchors[0])
            elif len(batch_anchors) == 1:
                batch_anchors.append(batch_anchors[0])
            
            t_idx = []
            c_idx = set()
            for t, c_list in batch_anchors:
                t_idx.append(t)
                c_idx.update(c_list)
                
            yield torch.tensor(t_idx, dtype=torch.long), torch.tensor(list(c_idx), dtype=torch.long)
            
    def __len__(self):
        return (len(self.valid_anchors) + self.batch_size - 1) // self.batch_size


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


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------
def read_text_to_table_gt(gt_file, sep=","):
    gt_map = {}
    if not os.path.exists(gt_file):
        return gt_map
    with open(gt_file, "r") as f:
        csvf = csv.reader(f, delimiter=sep)
        for row in csvf:
            if len(row) < 2: continue
            key, value = row[0].strip(), row[1].strip()
            # In DrugBank GT, we ensure table names match filenames
            gt_map.setdefault(key, []).append(value)
    return gt_map

def write_csv_list(fp, rows):
    with open(fp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r}\n")

def prepare_gt_dicts(col_ids, gt_map):
    col_to_table = [None] * len(col_ids)
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

    return text_to_tables, col_to_table

def build_sub_label_matrix(t_idx, c_idx, text_ids, text_to_tables, col_to_table, device):
    t_idx_list = t_idx.tolist()
    c_idx_list = c_idx.tolist()
    
    sub_label = torch.zeros(len(t_idx_list), len(c_idx_list), device=device)
    for i, t in enumerate(t_idx_list):
        tid = text_ids[t]
        tid_clean = tid.replace(".txt", "") if tid.endswith(".txt") else tid
        target_tables = text_to_tables.get(tid_clean, set())
        if not target_tables: continue
        for j, c in enumerate(c_idx_list):
            if col_to_table[c] in target_tables:
                sub_label[i, j] = 1.0
                
    return sub_label

# Removed _read_abstract_only and phase_split_and_copy to split_legacy_pharma.py


# ---------------------------------------------------------------------------
#  Phase 1: Build Features
# ---------------------------------------------------------------------------
def phase_build_features(args, split_name):
    """Profile and featurize texts + tables for a given split from JSON format."""
    feature_dir = os.path.join(args.output_dir, "features", split_name)
    os.makedirs(feature_dir, exist_ok=True)

    text_feat_path = os.path.join(feature_dir, "pharma-textfeatures.pt")
    col_feat_path = os.path.join(feature_dir, "pharma-columnfeatures.pt")

    # In the flipped paradigm, the ground truth is encoded directly in the JSON file.
    # However, to maintain compatibility with CMDL's evaluation and label matrix builder,
    # we will reconstruct a `gt_map` and save it to a dummy gt_path.
    gt_path = os.path.join(feature_dir, "pharma-text-tables-reconstructed.gt")

    anchor_groups_path = os.path.join(feature_dir, "pharma-anchor-groups.json")

    if args.skip_features and os.path.exists(text_feat_path) and os.path.exists(col_feat_path) and os.path.exists(anchor_groups_path):
        print(f"[Phase 1] Skipping feature building for {split_name} (cached)")
        with open(os.path.join(feature_dir, "pharma-textids.list"), encoding="utf-8") as f:
            text_ids = [r.strip() for r in f if r.strip()]
        with open(os.path.join(feature_dir, "pharma-colids.list"), encoding="utf-8") as f:
            col_ids = [r.strip() for r in f if r.strip()]
        with open(anchor_groups_path, "r", encoding="utf-8") as f:
            anchor_groups = json.load(f)
        text_f = torch.load(text_feat_path, weights_only=True)
        table_f = torch.load(col_feat_path, weights_only=True)
        return text_ids, col_ids, text_f, table_f, gt_path, anchor_groups

    print(f"[Phase 1] Building features for {split_name} from JSON...")
    
    # Path to the JSON data
    json_path = os.path.join(args.json_data_dir, f"{split_name}_row_level.json")
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON dataset not found at {json_path}")
        sys.exit(1)
        
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"  Loaded {len(dataset)} examples from {json_path}")

    # Set up WEM
    wem = WEM(args.wem_model_path, args.wem_dim)
    
    # Simple tokenizer (matches text_profiler logic)
    import re
    _WORD_RE = re.compile(r"[a-zA-Z0-9]+")
    def tokenize(text: str) -> list[str]:
        return [w for w in _WORD_RE.findall(text.lower()) if len(w) > 1]
        
    texts_dict = {}
    tables_dict = {} # tbl_name -> {col_name: [val1, val2...]}
    gt_map = {}
    anchor_groups = {} # doc_id -> list of col_ids (positives + negatives for this anchor)
    
    # Parse JSON
    for item in dataset:
        # 1. Parse Document (Anchor)
        doc_id = str(item["anchor_id"])
        
        # Combine anchor sentences into full text
        sentences = item["anchor_sentences"]
        full_text = " ".join(sentences)
        texts_dict[doc_id] = tokenize(full_text)
        
        # 2. Parse Tables (Positives & Negatives)
        gt_map[doc_id] = []
        anchor_col_ids = set()
        
        # Helper to parse structured table objects into column dicts
        def parse_table_chunk(table_obj, is_positive):
            tbl_name = f"table_{table_obj['id']}.csv"
            headers = table_obj.get("headers", [])
            rows = table_obj.get("rows", [])
            
            if is_positive:
                gt_map[doc_id].append(tbl_name)
                
            if tbl_name not in tables_dict:
                tables_dict[tbl_name] = {}
                
            for row in rows:
                content = row.get("content", [])
                for col_hdr, col_val in zip(headers, content):
                    col_hdr = str(col_hdr).replace('\n', ' ').replace('\r', '').strip()
                    col_val = str(col_val).replace('\n', ' ').replace('\r', '').strip()
                    if col_val:
                        if col_hdr not in tables_dict[tbl_name]:
                            tables_dict[tbl_name][col_hdr] = []
                        tables_dict[tbl_name][col_hdr].append(col_val)
                        anchor_col_ids.add(f"{tbl_name},{col_hdr}")
                                
        # Primary Positive
        if "primary_positive" in item:
            parse_table_chunk(item["primary_positive"], is_positive=True)
            
        # Additional Positives
        for pos in item.get("additional_positives", []):
             parse_table_chunk(pos, is_positive=True)
             
        # Negatives
        for neg in item.get("negatives", []):
             parse_table_chunk(neg, is_positive=False)

        anchor_groups[doc_id] = list(anchor_col_ids)

    # Reconstruct GT file
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for tid, tbls in gt_map.items():
            for tbl in tbls:
                writer.writerow([tid, tbl])

    with open(anchor_groups_path, "w", encoding="utf-8") as f:
        json.dump(anchor_groups, f, indent=2)

    print(f"  Parsed {len(texts_dict)} unique docs, {len(tables_dict)} unique table chunks")

    # Generate Text Features
    print("  Generating Text Embeddings...")
    text_ids = list(texts_dict.keys())
    text_f_list = []
    for tid in text_ids:
        tokens = texts_dict[tid]
        word_embs = [wem.get_vector(w) for w in tokens if wem.get_vector(w) is not None]
        if not word_embs:
            emb = np.zeros(args.wem_dim)
        else:
            emb = np.mean(word_embs, axis=0) # Average word embeddings
        text_f_list.append(emb)
    text_f = torch.tensor(np.array(text_f_list), dtype=torch.float32)

    # Generate Column Features
    print("  Generating Column Embeddings...")
    col_ids = []
    col_f_list = []
    for tbl_name, col_dict in tables_dict.items():
        for col_hdr, vals in col_dict.items():
            col_id = f"{tbl_name},{col_hdr}"
            col_ids.append(col_id)
            
            # Tokenize all values in this column for this table
            col_tokens = []
            for v in vals:
                col_tokens.extend(tokenize(v))
                
            word_embs = [wem.get_vector(w) for w in col_tokens if wem.get_vector(w) is not None]
            if not word_embs:
                emb = np.zeros(args.wem_dim)
            else:
                emb = np.mean(word_embs, axis=0)
            col_f_list.append(emb)
            
    col_f = torch.tensor(np.array(col_f_list), dtype=torch.float32)

    write_csv_list(os.path.join(feature_dir, "pharma-textids.list"), text_ids)
    write_csv_list(os.path.join(feature_dir, "pharma-colids.list"), col_ids)
    torch.save(text_f, text_feat_path)
    torch.save(col_f, col_feat_path)
    
    print(
        "  Saved: %d text features (%s), %d column features (%s)"
        % (len(text_ids), text_f.shape, len(col_ids), col_f.shape)
    )
    return text_ids, col_ids, text_f, col_f, gt_path, anchor_groups


# ---------------------------------------------------------------------------
#  Phase 2: Train
# ---------------------------------------------------------------------------
def phase_train_cmdl_style(
    args,
    train_text_ids,
    train_col_ids,
    train_text_f,
    train_col_f,
    train_gt_path,
    anchor_groups
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
    print("[Phase 2] CMDL-style joint training on device: %s" % device)

    # Prepare dynamic dictionary lookups instead of massive memory matrices
    train_gt_map = read_text_to_table_gt(train_gt_path)
    text_to_tables, col_to_table = prepare_gt_dicts(train_col_ids, train_gt_map)

    n_text = len(train_text_ids)
    n_col = len(train_col_ids)

    print("  Training subset: %d texts, %d columns" % (n_text, n_col))

    # Optional paper-style mini-batch sizing
    text_batch_size = args.batch_size

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

    paired_sampler = PairedCmdlSampler(train_text_ids, train_col_ids, anchor_groups, text_batch_size, shuffle=True)

    if args.steps_per_epoch <= 0:
        steps_per_epoch = len(paired_sampler)
    else:
        steps_per_epoch = args.steps_per_epoch

    print(
        "  epochs=%d text_bsz=%d steps_per_epoch=%s"
        % (args.epochs, text_batch_size, str(steps_per_epoch))
    )

    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        text_enet.train()
        col_enet.train()

        epoch_loss = 0.0
        epoch_prec = 0.0
        step_count = 0

        sampler_it = iter(paired_sampler)

        while step_count < steps_per_epoch:
            try:
                t_idx, c_idx = next(sampler_it)
            except StopIteration:
                sampler_it = iter(paired_sampler)
                t_idx, c_idx = next(sampler_it)

            # --- INDEX EXACT MATCHES ON CPU / GPU DYNAMICALLY ---
            t_batch = train_text_f[t_idx]
            c_batch = train_col_f[c_idx]
            # Build sub-label matrix and mask dynamically to avoid massive OOM allocations
            sub_label = build_sub_label_matrix(t_idx, c_idx, train_text_ids, text_to_tables, col_to_table, device)

            t_idx = t_idx.to(device)
            c_idx = c_idx.to(device)
            t_batch = t_batch.to(device)
            c_batch = c_batch.to(device)

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


# ---------------------------------------------------------------------------
#  Phase 3: Evaluate
# ---------------------------------------------------------------------------
def eval_matches(gt_map, query_matches):
    tp, fp, fn = 0, 0, 0
    for idx, matches in query_matches.items():
        if idx not in gt_map:
            continue
        gt = gt_map[idx]
        is_match = set(matches).intersection(set(gt))

        true_matches = len(is_match)
        false_matches = len(matches) - true_matches
        non_matches = len(gt) - true_matches

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

    text_emb_path = os.path.join(feature_dir, "pharma-0-trainedtext.npy")
    col_emb_path = os.path.join(feature_dir, "pharma-0-trainedcolumns.npy")
    np.save(text_emb_path, text_emb)
    np.save(col_emb_path, col_emb)

    print(f"[DEBUG] text_ids len: {len(text_ids)}, text_emb shape: {text_emb.shape}")
    print(f"[DEBUG] col_ids len: {len(col_ids)}, col_emb shape: {col_emb.shape}")

    text_emb_obj = TrainedEmbeddings(text_ids, text_emb_path)
    col_emb_obj = TrainedEmbeddings(col_ids, col_emb_path)
    col_emb_ind = TrainedEmbeddingsIndexer("pharma-trained", col_emb_obj, "table")
    col_emb_ind.create_index()
    for cid in col_ids:
        col_emb_ind.index_doc(cid)
    col_emb_ind.commit_index()

    raw_gt_map = read_text_to_table_gt(gt_path)
    text_id_set = set(text_ids)
    gt_map = {}
    skipped = 0
    for raw_text_id, raw_table_ids in raw_gt_map.items():
        # Try to find the text ID in profiled data (may or may not have extension)
        if raw_text_id in text_id_set:
            mapped_text_id = raw_text_id
        elif raw_text_id + ".txt" in text_id_set:
            mapped_text_id = raw_text_id + ".txt"
        else:
            skipped += 1
            continue
        # Tables in GT may or may not have .csv extension; ensure they match col_ids format
        mapped_tables = []
        for t in raw_table_ids:
            if not t.endswith(".csv"):
                mapped_tables.append(t + ".csv")
            else:
                mapped_tables.append(t)
        gt_map[mapped_text_id] = mapped_tables

    print("\n[Phase 3] Evaluation on %s:" % split_name)
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


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="CMDL-style Doc-to-Table training/evaluation on Pharma (PubMed + DrugBank)"
    )

    # Input data paths
    parser.add_argument(
        "--json_data_dir", type=str, default="Datasets/pharma_flipped_structured",
        help="Directory containing the structured JSON splits (train_row_level.json, etc.)"
    )
    # WEM model and features
    parser.add_argument(
        "--wem_model_path", type=str, default="CMDL/resources/fasttext/cc/cc.en.300.bin"
    )
    parser.add_argument("--wem_dim", type=int, default=300)

    # CMDL-style training parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--steps_per_epoch", type=int, default=26,
        help="CMDL notebook default is 26. Set <=0 to auto-derive from loaders."
    )
    parser.add_argument(
        "--ensure_all_examples_per_epoch", 
        action="store_true",
        default=True,
        help="Run enough steps so each example is seen once (Enabled by default)."
    )
    parser.add_argument(
        "--disable_ensure_all_examples_per_epoch",
        dest="ensure_all_examples_per_epoch",
        action="store_false",
        help="Turn off the default 'ensure all examples' behavior."
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--neg_weight", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--output_size", type=int, default=100)

    # Output and control flags
    parser.add_argument("--output_dir", type=str, default="CMDL/output_cmdl_pharma")
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # seed_everything(args.seed)

    print("=" * 60)
    print("CMDL-Style Pipeline on Pharma (Doc-to-Table)")
    if args.test_only:
        print("  Mode: TEST ONLY (loading saved model)")
    elif args.eval_only:
        print("  Mode: EVAL ONLY (loading saved model, val + test)")
    else:
        print("  Mode: FULL PIPELINE (split + features + train + eval)")
    print("=" * 60)

    # Phase 0: Split & Copy data into folders (REMOVED - now ingests JSON directly)
    # phase_split_and_copy(args)

    # TEST-ONLY mode
    if args.test_only:
        test_text_ids, test_col_ids, test_text_f, test_col_f, test_gt_path, test_anchor_groups = (
            phase_build_features(args, "test")
        )
        text_enet, col_enet = load_saved_model(args, test_text_f.shape[1], test_col_f.shape[1])
        phase_evaluate(
            args, text_enet, col_enet,
            test_text_ids, test_col_ids, test_text_f, test_col_f,
            test_gt_path, "test",
        )
        print("\n" + "=" * 60)
        print("Test evaluation complete! Results saved to %s" % args.output_dir)
        print("=" * 60)
        return

    # Build features for train and val
    train_text_ids, train_col_ids, train_text_f, train_col_f, train_gt_path, train_anchor_groups = (
        phase_build_features(args, "train")
    )
    val_text_ids, val_col_ids, val_text_f, val_col_f, val_gt_path, val_anchor_groups = (
        phase_build_features(args, "val")
    )

    # Train or load
    if not args.eval_only:
        text_enet, col_enet = phase_train_cmdl_style(
            args,
            train_text_ids, train_col_ids,
            train_text_f, train_col_f,
            train_gt_path,
            train_anchor_groups
        )
    else:
        text_enet, col_enet = load_saved_model(args, train_text_f.shape[1], train_col_f.shape[1])

    # Evaluate on val
    phase_evaluate(
        args, text_enet, col_enet,
        val_text_ids, val_col_ids, val_text_f, val_col_f,
        val_gt_path, "val",
    )

    # Evaluate on test
    test_text_ids, test_col_ids, test_text_f, test_col_f, test_gt_path, test_anchor_groups = (
        phase_build_features(args, "test")
    )
    phase_evaluate(
        args, text_enet, col_enet,
        test_text_ids, test_col_ids, test_text_f, test_col_f,
        test_gt_path, "test",
    )

    print("\n" + "=" * 60)
    print("Pipeline complete! Results saved to %s" % args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
