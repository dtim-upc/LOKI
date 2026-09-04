"""
Data loader for fine-tuning TaBERT on contrastive datasets.

Dataset format: pharma_flipped_structured
  - Anchors are DOCUMENTS (NL sentences)
  - Positives/Negatives are TABLES (headers + rows with content arrays)
"""

import json
import random
from typing import List, Dict, Any, Tuple, Generator, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset

from table_bert.table import Table, Column


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_row_level_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load a row-level JSON dataset file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def _extract_sentences(sentences_data) -> List[str]:
    """Extract sentence strings from various formats (list of str, list of dict, dict)."""
    extracted = []
    if isinstance(sentences_data, dict):
        try:
            sorted_keys = sorted(sentences_data.keys(), key=lambda k: int(k))
            for k in sorted_keys:
                item = sentences_data[k]
                if isinstance(item, dict):
                    extracted.append(item.get("text", ""))
                elif isinstance(item, str):
                    extracted.append(item)
        except ValueError:
            for item in sentences_data.values():
                if isinstance(item, dict):
                    extracted.append(item.get("text", ""))
                elif isinstance(item, str):
                    extracted.append(item)
    elif isinstance(sentences_data, list):
        for item in sentences_data:
            if isinstance(item, dict):
                extracted.append(item.get("text", ""))
            elif isinstance(item, str):
                extracted.append(item)
    return [t for t in extracted if t]


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def structured_to_tabert_table(
    table_id: str,
    headers: List[str],
    row_content_lists: List[List],
    tokenizer=None,
    max_rows: int = 3,
) -> Optional[Table]:
    """
    Build a TaBERT Table directly from structured headers + content arrays.

    Args:
        table_id:          Unique identifier for this table.
        headers:           Column name strings, e.g. ["drugbank_id", "name"].
        row_content_lists: List of rows, each a list of cell values aligned
                           with *headers*, e.g. [["DB001", "Drug A"], ...].
        tokenizer:         If provided, tokenize the table immediately.
        max_rows:          Truncate to this many rows (default 3, matching
                           TaBERT pre-training k=3).

    Returns:
        A TaBERT ``Table`` object, or *None* if no valid data.
    """
    if not headers or not row_content_lists:
        return None

    # Normalize cell values to strings
    data = []
    for row in row_content_lists:
        cells = [str(c).strip() if c is not None else "" for c in row]
        # Pad / truncate to match header count
        while len(cells) < len(headers):
            cells.append("")
        cells = cells[:len(headers)]
        data.append(cells)

    if not data:
        return None

    if max_rows > 0:
        data = data[:max_rows]

    # Build Column objects with sample_value from first row
    columns = []
    for i, name in enumerate(headers):
        col_name = str(name).strip() if name else "unknown"
        sample = data[0][i] if data[0][i] else col_name
        columns.append(Column(col_name, 'text', sample_value=sample))

    table = Table(id=str(table_id), header=columns, data=data)

    if tokenizer is not None:
        table.tokenize(tokenizer)

    return table


def example_to_tabert_table(example: Dict[str, Any], tokenizer=None) -> Optional[Table]:
    """
    Convert a dataset example's anchor table into a TaBERT Table object.

    Non-flipped format has:
      - anchor_headers: list of column name strings
      - anchor_rows: list of {row_idx, content: [...], formatted: "..."}

    Returns a Table with header columns and row data as lists of strings.
    """
    headers_raw = example.get("anchor_headers", [])
    rows_raw = example.get("anchor_rows", [])

    if not headers_raw or not rows_raw:
        return None

    # Build Column objects — type is 'text' for all pharma columns
    columns = []
    for h in headers_raw:
        col_name = str(h).strip() if h else "unknown"
        columns.append(Column(col_name, 'text'))

    # Build row data: list of lists of cell value strings
    row_data = []
    for row in rows_raw:
        if isinstance(row, dict):
            content = row.get("content", [])
            cells = [str(c).strip() if c is not None else "" for c in content]
        elif isinstance(row, list):
            cells = [str(c).strip() if c is not None else "" for c in row]
        else:
            continue
        # Ensure row has same number of cells as headers
        while len(cells) < len(columns):
            cells.append("")
        cells = cells[:len(columns)]
        row_data.append(cells)

    if not row_data:
        return None

    # Set sample_value on each column from first row
    for i, col in enumerate(columns):
        col.sample_value = row_data[0][i] if row_data[0][i] else col.name

    table = Table(
        id=str(example.get("anchor_id", "unknown")),
        header=columns,
        data=row_data
    )

    if tokenizer is not None:
        table.tokenize(tokenizer)

    return table


def example_to_single_row_tables(
    example: Dict[str, Any], tokenizer=None
) -> List[Table]:
    """
    Convert each row of an example into a separate single-row TaBERT Table.
    Used for row-level evaluation (non-flipped format).
    """
    headers_raw = example.get("anchor_headers", [])
    rows_raw = example.get("anchor_rows", [])

    if not headers_raw or not rows_raw:
        return []

    columns_template = []
    for h in headers_raw:
        col_name = str(h).strip() if h else "unknown"
        columns_template.append(Column(col_name, 'text'))

    tables = []
    for row in rows_raw:
        if isinstance(row, dict):
            content = row.get("content", [])
            cells = [str(c).strip() if c is not None else "" for c in content]
        elif isinstance(row, list):
            cells = [str(c).strip() if c is not None else "" for c in row]
        else:
            continue

        while len(cells) < len(columns_template):
            cells.append("")
        cells = cells[:len(columns_template)]

        # Create fresh columns per row with sample_value set
        cols = []
        for i, ct in enumerate(columns_template):
            c = Column(ct.name, ct.type, sample_value=cells[i] if cells[i] else ct.name)
            cols.append(c)

        t = Table(
            id=str(example.get("anchor_id", "unknown")),
            header=cols,
            data=[cells]
        )
        if tokenizer is not None:
            t.tokenize(tokenizer)
        tables.append(t)

    return tables


# ---------------------------------------------------------------------------
# Structured extraction helpers
# ---------------------------------------------------------------------------

def _extract_table_structured(table_obj: Dict[str, Any]) -> Optional[Tuple[List[str], List[List]]]:
    """
    Extract (headers, row_content_lists) from a table object in the new format.

    Returns None if the table_obj lacks the required fields.
    """
    headers = table_obj.get("headers", [])
    rows = table_obj.get("rows", [])
    if not headers or not rows:
        return None
    row_content_lists = []
    for r in rows:
        content = r.get("content", []) if isinstance(r, dict) else []
        if content:
            row_content_lists.append(content)
    if not row_content_lists:
        return None
    return headers, row_content_lists


# ---------------------------------------------------------------------------
# Triplet generation
# ---------------------------------------------------------------------------

def generate_triplets_for_example(
    example: Dict[str, Any],
    strategy: str = "limited",
    max_triplets: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generate contrastive triplets from a single example.

    Returns dicts with structured table data:
      - ``positive_headers`` / ``positive_rows`` (list of content lists)
      - ``negative_headers`` / ``negative_rows``
    """
    anchor_id = example.get("anchor_id")
    if anchor_id is None:
        return []

    # Collect positives — extract structured table data
    all_positives = []
    primary = example.get("primary_positive", {})
    if primary:
        pid = primary.get("id")
        structured = _extract_table_structured(primary)
        if pid is not None and structured is not None:
            headers, rows = structured
            all_positives.append((pid, headers, rows))

    for add_pos in example.get("additional_positives", []):
        pid = add_pos.get("id")
        structured = _extract_table_structured(add_pos)
        if pid is not None and structured is not None:
            headers, rows = structured
            all_positives.append((pid, headers, rows))

    # Collect negatives
    all_negatives = []
    for neg in example.get("negatives", []):
        nid = neg.get("id")
        structured = _extract_table_structured(neg)
        if nid is not None and structured is not None:
            headers, rows = structured
            all_negatives.append((nid, headers, rows))

    if not all_positives or not all_negatives:
        return []

    triplets = []

    def _make_triplet(pos_tuple, neg_tuple):
        pos_id, pos_headers, pos_rows = pos_tuple
        neg_id, neg_headers, neg_rows = neg_tuple
        return {
            'anchor_id': anchor_id,
            'positive_id': pos_id,
            'negative_id': neg_id,
            'positive_headers': pos_headers,
            'positive_rows': pos_rows,
            'negative_headers': neg_headers,
            'negative_rows': neg_rows,
        }

    if strategy == "primary_only":
        triplets.append(_make_triplet(all_positives[0], all_negatives[0]))
    elif strategy == "limited":
        count = 0
        for pos in all_positives:
            for neg in all_negatives:
                if count >= max_triplets:
                    break
                triplets.append(_make_triplet(pos, neg))
                count += 1
            if count >= max_triplets:
                break
    elif strategy == "random":
        pairs = [(p, n) for p in all_positives for n in all_negatives]
        random.shuffle(pairs)
        for pos, neg in pairs[:max_triplets]:
            triplets.append(_make_triplet(pos, neg))
    else:  # "full"
        for pos in all_positives:
            for neg in all_negatives:
                triplets.append(_make_triplet(pos, neg))

    return triplets


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TaBERTContrastiveDataset(Dataset):
    """
    PyTorch Dataset for TaBERT contrastive fine-tuning.

    Anchors are DOCUMENTS (NL sentences), Positives/Negatives are TABLES
    built from structured headers + content arrays.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        is_flipped: bool = True,
        sample_row_num: int = 3,
        triplet_strategy: str = "limited",
        max_triplets_per_example: int = 10,
        max_context_len: int = 128,
        max_examples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.is_flipped = is_flipped
        self.sample_row_num = sample_row_num
        self.max_context_len = max_context_len

        print(f"Loading dataset from {data_path} (is_flipped={is_flipped})...")
        raw_data = load_row_level_dataset(data_path)
        if max_examples is not None:
            raw_data = raw_data[:max_examples]

        self.items: List[Dict[str, Any]] = []
        skipped = 0

        for example in raw_data:
            triplets = generate_triplets_for_example(
                example,
                strategy=triplet_strategy,
                max_triplets=max_triplets_per_example,
            )
            if not triplets:
                skipped += 1
                continue

            if is_flipped:
                # Anchor = doc sentences (NL context), pos/neg = structured tables
                doc_sentences = example.get("anchor_sentences", [])
                if not doc_sentences:
                    skipped += 1
                    continue
                for triplet in triplets:
                    pos_table = structured_to_tabert_table(
                        table_id=triplet['positive_id'],
                        headers=triplet['positive_headers'],
                        row_content_lists=triplet['positive_rows'],
                        tokenizer=tokenizer,
                        max_rows=sample_row_num,
                    )
                    neg_table = structured_to_tabert_table(
                        table_id=triplet['negative_id'],
                        headers=triplet['negative_headers'],
                        row_content_lists=triplet['negative_rows'],
                        tokenizer=tokenizer,
                        max_rows=sample_row_num,
                    )
                    if pos_table is None or neg_table is None:
                        continue
                    self.items.append({
                        'doc_sentences': doc_sentences,
                        'pos_table': pos_table,
                        'neg_table': neg_table,
                    })
            else:
                # Anchor = table rows (Table object), pos/neg = doc sentences
                table = example_to_tabert_table(example, tokenizer=tokenizer)
                if table is None:
                    skipped += 1
                    continue
                if sample_row_num > 0 and len(table.data) > sample_row_num:
                    table = table.with_rows(table.data[:sample_row_num])
                    table.tokenize(tokenizer)
                for triplet in triplets:
                    # Non-flipped: positives are doc sentences
                    pos_sents = _extract_sentences(
                        example.get("primary_positive", {}).get("sentences", [])
                    )
                    neg_sents = []
                    for neg in example.get("negatives", []):
                        neg_sents.extend(_extract_sentences(neg.get("sentences", [])))
                    if not pos_sents or not neg_sents:
                        continue
                    self.items.append({
                        'table': table,
                        'pos_sentences': pos_sents,
                        'neg_sentences': neg_sents,
                    })

        print(f"Built {len(self.items)} triplets from {len(raw_data)} examples "
              f"({skipped} skipped)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        if self.is_flipped:
            # Flipped: anchor doc sentence → context, pos/neg → tables
            doc_sent = random.choice(item['doc_sentences'])
            context_tokens = self.tokenizer.tokenize(doc_sent)[:self.max_context_len]
            return {
                'context': context_tokens,
                'pos_table': item['pos_table'],
                'neg_table': item['neg_table'],
            }
        else:
            # Non-flipped: anchor → table, pos/neg sentences → contexts
            pos_sent = random.choice(item['pos_sentences'])
            neg_sent = random.choice(item['neg_sentences'])
            pos_tokens = self.tokenizer.tokenize(pos_sent)[:self.max_context_len]
            neg_tokens = self.tokenizer.tokenize(neg_sent)[:self.max_context_len]
            return {
                'table': item['table'],
                'pos_context': pos_tokens,
                'neg_context': neg_tokens,
            }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_triplets(batch: List[Dict]) -> Dict[str, List]:
    """Collate triplets — format depends on is_flipped (auto-detected from keys)."""
    if 'context' in batch[0]:
        # Flipped mode
        return {
            'contexts': [item['context'] for item in batch],
            'pos_tables': [item['pos_table'] for item in batch],
            'neg_tables': [item['neg_table'] for item in batch],
        }
    else:
        # Non-flipped mode
        return {
            'tables': [item['table'] for item in batch],
            'pos_contexts': [item['pos_context'] for item in batch],
            'neg_contexts': [item['neg_context'] for item in batch],
        }


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def load_annotations(annotation_file: str) -> Dict[int, List[List[int]]]:
    """Load row-sentence annotations for evaluation (Annotated_Test.json)."""
    if not Path(annotation_file).exists():
        print(f"Annotation file not found: {annotation_file}")
        return {}

    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = {}
    if isinstance(data, list):
        for entry in data:
            anchor_id = entry.get("anchor_id")
            if anchor_id is not None:
                annotations[anchor_id] = entry.get("highlighted_cells", [])
    return annotations
