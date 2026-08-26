import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Generator, Set, Optional
from tqdm.auto import tqdm
import random


_DEFAULT_COLUMN_SKETCH_VALUES = 3

class IdBasedEmbeddingCache:
    """
    Cache for table and context embeddings using IDs as keys instead of text hashes.
    This approach is more memory efficient as it stores embeddings for whole tables 
    and contexts together.
    """
    def __init__(self):
        self.table_cache = {}  # Maps anchor_id to matrix of row embeddings
        self.context_cache = {}  # Maps context_id to matrix of sentence embeddings
        self.table_schema_cache = {}  # Maps table_id to schema sketch embedding tensor(s)
        self.table_cell_cache = {}  # Maps table_id to [num_rows, num_cols, embedding_dim] cell embeddings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device} for embedding cache")
    
    def add_table_embeddings(self, anchor_id: int, row_embeddings: torch.Tensor):
        """
        Add a table's row embeddings to the cache.
        
        Args:
            anchor_id: ID of the anchor table
            row_embeddings: Tensor of shape [num_rows, embedding_dim] containing embeddings for all rows
        """
        # Move to appropriate device and store
        self.table_cache[anchor_id] = row_embeddings.to(self.device)
    
    def add_context_embeddings(self, context_id: int, sentence_embeddings: torch.Tensor):
        """
        Add a context's sentence embeddings to the cache.
        
        Args:
            context_id: ID of the context (from primary_positive, additional_positives, or negatives)
            sentence_embeddings: Tensor of shape [num_sentences, embedding_dim] containing embeddings for all sentences
        """
        # Move to appropriate device and store
        self.context_cache[context_id] = sentence_embeddings.to(self.device)

    def add_table_schema_embedding(self, table_id: int, schema_embedding: torch.Tensor):
        """
        Add table-side schema/sketch embeddings to the cache.

        Args:
            table_id: ID of the table
            schema_embedding: Tensor of shape [embedding_dim], [1, embedding_dim],
                or [num_columns, embedding_dim]
        """
        if schema_embedding.dim() > 2 and schema_embedding.size(0) == 1:
            schema_embedding = schema_embedding.squeeze(0)
        self.table_schema_cache[table_id] = schema_embedding.to(self.device)

    def add_table_cell_embeddings(self, table_id: int, cell_embeddings: torch.Tensor):
        """
        Add table-side cell embeddings to the cache.

        Args:
            table_id: ID of the table
            cell_embeddings: Tensor of shape [num_rows, num_cols, embedding_dim]
        """
        if cell_embeddings.dim() > 3 and cell_embeddings.size(0) == 1:
            cell_embeddings = cell_embeddings.squeeze(0)
        self.table_cell_cache[table_id] = cell_embeddings.to(self.device)
    
    def get_table_embeddings(self, anchor_id: int) -> Optional[torch.Tensor]:
        """
        Get table embeddings from the cache if available.
        
        Args:
            anchor_id: ID of the anchor table
            
        Returns:
            Tensor of shape [num_rows, embedding_dim] or None if not in cache
            
        Note:
            Returns a CLONE of the cached tensor to ensure it can be used in
            gradient computation. SentenceTransformer.encode() creates tensors
            under torch.inference_mode() which can't be used in autograd.
        """
        cached = self.table_cache.get(anchor_id)
        if cached is not None:
            return cached.clone()  # Clone to enable gradient tracking
        return None
    
    def get_context_embeddings(self, context_id: int) -> Optional[torch.Tensor]:
        """
        Get context embeddings from the cache if available.
        
        Args:
            context_id: ID of the context
            
        Returns:
            Tensor of shape [num_sentences, embedding_dim] or None if not in cache
            
        Note:
            Returns a CLONE of the cached tensor to ensure it can be used in
            gradient computation. SentenceTransformer.encode() creates tensors
            under torch.inference_mode() which can't be used in autograd.
        """
        cached = self.context_cache.get(context_id)
        if cached is not None:
            return cached.clone()  # Clone to enable gradient tracking
        return None

    def get_table_schema_embedding(self, table_id: int) -> Optional[torch.Tensor]:
        """
        Get cached table-side schema/sketch embeddings if available.

        Args:
            table_id: ID of the table

        Returns:
            Tensor of shape [embedding_dim] or [num_columns, embedding_dim],
            or None if not in cache
        """
        cached = self.table_schema_cache.get(table_id)
        if cached is not None:
            return cached.clone()
        return None

    def get_table_cell_embeddings(self, table_id: int) -> Optional[torch.Tensor]:
        """
        Get cached table-side cell embeddings if available.

        Args:
            table_id: ID of the table

        Returns:
            Tensor of shape [num_rows, num_cols, embedding_dim], or None if not in cache
        """
        cached = self.table_cell_cache.get(table_id)
        if cached is not None:
            return cached.clone()
        return None
    
    def clear(self):
        """Clear the cache."""
        self.table_cache.clear()
        self.context_cache.clear()
        self.table_schema_cache.clear()
        self.table_cell_cache.clear()
    
    def stats(self) -> Dict[str, int]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'num_tables': len(self.table_cache),
            'num_contexts': len(self.context_cache),
            'num_table_schemas': len(self.table_schema_cache),
            'num_table_cells': len(self.table_cell_cache),
            'device': self.device
        }

def load_row_level_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a pre-processed row-level dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of processed examples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def _extract_rows_robust(item: dict) -> List[str]:
    """Helper to robustly extract table rows from various formats."""
    rows = []
    
    # Try generic "rows" or "anchor_rows"
    rows_list = item.get("anchor_rows", []) 
    if not rows_list and "anchor_sentences" in item:
        rows_list = item["anchor_sentences"]
    if not rows_list and "rows" in item: 
        rows_list = item["rows"]
    if not rows_list and "sentences" in item:
        rows_list = item["sentences"]
            
    if rows_list:
        headers = item.get("headers", [])
        for r in rows_list:
            if isinstance(r, str):
                rows.append(r.strip())
            elif isinstance(r, dict):
                if "formatted" in r:
                    rows.append(r["formatted"].strip())
                elif "content" in r:
                    # Generic content list [c1, c2, ...]
                    if headers and len(headers) == len(r["content"]):
                        parts = [f"{h}: {v}" for h, v in zip(headers, r["content"]) if str(v).strip()]
                        rows.append("; ".join(parts) + ".")
                    else:
                        rows.append(" | ".join([str(c) for c in r["content"]]))
            elif isinstance(r, list):
                # Raw cell list
                if headers and len(headers) == len(r):
                    parts = [f"{h}: {v}" for h, v in zip(headers, r) if str(v).strip()]
                    rows.append("; ".join(parts) + ".")
                else:
                    rows.append(" | ".join([str(c) for c in r]))
                    
    # MIMIC fallback
    if not rows and "tables" in item:
        tables_dict = item.get("tables", {})
        if isinstance(tables_dict, str):
            import ast
            try: tables_dict = ast.literal_eval(tables_dict)
            except: tables_dict = {}
        for t_name, t_data in tables_dict.items():
            for r in t_data.get("rows", []):
                 if isinstance(r, dict) and "formatted" in r:
                      rows.append(r["formatted"].strip())
                 elif isinstance(r, dict) and "content" in r:
                      rows.append(" | ".join([str(c) for c in r["content"]]))
                      
    return [r for r in rows if r]

def _extract_headers_robust(item: dict) -> List[str]:
    """Extract table headers when available."""
    headers = item.get("headers", [])
    if isinstance(headers, list):
        return [str(header).strip() for header in headers if str(header).strip()]
    if isinstance(headers, dict):
        return [str(value).strip() for value in headers.values() if str(value).strip()]
    if isinstance(headers, str):
        header = headers.strip()
        return [header] if header else []
    return []


def _normalize_text_piece(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() == "nan":
        return ""
    return text


def _format_value_only_sequence(values: List[str]) -> str:
    cleaned_values = [_normalize_text_piece(value) for value in values]
    cleaned_values = [value for value in cleaned_values if value]
    if not cleaned_values:
        return ""
    return "; ".join(cleaned_values) + "."


def _extract_structured_table_content(item: dict) -> Tuple[List[str], List[List[str]]]:
    """Extract headers plus row-aligned cell values when structured content is available."""
    headers = _extract_headers_robust(item)

    rows_list = item.get("anchor_rows", [])
    if not rows_list and "rows" in item:
        rows_list = item["rows"]
    if not rows_list and "sentences" in item and "headers" in item:
        rows_list = item["sentences"]

    if not headers and isinstance(rows_list, list):
        for row in rows_list:
            if isinstance(row, dict) and row and "content" not in row and "formatted" not in row:
                headers = [str(key).strip() for key in row.keys() if str(key).strip()]
                if headers:
                    break

    structured_rows: List[List[str]] = []

    def coerce_row_cells(raw_row: Any) -> Optional[List[str]]:
        cells: Optional[List[str]] = None

        if isinstance(raw_row, dict):
            if isinstance(raw_row.get("content"), list):
                cells = [_normalize_text_piece(cell) for cell in raw_row["content"]]
            elif headers:
                cells = [_normalize_text_piece(raw_row.get(header, "")) for header in headers]
        elif isinstance(raw_row, list):
            cells = [_normalize_text_piece(cell) for cell in raw_row]

        if cells is None:
            return None

        if headers:
            if len(cells) < len(headers):
                cells = cells + [""] * (len(headers) - len(cells))
            cells = cells[:len(headers)]

        return cells

    if isinstance(rows_list, list):
        for row in rows_list:
            coerced_cells = coerce_row_cells(row)
            if coerced_cells is not None:
                structured_rows.append(coerced_cells)

    if headers:
        return headers, structured_rows

    tables_dict = item.get("tables", {})
    if isinstance(tables_dict, str):
        import ast
        try:
            tables_dict = ast.literal_eval(tables_dict)
        except Exception:
            tables_dict = {}
    if isinstance(tables_dict, dict) and tables_dict:
        fallback_headers = [str(name).strip() for name in tables_dict.keys() if str(name).strip()]
        return fallback_headers, []

    return [], []


def _select_representative_column_values(
    structured_rows: List[List[str]],
    column_index: int,
    max_values: int = _DEFAULT_COLUMN_SKETCH_VALUES,
) -> List[str]:
    """Pick diverse non-empty values for one column using first-occurrence coverage."""
    representative_values = []
    seen_values = set()

    for row in structured_rows:
        if column_index >= len(row):
            continue
        value = _normalize_text_piece(row[column_index])
        if not value:
            continue
        normalized_key = value.lower()
        if normalized_key in seen_values:
            continue
        seen_values.add(normalized_key)
        representative_values.append(value)
        if len(representative_values) >= max_values:
            break

    return representative_values


def _build_column_sketch(header: str, representative_values: List[str]) -> str:
    header_text = _normalize_text_piece(header) or "unknown"
    if representative_values:
        return f"Column {header_text}. Example values: {'; '.join(representative_values)}."
    return f"Column {header_text}."


def _build_cell_text(header: str, value: Any) -> str:
    header_text = _normalize_text_piece(header) or "unknown"
    value_text = _normalize_text_piece(value)
    if not value_text:
        return ""
    return f"{header_text}: {value_text}"


def _extract_table_cell_texts(item: dict) -> List[List[str]]:
    """Build per-row, per-column header:value cell texts for structural matching."""
    headers, structured_rows = _extract_structured_table_content(item)
    if not headers or not structured_rows:
        return []

    cell_text_rows: List[List[str]] = []
    for row in structured_rows:
        row_cells = []
        for column_index, header in enumerate(headers):
            value = row[column_index] if column_index < len(row) else ""
            row_cells.append(_build_cell_text(header, value))
        if any(row_cells):
            cell_text_rows.append(row_cells)

    return cell_text_rows


def _normalize_schema_texts(schema_texts: Any) -> List[str]:
    if schema_texts is None:
        return []
    if isinstance(schema_texts, str):
        normalized_text = _normalize_text_piece(schema_texts)
        return [normalized_text] if normalized_text else []
    if isinstance(schema_texts, list):
        normalized_texts = [_normalize_text_piece(text) for text in schema_texts]
        return [text for text in normalized_texts if text]
    return []

def _extract_row_values_robust(item: dict) -> List[str]:
    """
    Extract table row values without prefixing column headers.

    This is used by the optional header-conditioning path so row content and
    table schema remain disentangled.
    """
    rows = []

    rows_list = item.get("anchor_rows", [])
    if not rows_list and "rows" in item:
        rows_list = item["rows"]
    if not rows_list and "sentences" in item and "headers" in item:
        rows_list = item["sentences"]

    if rows_list:
        for row in rows_list:
            if isinstance(row, str):
                rows.append(row.strip())
            elif isinstance(row, dict):
                if "content" in row:
                    values = [str(cell).strip() for cell in row["content"] if str(cell).strip()]
                    if values:
                        rows.append(_format_value_only_sequence(values))
                elif "formatted" in row:
                    rows.append(row["formatted"].strip())
            elif isinstance(row, list):
                values = [str(cell).strip() for cell in row if str(cell).strip()]
                if values:
                    rows.append(_format_value_only_sequence(values))

    if not rows and "tables" in item:
        tables_dict = item.get("tables", {})
        if isinstance(tables_dict, str):
            import ast
            try:
                tables_dict = ast.literal_eval(tables_dict)
            except Exception:
                tables_dict = {}
        for _, table_data in tables_dict.items():
            for row in table_data.get("rows", []):
                if isinstance(row, dict) and "content" in row:
                    values = [str(cell).strip() for cell in row["content"] if str(cell).strip()]
                    if values:
                        rows.append(_format_value_only_sequence(values))
                elif isinstance(row, dict) and "formatted" in row:
                    rows.append(row["formatted"].strip())

    return [row for row in rows if row]

def _extract_table_rows_for_model(item: dict, use_header_conditioning: bool = False) -> List[str]:
    """Return the table-side row texts to feed into the encoder."""
    if use_header_conditioning:
        return _extract_row_values_robust(item)
    return _extract_rows_robust(item)

def _extract_table_schema_text(item: dict) -> List[str]:
    """Build per-column schema sketches for Q/K conditioning."""
    headers, structured_rows = _extract_structured_table_content(item)
    if headers:
        return [
            _build_column_sketch(
                header,
                _select_representative_column_values(structured_rows, column_index),
            )
            for column_index, header in enumerate(headers)
        ]
    return []

def _extract_sentences_robust(sentences_data) -> List[str]:
    """Helper to robustly extract sentences from MIMIC format (dict or list)"""
    extracted_texts = []
    if isinstance(sentences_data, dict):
        try:
            # Try to sort by integer key "0", "1", etc.
            sorted_keys = sorted(sentences_data.keys(), key=lambda k: int(k))
            for k in sorted_keys:
                item = sentences_data[k]
                if isinstance(item, dict):
                    extracted_texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    extracted_texts.append(item)
        except ValueError:
            # Fallback for non-integer keys
            for item in sentences_data.values():
                if isinstance(item, dict):
                    extracted_texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    extracted_texts.append(item)
    elif isinstance(sentences_data, list):
        for item in sentences_data:
            if isinstance(item, dict):
                extracted_texts.append(item.get("text", ""))
            elif isinstance(item, str):
                extracted_texts.append(item)
    # Check for direct string list in Pharma Flipped
    elif isinstance(sentences_data, str):
        extracted_texts.append(sentences_data)

    return [t.strip() for t in extracted_texts if t]

def collect_all_ids_and_texts(examples: List[Dict[str, Any]], split_name: str = "unknown", task_direction: str = "TABLE_TO_DOC", native_direction: str = "TABLE_TO_DOC", use_header_conditioning: bool = False) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], Dict[int, List[str]], Dict[int, List[List[str]]], str]:
    """
    Collect all unique tables and documents, dynamically mapping them to 'table_pool' and 'doc_pool'.
    """
    tables_dict = {}  # canonical table_id -> [rows]
    docs_dict = {}   # canonical doc_id -> [sentences]
    table_schemas_dict = {}  # canonical table_id -> per-column schema sketches
    table_cells_dict = {}  # canonical table_id -> per-row cell texts
    
    task_direction = task_direction.upper()
    native_direction = native_direction.upper()
    
    print(f"Collecting data for {split_name} split (Task: {task_direction}, Native: {native_direction})...")
    for example in tqdm(examples, desc=f"Processing {split_name}"):
        # 1. Process Anchor
        anchor_id = example.get("anchor_id")
        if anchor_id:
            if native_direction == "TABLE_TO_DOC":
                if anchor_id not in tables_dict:
                    tables_dict[anchor_id] = _extract_table_rows_for_model(example, use_header_conditioning=use_header_conditioning)
                    table_schemas_dict[anchor_id] = _extract_table_schema_text(example)
                    table_cells_dict[anchor_id] = _extract_table_cell_texts(example)
            else: # DOC_TO_TABLE (native)
                if anchor_id not in docs_dict:
                    docs_dict[anchor_id] = _extract_sentences_robust(example.get("anchor_sentences", []))
        
        # 2. Process Positives/Negatives
        def process_doc_item(item):
            if not item or "id" not in item: return
            item_id = item["id"]
            if item_id not in docs_dict:
                docs_dict[item_id] = _extract_sentences_robust(item.get("sentences", []))
                
        def process_table_item(item):
            if not item or "id" not in item: return
            item_id = item["id"]
            if item_id not in tables_dict:
                tables_dict[item_id] = _extract_table_rows_for_model(item, use_header_conditioning=use_header_conditioning)
                table_schemas_dict[item_id] = _extract_table_schema_text(item)
                table_cells_dict[item_id] = _extract_table_cell_texts(item)

        # Extraction keys
        for key in ["primary_positive", "additional_positives", "negatives"]:
            items = example.get(key, [])
            if not isinstance(items, list): items = [items]
            for it in items:
                if native_direction == "TABLE_TO_DOC":
                    # pos/negs are Docs
                    process_doc_item(it)
                else:
                    # pos/negs are Tables
                    process_table_item(it)
    
    if native_direction == "TABLE_TO_DOC":
        print(f"Found {len(tables_dict)} unique tables and {len(docs_dict)} unique documents in {split_name}")
    else:
        print(f"Found {len(docs_dict)} unique documents and {len(tables_dict)} unique tables in {split_name}")
    return tables_dict, docs_dict, table_schemas_dict, table_cells_dict, split_name

def generate_triplet_batches_for_single_example(
    example: Dict[str, Any], 
    training_batch_size: int, 
    shuffle_triplets: bool = True,
    drop_last: bool = False,
    triplet_strategy: str = "limited",
    max_triplets_per_example: int = 10,
    task_direction: str = "TABLE_TO_DOC",
    native_direction: str = "TABLE_TO_DOC",
    use_header_conditioning: bool = False,
) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Generate triplet batches for a single example.
    
    Args:
        task_direction: What the model wants (TABLE_TO_DOC or DOC_TO_TABLE)
        native_direction: How the file is stored (TABLE_TO_DOC or DOC_TO_TABLE)
    """
    anchor_id = example.get("anchor_id")
    if anchor_id is None:
        return
    
    task_direction = task_direction.upper()
    native_direction = native_direction.upper()

    # 1. Extract Anchor Texts (Anchor is the Query)
    if native_direction == "TABLE_TO_DOC":
        anchor_texts = _extract_table_rows_for_model(example, use_header_conditioning=use_header_conditioning)
        anchor_schema_text = _extract_table_schema_text(example) if use_header_conditioning else None
        anchor_cell_texts = _extract_table_cell_texts(example)
    else: # DOC_TO_TABLE (native)
        anchor_texts = _extract_sentences_robust(example.get("anchor_sentences", []))
        anchor_schema_text = None
        anchor_cell_texts = None
    
    if not anchor_texts: return

    # 2. Extract Positives/Negatives (Candidates)
    def get_candidate_data(item):
        if not item or "id" not in item: return None, [], None, None
        iid = item["id"]
        if native_direction == "TABLE_TO_DOC":
            return iid, _extract_sentences_robust(item.get("sentences", [])), None, None
        else:
            return (
                iid,
                _extract_table_rows_for_model(item, use_header_conditioning=use_header_conditioning),
                (_extract_table_schema_text(item) if use_header_conditioning else None),
                _extract_table_cell_texts(item),
            )

    # triplets collection
    triplets = []
    
    # Collect positives
    all_positives = []
    for key in ["primary_positive", "additional_positives"]:
        val = example.get(key, [])
        items = val if isinstance(val, list) else [val]
        for it in items:
            it_id, it_texts, it_schema_text, it_cell_texts = get_candidate_data(it)
            if it_id is not None and it_texts:
                all_positives.append((it_id, it_texts, it_schema_text, it_cell_texts))
                
    # Collect negatives
    all_negatives = []
    for it in example.get("negatives", []):
        it_id, it_texts, it_schema_text, it_cell_texts = get_candidate_data(it)
        if it_id is not None and it_texts:
            all_negatives.append((it_id, it_texts, it_schema_text, it_cell_texts))

    if all_positives and all_negatives:
        # Triplet Sampling Logic
        if triplet_strategy == "primary_only":
            triplet_list = [(all_positives[0], all_negatives[0])]
        elif triplet_strategy == "limited":
            # Round-robin over positives so all positives are represented even with small budgets.
            # Each positive sees negatives sequentially from the shared pool.
            # e.g. 2 pos × 4 neg, budget=4: (p0,n0),(p1,n0),(p0,n1),(p1,n1)
            triplet_list = []
            neg_cursors = [0] * len(all_positives)
            while len(triplet_list) < max_triplets_per_example:
                made_progress = False
                for i, p in enumerate(all_positives):
                    if neg_cursors[i] < len(all_negatives):
                        triplet_list.append((p, all_negatives[neg_cursors[i]]))
                        neg_cursors[i] += 1
                        made_progress = True
                        if len(triplet_list) >= max_triplets_per_example:
                            break
                if not made_progress:
                    break
        elif triplet_strategy == "balanced":
            # Partition negatives into equal groups (one per positive, by index position).
            # Then interleave round-robin: (p0,bucket0[0]),(p1,bucket1[0]),(p0,bucket0[1]),...
            # Guarantees domain symmetry when negatives are ordered [diag_negs..., med_negs...].
            n_pos = len(all_positives)
            group_size = max(1, len(all_negatives) // n_pos)
            buckets = []
            for i in range(n_pos):
                start = i * group_size
                end = start + group_size if i < n_pos - 1 else len(all_negatives)
                buckets.append(list(all_negatives[start:end]))
            triplet_list = []
            bucket_indices = [0] * n_pos
            while len(triplet_list) < max_triplets_per_example:
                made_progress = False
                for i, (p, bucket) in enumerate(zip(all_positives, buckets)):
                    if bucket_indices[i] < len(bucket):
                        triplet_list.append((p, bucket[bucket_indices[i]]))
                        bucket_indices[i] += 1
                        made_progress = True
                        if len(triplet_list) >= max_triplets_per_example:
                            break
                if not made_progress:
                    break
        elif triplet_strategy == "random":
            primary_pos = all_positives[0]
            primary_triplets = [(primary_pos, n) for n in all_negatives]
            random.shuffle(primary_triplets)
            triplet_list = primary_triplets[:max_triplets_per_example]
            if len(triplet_list) < max_triplets_per_example and len(all_positives) > 1:
                remaining = max_triplets_per_example - len(triplet_list)
                other_pairs = [(p, n) for p in all_positives[1:] for n in all_negatives]
                if other_pairs:
                    triplet_list.extend(random.sample(other_pairs, min(remaining, len(other_pairs))))
        else: # "full"
            triplet_list = [(p, n) for p in all_positives for n in all_negatives]

        # Construct final dicts
        for (pos_id, pos_texts, pos_schema_text, pos_cell_texts), (neg_id, neg_texts, neg_schema_text, neg_cell_texts) in triplet_list:
            triplets.append({
                'anchor_id': anchor_id,
                'anchor_texts': anchor_texts,
                'anchor_schema_text': anchor_schema_text,
                'anchor_cell_texts': anchor_cell_texts,
                'positive_id': pos_id,
                'positive_texts': pos_texts,
                'positive_schema_text': pos_schema_text,
                'positive_cell_texts': pos_cell_texts,
                'negative_id': neg_id,
                'negative_texts': neg_texts,
                'negative_schema_text': neg_schema_text,
                'negative_cell_texts': neg_cell_texts,
            })
    
    if shuffle_triplets and triplets:
        random.shuffle(triplets)
    
    for i in range(0, len(triplets), training_batch_size):
        batch = triplets[i:i + training_batch_size]
        if drop_last and len(batch) < training_batch_size: continue
        yield batch

def prepare_mixed_triplet_batches(examples: List[Dict[str, Any]], batch_size: int, 
                                shuffle_triplets: bool = True, drop_last: bool = True,
                                triplet_strategy: str = "limited", max_triplets_per_example: int = 10,
                                task_direction: str = "TABLE_TO_DOC",
                                native_direction: str = "TABLE_TO_DOC",
                                use_header_conditioning: bool = False) -> List[List[Dict[str, Any]]]:
    """
    Prepare mixed batches with Task-Aware support.
    """
    all_triplets = []
    triplet_counts = []
    
    for example in tqdm(examples, desc="Collecting mixed triplets"):
        example_triplet_count = 0
        gen = generate_triplet_batches_for_single_example(
            example=example,
            training_batch_size=999999,
            shuffle_triplets=False,
            drop_last=False,
            triplet_strategy=triplet_strategy,
            max_triplets_per_example=max_triplets_per_example,
            task_direction=task_direction,
            native_direction=native_direction,
            use_header_conditioning=use_header_conditioning,
        )
        for batch in gen:
            all_triplets.extend(batch)
            example_triplet_count += len(batch)
        triplet_counts.append(example_triplet_count)
    
    # Print statistics about triplet generation
    if triplet_counts:
        import numpy as np
        print(f"\n📊 Triplet Generation Statistics:")
        print(f"   Total examples: {len(examples)}")
        print(f"   Total triplets: {len(all_triplets)}")
        print(f"   Avg triplets/example: {np.mean(triplet_counts):.2f}")
        print(f"   Min triplets/example: {np.min(triplet_counts)}")
        print(f"   Max triplets/example: {np.max(triplet_counts)}")
        print(f"   Examples with 0 triplets: {sum(1 for c in triplet_counts if c == 0)}")
    
    # Shuffle all triplets together if requested
    if shuffle_triplets and all_triplets:
        random.shuffle(all_triplets)
    
    # Create batches of the requested size
    batches = []
    for i in range(0, len(all_triplets), batch_size):
        batch = all_triplets[i:i + batch_size]
        
        # Skip incomplete batches if drop_last is True
        if drop_last and len(batch) < batch_size:
            continue
            
        batches.append(batch)
    
    dropped_triplets = len(all_triplets) - sum(len(b) for b in batches)
    if dropped_triplets > 0:
        print(f"⚠️  Dropped {dropped_triplets} triplets from incomplete batches (drop_last=True)")
    
    print(f"Generated {len(batches)} mixed batches from {len(all_triplets)} total triplets")
    return batches

def prepare_example_isolated_batches(examples: List[Dict[str, Any]], batch_size: int, 
                                 shuffle_triplets: bool = True, drop_last: bool = True,
                                 triplet_strategy: str = "limited", max_triplets_per_example: int = 10,
                                 task_direction: str = "TABLE_TO_DOC",
                                 native_direction: str = "TABLE_TO_DOC",
                                 use_header_conditioning: bool = False) -> List[List[Dict[str, Any]]]:
    """
    Prepare isolated batches with Task-Aware support.
    """
    all_batches = []
    triplet_counts = []
    for example in tqdm(examples, desc="Collecting isolated batches"):
        example_batches = list(generate_triplet_batches_for_single_example(
            example=example,
            training_batch_size=batch_size,
            shuffle_triplets=shuffle_triplets,
            drop_last=drop_last,
            triplet_strategy=triplet_strategy,
            max_triplets_per_example=max_triplets_per_example,
            task_direction=task_direction,
            native_direction=native_direction,
            use_header_conditioning=use_header_conditioning,
        ))
        all_batches.extend(example_batches)
        triplet_counts.append(sum(len(batch) for batch in example_batches))
    
    # Print statistics
    if triplet_counts:
        import numpy as np
        total_triplets = sum(triplet_counts)
        print(f"\n📊 Triplet Generation Statistics (Isolated Batching):")
        print(f"   Total examples: {len(examples)}")
        print(f"   Total triplets: {total_triplets}")
        print(f"   Avg triplets/example: {np.mean(triplet_counts):.2f}")
        print(f"   Min triplets/example: {np.min(triplet_counts)}")
        print(f"   Max triplets/example: {np.max(triplet_counts)}")
        print(f"   Examples with 0 triplets: {sum(1 for c in triplet_counts if c == 0)}")
    
    return all_batches

def prepare_triplet_batches(examples: List[Dict[str, Any]], batch_size: int, 
                          shuffle_triplets: bool = True, drop_last: bool = True,
                          mix_examples: bool = True, triplet_strategy: str = "limited",
                          max_triplets_per_example: int = 10,
                          task_direction: str = "TABLE_TO_DOC",
                          native_direction: str = "TABLE_TO_DOC",
                          use_header_conditioning: bool = False) -> List[List[Dict[str, Any]]]:
    """
    Prepare triplet batches with Task-Aware support.
    """
    print(f"Triplet Strategy: {triplet_strategy.upper()} | Task: {task_direction} | Native: {native_direction}")
    
    if mix_examples:
        return prepare_mixed_triplet_batches(examples, batch_size, shuffle_triplets, drop_last,
                                            triplet_strategy, max_triplets_per_example,
                                            task_direction, native_direction,
                                            use_header_conditioning)
    else:
        return prepare_example_isolated_batches(examples, batch_size, shuffle_triplets, drop_last,
                                               triplet_strategy, max_triplets_per_example,
                                               task_direction, native_direction,
                                               use_header_conditioning)


# ============================================================================
# MIMIC-100 DATA PREPROCESSING
# ============================================================================
# 
# The MIMIC data preprocessing functions have been moved to a standalone script
# in the annotation_pipeline folder for cleaner separation of concerns.
#
# To preprocess raw MIMIC-100 data:
#
#   cd annotation_pipeline
#   python preprocess_mimic.py --mimic_root ./mimic_100 --output_dir ./mimic_data
#
# The preprocessed JSON files can then be used with this training code:
#
#   python run_cross_attention.py \
#       --train_file mimic_data/train_row_level_v2.json \
#       --eval_file mimic_data/val_row_level_v2.json \
#       --test_file mimic_data/test_row_level_v2.json
#
# ============================================================================