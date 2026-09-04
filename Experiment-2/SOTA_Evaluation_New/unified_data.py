"""
unified_data.py

Core data abstraction module for the Task-Aware Data Ingestion Layer.
Allows reading canonical (table-as-anchor) LOKI JSONs dynamically as either
TABLE_TO_DOC or DOC_TO_TABLE tasks without physically flipping the files.

Includes content-hash-based deduplication to resolve the inflated-ID issue
where identical document/table content appears under different IDs due to
horizontal partitioning in preprocessing (affects pharma, multihiertt, etc.)
"""

import os
import json
import hashlib
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_DEFAULT_COLUMN_SKETCH_VALUES = 3

# --- Helper Text Extractors (from existing data.py / evaluate_loki.py) ---
def _extract_sentences_robust(doc_item: dict) -> list:
    """Extracts text sentences from a document item interchangeably.

    Handles three sentence storage layouts:
      - list of str / dict-with-"text"  (Protrix, Pharma, mimic_flipped anchor_sentences)
      - dict keyed by string int index, values are {"text": ...}  (MIMIC TABLE_TO_DOC docs)
      - sections list [{"sentences": [{"text": ...}]}]  (MIMIC sections fallback)
    """
    if doc_item is None:
        return []
    sentences = []
    # Priority: anchor_sentences (mimic_flipped) > sentences (all others)
    if "anchor_sentences" in doc_item:
        rows_list = doc_item["anchor_sentences"]
    elif "sentences" in doc_item:
        rows_list = doc_item["sentences"]
    else:
        rows_list = []

    if isinstance(rows_list, dict):
        # MIMIC TABLE_TO_DOC format: {"0": {"text": "..."}, "1": {"text": "..."}, ...}
        try:
            sorted_keys = sorted(rows_list.keys(), key=lambda k: int(k))
        except (ValueError, TypeError):
            sorted_keys = sorted(rows_list.keys())
        for k in sorted_keys:
            val = rows_list[k]
            text = val.get("text", "") if isinstance(val, dict) else str(val)
            if text.strip():
                sentences.append(text.strip())
    elif rows_list:
        for s in rows_list:
            text = s if isinstance(s, str) else (s.get("text", "") if isinstance(s, dict) else "")
            if text.strip():
                sentences.append(text.strip())
    elif "sections" in doc_item:
        # MIMIC sections fallback
        for sec in doc_item["sections"]:
            for sentence_info in sec.get("sentences", []):
                text = sentence_info.get("text", "") if isinstance(sentence_info, dict) else (str(sentence_info) if isinstance(sentence_info, str) else "")
                if text.strip():
                    sentences.append(text.strip())
    return sentences

def _extract_rows_robust(item: dict, dataset_format: str) -> list:
    """Extracts table rows correctly based on format."""
    rows = []
    
    # Strategy: Unified Check. Look for "rows" list and extract "formatted" field.
    # This works for Protrix, Mimic (inner table), and Flipped Pharma.
    rows_list = item.get("anchor_rows", []) 
    if not rows_list and "rows" in item: 
        rows_list = item["rows"]
    if not rows_list and "sentences" in item:
        # some legacy formats used "sentences" for flattened row text
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
                    
    # MIMIC fallback (multiple tables under "tables" key)
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


def _extract_headers_robust(item: dict) -> list:
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


def _normalize_text_piece(value) -> str:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() == "nan":
        return ""
    return text


def _format_value_only_sequence(values: list) -> str:
    cleaned_values = [_normalize_text_piece(value) for value in values]
    cleaned_values = [value for value in cleaned_values if value]
    if not cleaned_values:
        return ""
    return "; ".join(cleaned_values) + "."


def _extract_structured_table_content(item: dict) -> tuple:
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

    structured_rows = []

    def coerce_row_cells(raw_row):
        cells = None

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
    structured_rows: list,
    column_index: int,
    max_values: int = _DEFAULT_COLUMN_SKETCH_VALUES,
) -> list:
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


def _build_column_sketch(header: str, representative_values: list) -> str:
    header_text = _normalize_text_piece(header) or "unknown"
    if representative_values:
        return f"Column {header_text}. Example values: {'; '.join(representative_values)}."
    return f"Column {header_text}."


def _build_cell_text(header: str, value) -> str:
    header_text = _normalize_text_piece(header) or "unknown"
    value_text = _normalize_text_piece(value)
    if not value_text:
        return ""
    return f"{header_text}: {value_text}"


def _normalize_schema_texts(schema_texts) -> list:
    if schema_texts is None:
        return []
    if isinstance(schema_texts, str):
        normalized_text = _normalize_text_piece(schema_texts)
        return [normalized_text] if normalized_text else []
    if isinstance(schema_texts, list):
        normalized_texts = [_normalize_text_piece(text) for text in schema_texts]
        return [text for text in normalized_texts if text]
    return []


def _extract_table_cell_texts(item: dict) -> list:
    """Build per-row, per-column header:value cell texts for structural matching."""
    headers, structured_rows = _extract_structured_table_content(item)
    if not headers or not structured_rows:
        return []

    cell_text_rows = []
    for row in structured_rows:
        row_cells = []
        for column_index, header in enumerate(headers):
            value = row[column_index] if column_index < len(row) else ""
            row_cells.append(_build_cell_text(header, value))
        if any(row_cells):
            cell_text_rows.append(row_cells)

    return cell_text_rows


def _extract_row_values_robust(item: dict) -> list:
    """Extract table rows without column-name prefixes."""
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


def _extract_table_rows_for_model(item: dict, dataset_format: str = "other", use_header_conditioning: bool = False) -> list:
    """Return the table-side row texts expected by the Rewind checkpoint."""
    if use_header_conditioning:
        return _extract_row_values_robust(item)
    return _extract_rows_robust(item, dataset_format)


def _extract_table_schema_text(item: dict) -> list:
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


# ===========================================================================
# Content-hash Deduplication Engine
# ===========================================================================

def _content_hash(text_list: list) -> str:
    """Compute SHA-256 hash of a list of text strings (rows or sentences)."""
    combined = "\n".join(str(t) for t in text_list)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _deduplicate_by_content(id_to_data: dict) -> tuple:
    """
    Deduplicate a mapping {id -> [text_list]} by content hash.
    
    Returns:
        deduped_data: {canonical_id -> [text_list]} -- only unique content entries
        id_remap:     {original_id -> canonical_id} -- maps every ID to its canon
    """
    hash_to_canonical_id = {}   # content_hash -> first-seen id (canonical)
    id_remap = {}               # every id -> canonical id
    deduped_data = {}           # canonical_id -> data
    
    for item_id, data in id_to_data.items():
        h = _content_hash(data)
        if h not in hash_to_canonical_id:
            # First time seeing this content -- this ID becomes canonical
            hash_to_canonical_id[h] = item_id
            deduped_data[item_id] = data
            id_remap[item_id] = item_id
        else:
            # Duplicate content -- remap to the canonical ID
            canonical = hash_to_canonical_id[h]
            id_remap[item_id] = canonical
    
    return deduped_data, id_remap


def _deduplicate_tables_with_schema(id_to_rows: dict, id_to_schema: dict) -> tuple:
    """
    Deduplicate tables by the combination of schema text and row values.

    Using row values alone would collapse distinct schemas that share identical
    cell content, which breaks schema-aware evaluation.
    """
    hash_to_canonical_id = {}
    id_remap = {}
    deduped_rows = {}
    deduped_schemas = {}

    for table_id, row_values in id_to_rows.items():
        schema_texts = _normalize_schema_texts(id_to_schema.get(table_id, []))
        combined = ["__schema__"] + schema_texts + list(row_values)
        content_hash = _content_hash(combined)

        if content_hash not in hash_to_canonical_id:
            hash_to_canonical_id[content_hash] = table_id
            deduped_rows[table_id] = row_values
            deduped_schemas[table_id] = schema_texts
            id_remap[table_id] = table_id
        else:
            canonical = hash_to_canonical_id[content_hash]
            id_remap[table_id] = canonical

    return deduped_rows, deduped_schemas, id_remap


def _flatten_cell_text_rows(cell_text_rows: list) -> list:
    flattened = []
    if not isinstance(cell_text_rows, list):
        return flattened
    for row_cells in cell_text_rows:
        flattened.append("__row__")
        if isinstance(row_cells, list):
            for cell_text in row_cells:
                normalized_text = _normalize_text_piece(cell_text)
                if normalized_text:
                    flattened.append(normalized_text)
    return flattened


def _deduplicate_tables_with_structure(id_to_rows: dict, id_to_schema: dict, id_to_cells: dict) -> tuple:
    """Deduplicate tables by the exact structured view used by Rewind evaluation."""
    hash_to_canonical_id = {}
    id_remap = {}
    deduped_rows = {}
    deduped_schemas = {}
    deduped_cells = {}

    for table_id, row_texts in id_to_rows.items():
        schema_texts = _normalize_schema_texts(id_to_schema.get(table_id, []))
        cell_text_rows = id_to_cells.get(table_id, [])
        combined = ["__rows__"] + list(row_texts) + ["__schema__"] + schema_texts + ["__cells__"] + _flatten_cell_text_rows(cell_text_rows)
        content_hash = _content_hash(combined)

        if content_hash not in hash_to_canonical_id:
            hash_to_canonical_id[content_hash] = table_id
            deduped_rows[table_id] = row_texts
            deduped_schemas[table_id] = schema_texts
            deduped_cells[table_id] = cell_text_rows
            id_remap[table_id] = table_id
        else:
            canonical = hash_to_canonical_id[content_hash]
            id_remap[table_id] = canonical

    return deduped_rows, deduped_schemas, deduped_cells, id_remap


# --- Dynamic Inversion Engine ---

def build_inverted_mapping(examples: list, dataset_format: str = "protrix", native_direction: str = "TABLE_TO_DOC"):
    """
    O(N) Scan: Inverts JSONs based on native direction to get requested direction.
    
    If native_direction="TABLE_TO_DOC", inverts to treat Docs as anchors.
    If native_direction="DOC_TO_TABLE", inverts to treat Tables as anchors.
    """
    raw_query_to_positives = defaultdict(set)
    raw_all_tables_data = {}
    raw_all_docs_data = {}
    
    native_direction = native_direction.upper()
    
    for ex in examples:
        # 1. Extract Anchor Data
        anchor_id = ex["anchor_id"]
        if native_direction == "TABLE_TO_DOC":
            if anchor_id not in raw_all_tables_data:
                raw_all_tables_data[anchor_id] = _extract_rows_robust(ex, dataset_format)
        else: # DOC_TO_TABLE
            if anchor_id not in raw_all_docs_data:
                raw_all_docs_data[anchor_id] = _extract_sentences_robust(ex)
            
        # 2. Extract Positives/Negatives
        def process_item(item, is_positive):
            if not item or "id" not in item: return
            item_id = item["id"]
            
            # Map native positives/negatives to their pools
            if native_direction == "TABLE_TO_DOC":
                # Positives are docs
                raw_query_to_positives[item_id].add(anchor_id) # Result: doc -> set(tables)
                if item_id not in raw_all_docs_data:
                    raw_all_docs_data[item_id] = _extract_sentences_robust(item)
            else: # DOC_TO_TABLE
                # Positives are tables
                raw_query_to_positives[item_id].add(anchor_id) # Result: table -> set(docs)
                if item_id not in raw_all_tables_data:
                    raw_all_tables_data[item_id] = _extract_rows_robust(item, dataset_format)
                    
        def process_neg(item):
            if not item or "id" not in item: return
            item_id = item["id"]
            if native_direction == "TABLE_TO_DOC":
                if item_id not in raw_all_docs_data:
                    raw_all_docs_data[item_id] = _extract_sentences_robust(item)
            else:
                if item_id not in raw_all_tables_data:
                    raw_all_tables_data[item_id] = _extract_rows_robust(item, dataset_format)

        # Extraction logic
        for key in ["primary_positive", "additional_positives"]:
            items = ex.get(key, [])
            if not isinstance(items, list): items = [items]
            for item in items:
                # Handle stringified items from legacy pipelines
                if isinstance(item, str):
                    import ast
                    try: item = ast.literal_eval(item)
                    except: continue
                process_item(item, is_positive=True)
        
        negs = ex.get("negatives", [])
        if not isinstance(negs, list): negs = [negs]
        for item in negs:
            if isinstance(item, str):
                import ast
                try: item = ast.literal_eval(item)
                except: continue
            process_neg(item)
    
    # Deduplication
    all_docs_data, doc_remap = _deduplicate_by_content(raw_all_docs_data)
    all_tables_data, table_remap = _deduplicate_by_content(raw_all_tables_data)
    
    # Rebuild mapping with canonical IDs
    gt_map = defaultdict(set)
    for raw_q_id, raw_p_ids in raw_query_to_positives.items():
        q_remap = doc_remap if native_direction == "TABLE_TO_DOC" else table_remap
        p_remap = table_remap if native_direction == "TABLE_TO_DOC" else doc_remap
        
        canon_q = q_remap.get(raw_q_id, raw_q_id)
        for p_id in raw_p_ids:
            gt_map[canon_q].add(p_remap.get(p_id, p_id))

    return gt_map, all_tables_data, all_docs_data

# --- Unified Extraction Layer ---

def extract_tables_and_docs_unified(examples: list, task: str = "DOC_TO_TABLE", dataset_format: str = "protrix", native_direction: str = "TABLE_TO_DOC"):
    """
    Dynamically extracts tables, docs, and mapping based on task vs native direction.
    
    Args:
        task:             What the model wants (TABLE_TO_DOC or DOC_TO_TABLE)
        native_direction: How the file is stored (TABLE_TO_DOC or DOC_TO_TABLE)
    """
    task = task.upper()
    native_direction = native_direction.upper()
    
    # Scenario A: Task matches Native -> Direct parsing with deduplication
    if task == native_direction:
        logger.info(f"Extracting task={task} natively (direction matches)...")
        raw_tables_dict = {}
        raw_docs_dict = {}
        raw_gt_map = defaultdict(list)
        
        for ex in examples:
            anchor_id = ex["anchor_id"]
            if task == "TABLE_TO_DOC":
                if anchor_id not in raw_tables_dict:
                    raw_tables_dict[anchor_id] = _extract_rows_robust(ex, dataset_format)
            else:
                if anchor_id not in raw_docs_dict:
                    raw_docs_dict[anchor_id] = _extract_sentences_robust(ex)
            
            def process_item(item, is_positive):
                if not item or "id" not in item: return
                item_id = item["id"]
                if task == "TABLE_TO_DOC":
                    if item_id not in raw_docs_dict: raw_docs_dict[item_id] = _extract_sentences_robust(item)
                    if is_positive: raw_gt_map[anchor_id].append(item_id)
                else: # DOC_TO_TABLE
                    if item_id not in raw_tables_dict: raw_tables_dict[item_id] = _extract_rows_robust(item, dataset_format)
                    if is_positive: raw_gt_map[anchor_id].append(item_id)

            # Unified parsing for primaries, additionals, negatives
            for key, is_pos in [("primary_positive", True), ("additional_positives", True), ("negatives", False)]:
                items = ex.get(key, [])
                if not isinstance(items, list): items = [items]
                for it in items:
                    if isinstance(it, str):
                        import ast
                        try: it = ast.literal_eval(it)
                        except: continue
                    process_item(it, is_pos)
        
        # Deduplication
        docs_dict, doc_remap = _deduplicate_by_content(raw_docs_dict)
        tables_dict, table_remap = _deduplicate_by_content(raw_tables_dict)
        
        gt_map = {}
        for q_id, p_ids in raw_gt_map.items():
            q_remap = table_remap if task == "TABLE_TO_DOC" else doc_remap
            p_remap = doc_remap if task == "TABLE_TO_DOC" else table_remap
            
            canon_q = q_remap.get(q_id, q_id)
            if canon_q not in gt_map: gt_map[canon_q] = set()
            for pid in p_ids: gt_map[canon_q].add(p_remap.get(pid, pid))
        
        # Convert set to list for output consistency
        gt_map = {k: list(v) for k, v in gt_map.items()}

    # Scenario B: Task differs from Native -> Perform Inversion
    else:
        logger.info(f"Extracting task={task}: Inverting {native_direction} dynamically...")
        gt_map, all_tables_data, all_docs_data = build_inverted_mapping(examples, dataset_format, native_direction)
        
        # Convert set to list
        gt_map = {k: list(v) for k, v in gt_map.items()}
        tables_dict = all_tables_data
        docs_dict = all_docs_data

    logger.info(f"Yielded | Queries: {len(gt_map)} | Docs: {len(docs_dict)} | Tables: {len(tables_dict)}")
    return tables_dict, docs_dict, gt_map


def build_inverted_mapping_schema_aware(examples: list, native_direction: str = "TABLE_TO_DOC"):
    """
    Invert examples while preserving schema-aware table serialization.

    Returns:
        gt_map, all_tables_data, all_docs_data, all_table_schemas
    """
    gt_map, all_tables_data, all_docs_data, all_table_schemas, _all_table_cells = build_inverted_mapping_structured(
        examples,
        native_direction=native_direction,
        use_header_conditioning=True,
        use_cell_level_matching=False,
    )
    return gt_map, all_tables_data, all_docs_data, all_table_schemas


def build_inverted_mapping_structured(
    examples: list,
    dataset_format: str = "other",
    native_direction: str = "TABLE_TO_DOC",
    use_header_conditioning: bool = False,
    use_cell_level_matching: bool = False,
):
    """Invert examples while preserving the structured view expected by Rewind checkpoints."""
    raw_query_to_positives = defaultdict(set)
    raw_all_tables_data = {}
    raw_all_docs_data = {}
    raw_all_table_schemas = {}
    raw_all_table_cells = {}

    native_direction = native_direction.upper()

    for ex in examples:
        anchor_id = ex["anchor_id"]
        if native_direction == "TABLE_TO_DOC":
            if anchor_id not in raw_all_tables_data:
                raw_all_tables_data[anchor_id] = _extract_table_rows_for_model(
                    ex,
                    dataset_format=dataset_format,
                    use_header_conditioning=use_header_conditioning,
                )
                raw_all_table_schemas[anchor_id] = _extract_table_schema_text(ex) if use_header_conditioning else []
                raw_all_table_cells[anchor_id] = _extract_table_cell_texts(ex) if use_cell_level_matching else []
        else:
            if anchor_id not in raw_all_docs_data:
                raw_all_docs_data[anchor_id] = _extract_sentences_robust(ex)

        def process_item(item):
            if not item or "id" not in item:
                return
            item_id = item["id"]

            if native_direction == "TABLE_TO_DOC":
                raw_query_to_positives[item_id].add(anchor_id)
                if item_id not in raw_all_docs_data:
                    raw_all_docs_data[item_id] = _extract_sentences_robust(item)
            else:
                raw_query_to_positives[item_id].add(anchor_id)
                if item_id not in raw_all_tables_data:
                    raw_all_tables_data[item_id] = _extract_table_rows_for_model(
                        item,
                        dataset_format=dataset_format,
                        use_header_conditioning=use_header_conditioning,
                    )
                    raw_all_table_schemas[item_id] = _extract_table_schema_text(item) if use_header_conditioning else []
                    raw_all_table_cells[item_id] = _extract_table_cell_texts(item) if use_cell_level_matching else []

        def process_neg(item):
            if not item or "id" not in item:
                return
            item_id = item["id"]
            if native_direction == "TABLE_TO_DOC":
                if item_id not in raw_all_docs_data:
                    raw_all_docs_data[item_id] = _extract_sentences_robust(item)
            else:
                if item_id not in raw_all_tables_data:
                    raw_all_tables_data[item_id] = _extract_table_rows_for_model(
                        item,
                        dataset_format=dataset_format,
                        use_header_conditioning=use_header_conditioning,
                    )
                    raw_all_table_schemas[item_id] = _extract_table_schema_text(item) if use_header_conditioning else []
                    raw_all_table_cells[item_id] = _extract_table_cell_texts(item) if use_cell_level_matching else []

        for key in ["primary_positive", "additional_positives"]:
            items = ex.get(key, [])
            if not isinstance(items, list):
                items = [items]
            for item in items:
                if isinstance(item, str):
                    import ast
                    try:
                        item = ast.literal_eval(item)
                    except Exception:
                        continue
                process_item(item)

        negs = ex.get("negatives", [])
        if not isinstance(negs, list):
            negs = [negs]
        for item in negs:
            if isinstance(item, str):
                import ast
                try:
                    item = ast.literal_eval(item)
                except Exception:
                    continue
            process_neg(item)

    all_docs_data, doc_remap = _deduplicate_by_content(raw_all_docs_data)
    all_tables_data, all_table_schemas, all_table_cells, table_remap = _deduplicate_tables_with_structure(
        raw_all_tables_data,
        raw_all_table_schemas,
        raw_all_table_cells,
    )

    gt_map = defaultdict(set)
    for raw_q_id, raw_p_ids in raw_query_to_positives.items():
        q_remap = doc_remap if native_direction == "TABLE_TO_DOC" else table_remap
        p_remap = table_remap if native_direction == "TABLE_TO_DOC" else doc_remap

        canon_q = q_remap.get(raw_q_id, raw_q_id)
        for p_id in raw_p_ids:
            gt_map[canon_q].add(p_remap.get(p_id, p_id))

    return gt_map, all_tables_data, all_docs_data, all_table_schemas, all_table_cells


def extract_tables_docs_and_schemas_unified(
    examples: list,
    task: str = "DOC_TO_TABLE",
    dataset_format: str = "protrix",
    native_direction: str = "TABLE_TO_DOC",
):
    """
    Extract tables, docs, GT map, and table schema text for schema-aware LOKI.

    Table rows are serialized as values-only strings and table deduplication uses
    schema+values together so distinct schemas are not merged accidentally.
    """
    tables_dict, docs_dict, gt_map, table_schemas_dict, _table_cells_dict = extract_tables_docs_and_structures_unified(
        examples,
        task=task,
        dataset_format=dataset_format,
        native_direction=native_direction,
        use_header_conditioning=True,
        use_cell_level_matching=False,
    )
    return tables_dict, docs_dict, gt_map, table_schemas_dict


def extract_tables_docs_and_structures_unified(
    examples: list,
    task: str = "DOC_TO_TABLE",
    dataset_format: str = "protrix",
    native_direction: str = "TABLE_TO_DOC",
    use_header_conditioning: bool = False,
    use_cell_level_matching: bool = False,
):
    """Extract tables, docs, GT map, schemas, and cell texts for Rewind checkpoints."""
    task = task.upper()
    native_direction = native_direction.upper()

    if task == native_direction:
        logger.info(f"Extracting structured task={task} natively (direction matches)...")
        raw_tables_dict = {}
        raw_docs_dict = {}
        raw_table_schemas = {}
        raw_table_cells = {}
        raw_gt_map = defaultdict(list)

        for ex in examples:
            anchor_id = ex["anchor_id"]
            if task == "TABLE_TO_DOC":
                if anchor_id not in raw_tables_dict:
                    raw_tables_dict[anchor_id] = _extract_table_rows_for_model(
                        ex,
                        dataset_format=dataset_format,
                        use_header_conditioning=use_header_conditioning,
                    )
                    raw_table_schemas[anchor_id] = _extract_table_schema_text(ex) if use_header_conditioning else []
                    raw_table_cells[anchor_id] = _extract_table_cell_texts(ex) if use_cell_level_matching else []
            else:
                if anchor_id not in raw_docs_dict:
                    raw_docs_dict[anchor_id] = _extract_sentences_robust(ex)

            def process_item(item, is_positive):
                if not item or "id" not in item:
                    return
                item_id = item["id"]
                if task == "TABLE_TO_DOC":
                    if item_id not in raw_docs_dict:
                        raw_docs_dict[item_id] = _extract_sentences_robust(item)
                    if is_positive:
                        raw_gt_map[anchor_id].append(item_id)
                else:
                    if item_id not in raw_tables_dict:
                        raw_tables_dict[item_id] = _extract_table_rows_for_model(
                            item,
                            dataset_format=dataset_format,
                            use_header_conditioning=use_header_conditioning,
                        )
                        raw_table_schemas[item_id] = _extract_table_schema_text(item) if use_header_conditioning else []
                        raw_table_cells[item_id] = _extract_table_cell_texts(item) if use_cell_level_matching else []
                    if is_positive:
                        raw_gt_map[anchor_id].append(item_id)

            for key, is_pos in [("primary_positive", True), ("additional_positives", True), ("negatives", False)]:
                items = ex.get(key, [])
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    if isinstance(item, str):
                        import ast
                        try:
                            item = ast.literal_eval(item)
                        except Exception:
                            continue
                    process_item(item, is_pos)

        docs_dict, doc_remap = _deduplicate_by_content(raw_docs_dict)
        tables_dict, table_schemas_dict, table_cells_dict, table_remap = _deduplicate_tables_with_structure(
            raw_tables_dict,
            raw_table_schemas,
            raw_table_cells,
        )

        gt_map = {}
        for q_id, p_ids in raw_gt_map.items():
            q_remap = table_remap if task == "TABLE_TO_DOC" else doc_remap
            p_remap = doc_remap if task == "TABLE_TO_DOC" else table_remap

            canon_q = q_remap.get(q_id, q_id)
            if canon_q not in gt_map:
                gt_map[canon_q] = set()
            for pid in p_ids:
                gt_map[canon_q].add(p_remap.get(pid, pid))

        gt_map = {k: list(v) for k, v in gt_map.items()}
    else:
        logger.info(f"Extracting structured task={task}: Inverting {native_direction} dynamically...")
        gt_map, tables_dict, docs_dict, table_schemas_dict, table_cells_dict = build_inverted_mapping_structured(
            examples,
            dataset_format=dataset_format,
            native_direction=native_direction,
            use_header_conditioning=use_header_conditioning,
            use_cell_level_matching=use_cell_level_matching,
        )
        gt_map = {k: list(v) for k, v in gt_map.items()}

    logger.info(
        f"Yielded structured view | Queries: {len(gt_map)} | Docs: {len(docs_dict)} | Tables: {len(tables_dict)} | Schemas: {len(table_schemas_dict)} | Cells: {len(table_cells_dict)}"
    )
    return tables_dict, docs_dict, gt_map, table_schemas_dict, table_cells_dict


def subsample_queries(gt_map: dict, max_queries: int, seed: int = 42) -> dict:
    """Subsample queries while keeping full candidate pool."""
    if max_queries <= 0 or len(gt_map) <= max_queries:
        return gt_map
    import random
    rng = random.Random(seed)
    all_query_ids = sorted(gt_map.keys(), key=str)
    selected = rng.sample(all_query_ids, max_queries)
    subsampled = {qid: gt_map[qid] for qid in selected}
    logger.info(f"Query subsampling: {len(gt_map)} -> {len(subsampled)} (seed={seed})")
    return subsampled


class UnifiedDataView:
    """Wrapper for dataset view."""
    def __init__(self, data_path: str, dataset_format: str = "protrix", task: str = "DOC_TO_TABLE", native_direction: str = "TABLE_TO_DOC"):
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_examples = json.load(f)
        self.dataset_format = dataset_format
        self.task = task.upper()
        self.native_direction = native_direction.upper()
        
    def get_tables_and_docs(self):
        return extract_tables_and_docs_unified(self.raw_examples, self.task, self.dataset_format, self.native_direction)


# ---------------------------------------------------------------------------
# Structured table extraction (used only by TaBERT evaluator)
# ---------------------------------------------------------------------------

def _extract_structured_data(item: dict):
    """Extract headers + content arrays from a table item.

    Returns ``{"headers": [...], "rows": [[cell, ...], ...]}``
    or ``None`` if the item doesn't carry structured data.
    """
    headers = item.get("headers")
    rows_list = item.get("rows", [])
    if not rows_list and "anchor_rows" in item:
        rows_list = item["anchor_rows"]

    if not headers or not rows_list:
        return None

    content_rows = []
    for r in rows_list:
        if isinstance(r, dict) and "content" in r:
            content_rows.append([str(c) for c in r["content"]])
        elif isinstance(r, list):
            content_rows.append([str(c) for c in r])

    if not content_rows:
        return None
    return {"headers": list(headers), "rows": content_rows}


def extract_structured_tables(examples: list, task: str = "DOC_TO_TABLE",
                              native_direction: str = "TABLE_TO_DOC"):
    """Return structured table data keyed by table ID.

    This scans the same examples used by ``extract_tables_and_docs_unified``
    and collects ``{tid: {"headers": [...], "rows": [[cell, ...], ...]}}``
    for every table item that carries structured (headers + content) data.

    Called only by ``evaluate_tabert.py`` so that TaBERT can build proper
    multi-column ``Table`` objects instead of parsing formatted strings.
    """
    task = task.upper()
    native_direction = native_direction.upper()

    structured = {}

    for ex in examples:
        anchor_id = ex.get("anchor_id")

        # Determine which items are tables based on task direction
        if task == native_direction:
            if task == "TABLE_TO_DOC":
                # anchor is a table
                if anchor_id not in structured:
                    s = _extract_structured_data(ex)
                    if s:
                        structured[anchor_id] = s
            else:
                # DOC_TO_TABLE: positives/negatives are tables
                pass

            for key in ("primary_positive", "additional_positives", "negatives"):
                items = ex.get(key, [])
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    if not isinstance(item, dict) or "id" not in item:
                        continue
                    item_id = item["id"]
                    if task == "TABLE_TO_DOC":
                        # items are docs (not tables) — skip
                        pass
                    else:
                        # DOC_TO_TABLE: items are tables
                        if item_id not in structured:
                            s = _extract_structured_data(item)
                            if s:
                                structured[item_id] = s
        else:
            # Inverted scenario — anchor role is swapped
            if native_direction == "TABLE_TO_DOC":
                # anchor was table natively, but task is DOC_TO_TABLE
                # so anchor becomes a doc, pos/neg become tables via inversion
                # But structured data for tables comes from the original anchor
                if anchor_id not in structured:
                    s = _extract_structured_data(ex)
                    if s:
                        structured[anchor_id] = s
            else:
                # native DOC_TO_TABLE, task TABLE_TO_DOC
                # pos/neg were tables natively
                for key in ("primary_positive", "additional_positives", "negatives"):
                    items = ex.get(key, [])
                    if not isinstance(items, list):
                        items = [items]
                    for item in items:
                        if not isinstance(item, dict) or "id" not in item:
                            continue
                        item_id = item["id"]
                        if item_id not in structured:
                            s = _extract_structured_data(item)
                            if s:
                                structured[item_id] = s

    logger.info(f"Extracted structured table data for {len(structured)} tables")
    return structured
