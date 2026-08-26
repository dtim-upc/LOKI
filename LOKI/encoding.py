from typing import Dict, List, Tuple, Any, Optional
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

# Import from our new data module
from data import IdBasedEmbeddingCache, collect_all_ids_and_texts, _normalize_schema_texts

# Maximum number of texts to pool before encoding in one chunk.
# Prevents tokenizer OOM for very large splits. Each chunk still benefits
# from large batch_size internally via SentenceTransformer.encode().
_MAX_POOL_CHUNK_SIZE = 50_000


def _pool_and_encode(
    items_dict: Dict[int, List[str]],
    model: SentenceTransformer,
    batch_size: int,
    device: str,
    use_amp: bool,
    target_dtype: torch.dtype,
    desc: str = "Encoding",
) -> Dict[int, torch.Tensor]:
    """
    Pool texts from many items into a flat list, encode in one (or few) large
    model.encode() call(s), then scatter embeddings back per item.

    This saturates the GPU by processing hundreds/thousands of texts per
    forward pass instead of the ~3-15 texts a single item contains.

    Args:
        items_dict: Mapping of item_id -> list of text strings.
        model: The SentenceTransformer encoder.
        batch_size: batch_size passed to model.encode() for internal mini-batching.
        device: Target device ('cpu' or 'cuda').
        use_amp: Whether to wrap encode in torch.cuda.amp.autocast.
        target_dtype: Desired output dtype (e.g. torch.bfloat16).
        desc: Description for the progress bar.

    Returns:
        Dict mapping item_id -> Tensor of shape [num_texts, embedding_dim].
        Items with no texts are omitted from the result.
    """
    # 1. Flatten all texts and record boundaries
    all_texts: List[str] = []
    boundaries: List[Tuple[int, int, int]] = []  # (item_id, start, end)
    for item_id, texts in items_dict.items():
        if not texts:
            continue
        start = len(all_texts)
        all_texts.extend(texts)
        boundaries.append((item_id, start, len(all_texts)))

    if not all_texts:
        return {}

    total_texts = len(all_texts)
    result: Dict[int, torch.Tensor] = {}

    # 2. Encode in chunks if the pool is very large (memory safety)
    all_embeddings_parts: List[torch.Tensor] = []
    with torch.no_grad():
        for chunk_start in range(0, total_texts, _MAX_POOL_CHUNK_SIZE):
            chunk_end = min(chunk_start + _MAX_POOL_CHUNK_SIZE, total_texts)
            chunk_texts = all_texts[chunk_start:chunk_end]

            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    chunk_emb = model.encode(
                        chunk_texts,
                        convert_to_tensor=True,
                        show_progress_bar=(total_texts > 500),
                        device=device,
                        batch_size=batch_size,
                    )
            else:
                chunk_emb = model.encode(
                    chunk_texts,
                    convert_to_tensor=True,
                    show_progress_bar=(total_texts > 500),
                    device=device,
                    batch_size=batch_size,
                )

            if chunk_emb.dtype != target_dtype:
                chunk_emb = chunk_emb.to(dtype=target_dtype)
            all_embeddings_parts.append(chunk_emb)

    # Concatenate chunks (usually just one)
    if len(all_embeddings_parts) == 1:
        all_embeddings = all_embeddings_parts[0]
    else:
        all_embeddings = torch.cat(all_embeddings_parts, dim=0)

    # 3. Scatter back into per-item tensors
    for item_id, start, end in boundaries:
        result[item_id] = all_embeddings[start:end]

    return result


def _pool_and_encode_cell_grids(
    table_cells_dict: Dict[int, List[List[str]]],
    model: SentenceTransformer,
    batch_size: int,
    device: str,
    use_amp: bool,
    target_dtype: torch.dtype,
) -> Dict[int, torch.Tensor]:
    """
    Pool cell texts from all tables, encode once, scatter back as per-table
    [num_rows, num_cols, embedding_dim] grids.
    """
    # Flatten all cell texts across all tables, tracking position metadata
    all_texts: List[str] = []
    # (table_id, row_idx, col_idx, flat_index)
    position_map: List[Tuple[int, int, int]] = []
    # Per-table grid dimensions
    grid_dims: Dict[int, Tuple[int, int]] = {}  # table_id -> (num_rows, max_cols)

    for table_id, cell_text_rows in table_cells_dict.items():
        if not cell_text_rows:
            continue
        num_rows = len(cell_text_rows)
        max_cols = max((len(row) for row in cell_text_rows), default=0)
        if max_cols == 0:
            continue
        grid_dims[table_id] = (num_rows, max_cols)

        for row_idx, row_cells in enumerate(cell_text_rows):
            padded = list(row_cells) + [""] * max(0, max_cols - len(row_cells))
            for col_idx, cell_text in enumerate(padded[:max_cols]):
                normalized = str(cell_text).strip()
                if not normalized:
                    continue
                position_map.append((table_id, row_idx, col_idx))
                all_texts.append(normalized)

    if not all_texts:
        return {}

    # Encode all cell texts at once
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                all_emb = model.encode(
                    all_texts,
                    convert_to_tensor=True,
                    show_progress_bar=(len(all_texts) > 500),
                    device=device,
                    batch_size=batch_size,
                )
        else:
            all_emb = model.encode(
                all_texts,
                convert_to_tensor=True,
                show_progress_bar=(len(all_texts) > 500),
                device=device,
                batch_size=batch_size,
            )

    if all_emb.dtype != target_dtype:
        all_emb = all_emb.to(dtype=target_dtype)

    emb_dim = all_emb.shape[-1]

    # Scatter into per-table grids
    grids: Dict[int, torch.Tensor] = {}
    for table_id, (nr, nc) in grid_dims.items():
        grids[table_id] = torch.zeros(nr, nc, emb_dim, device=all_emb.device, dtype=target_dtype)

    for flat_idx, (table_id, row_idx, col_idx) in enumerate(position_map):
        grids[table_id][row_idx, col_idx] = all_emb[flat_idx]

    return grids


def build_id_based_embedding_cache(
    examples: List[Any],
    sentence_encoder_model: SentenceTransformer,
    batch_size: int = 32,
    device: str = 'cpu',
    split_name: str = "unknown",
    verbose: bool = True,
    use_amp: bool = True,
    super_batch_size: int = 256,  # Batch size for pooled encode() calls (GPU-optimal)
    task_direction: str = "TABLE_TO_DOC",
    native_direction: str = "TABLE_TO_DOC",
    use_header_conditioning: bool = False,
    use_cell_level_matching: bool = False,
) -> IdBasedEmbeddingCache:
    """
    Builds a cache of embeddings for all unique tables and contexts using IDs as keys.

    Uses POOLED ENCODING: all texts across items are flattened into a single list
    and encoded in one model.encode() call (which handles internal mini-batching).
    This maximizes GPU utilization compared to encoding each item individually.

    Args:
        examples: A list of example dictionaries from the dataset
        sentence_encoder_model: The frozen SentenceTransformer model
        batch_size: Legacy parameter (kept for backward compatibility)
        device: The device to move the model and data to ('cpu', 'cuda')
        split_name: Name of the dataset split (e.g., "train", "val", "test")
        verbose: Whether to print progress messages
        use_amp: Whether to use automatic mixed precision (faster on modern GPUs)
        super_batch_size: Batch size for internal mini-batching in model.encode()

    Returns:
        An IdBasedEmbeddingCache instance populated with embeddings
    """
    if verbose:
        print(f"Building ID-based embedding cache for {split_name} split...")
    sentence_encoder_model.to(device)
    sentence_encoder_model.eval()

    use_amp = use_amp and torch.cuda.is_available() and device != 'cpu'

    # Collect all unique tables and contexts with their IDs
    tables_dict, contexts_dict, table_schemas_dict, table_cells_dict, split_name = collect_all_ids_and_texts(
        examples,
        split_name,
        task_direction,
        native_direction,
        use_header_conditioning=use_header_conditioning,
    )

    if verbose:
        if native_direction == "TABLE_TO_DOC":
            print(f"Found {len(tables_dict)} unique tables and {len(contexts_dict)} unique documents in {split_name} split")
        else:
            print(f"Found {len(contexts_dict)} unique documents and {len(tables_dict)} unique tables in {split_name} split")

    id_cache = IdBasedEmbeddingCache()

    if verbose:
        print(f"Using device: {device} for embedding cache")
        if use_amp:
            print(f"⚡ Mixed precision (AMP) enabled for faster encoding")

    target_dtype = torch.bfloat16 if use_amp else torch.float32

    # ============================================================
    # POOLED ENCODING: Tables (rows + schemas + cells)
    # ============================================================
    total_table_texts = sum(len(t) for t in tables_dict.values())
    if verbose:
        if native_direction == "TABLE_TO_DOC":
            print(f"Encoding {len(tables_dict)} unique tables ({total_table_texts} total row texts) for {split_name} split...")
        else:
            print(f"Encoding {len(contexts_dict)} unique documents for {split_name} split...")

    # -- Table rows: pool all row texts across tables, encode once --
    table_embeddings = _pool_and_encode(
        items_dict=tables_dict,
        model=sentence_encoder_model,
        batch_size=super_batch_size,
        device=device,
        use_amp=use_amp,
        target_dtype=target_dtype,
        desc=f"Encoding {split_name} tables",
    )
    for table_id, emb in table_embeddings.items():
        id_cache.add_table_embeddings(table_id, emb)

    # -- Table schemas: pool all schema texts across tables, encode once --
    normalized_schemas: Dict[int, List[str]] = {}
    for table_id, raw_schemas in table_schemas_dict.items():
        normed = _normalize_schema_texts(raw_schemas)
        if normed:
            normalized_schemas[table_id] = normed

    if normalized_schemas:
        schema_embeddings = _pool_and_encode(
            items_dict=normalized_schemas,
            model=sentence_encoder_model,
            batch_size=super_batch_size,
            device=device,
            use_amp=use_amp,
            target_dtype=target_dtype,
            desc=f"Encoding {split_name} schemas",
        )
        for table_id, emb in schema_embeddings.items():
            id_cache.add_table_schema_embedding(table_id, emb)

    # -- Table cells: pool all cell texts across tables, encode once --
    if use_cell_level_matching:
        cell_grids = _pool_and_encode_cell_grids(
            table_cells_dict=table_cells_dict,
            model=sentence_encoder_model,
            batch_size=super_batch_size,
            device=device,
            use_amp=use_amp,
            target_dtype=target_dtype,
        )
        for table_id, grid in cell_grids.items():
            id_cache.add_table_cell_embeddings(table_id, grid)

    # ============================================================
    # POOLED ENCODING: Contexts (sentences)
    # ============================================================
    total_ctx_texts = sum(len(t) for t in contexts_dict.values())
    if verbose:
        if native_direction == "TABLE_TO_DOC":
            print(f"Encoding {len(contexts_dict)} unique documents ({total_ctx_texts} total sentence texts) for {split_name} split...")
        else:
            print(f"Encoding {len(contexts_dict)} unique tables for {split_name} split...")

    context_embeddings = _pool_and_encode(
        items_dict=contexts_dict,
        model=sentence_encoder_model,
        batch_size=super_batch_size,
        device=device,
        use_amp=use_amp,
        target_dtype=target_dtype,
        desc=f"Encoding {split_name} contexts",
    )
    for ctx_id, emb in context_embeddings.items():
        id_cache.add_context_embeddings(ctx_id, emb)

    if verbose:
        print(f"ID-based embedding cache built successfully for {split_name} split.")
        print(f"Cache stats: {id_cache.stats()}")
    return id_cache


# Legacy helper kept for backward compatibility (unused by main cache builder)
def _encode_cell_text_grid_with_sentence_encoder(
    sentence_encoder_model: SentenceTransformer,
    cell_text_rows: List[List[str]],
    batch_size: int,
    device: str,
    use_amp: bool,
    target_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if not cell_text_rows:
        return None

    num_rows = len(cell_text_rows)
    max_cols = max((len(row) for row in cell_text_rows), default=0)
    if max_cols == 0:
        return None

    flat_texts: List[str] = []
    flat_positions: List[Tuple[int, int]] = []
    for row_index, row_cells in enumerate(cell_text_rows):
        padded_cells = list(row_cells) + [""] * max(0, max_cols - len(row_cells))
        for col_index, cell_text in enumerate(padded_cells[:max_cols]):
            normalized_text = str(cell_text).strip()
            if not normalized_text:
                continue
            flat_texts.append(normalized_text)
            flat_positions.append((row_index, col_index))

    if not flat_texts:
        return None

    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            cell_embeddings = sentence_encoder_model.encode(
                flat_texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=device,
                batch_size=min(batch_size, len(flat_texts)),
            )
    else:
        cell_embeddings = sentence_encoder_model.encode(
            flat_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device,
            batch_size=min(batch_size, len(flat_texts)),
        )

    if cell_embeddings.dtype != target_dtype:
        cell_embeddings = cell_embeddings.to(dtype=target_dtype)

    grid = torch.zeros(num_rows, max_cols, cell_embeddings.shape[-1], device=cell_embeddings.device, dtype=target_dtype)
    for embedding_index, (row_index, col_index) in enumerate(flat_positions):
        grid[row_index, col_index] = cell_embeddings[embedding_index]

    return grid


# Keep _batched_encode for backward compatibility but it's no longer used by main function
def _batched_encode(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    device: str,
    use_amp: bool,
    desc: str = "Encoding",
    target_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Encode texts in batches. DEPRECATED: Use incremental encoding instead for memory safety.
    """
    if not texts:
        return torch.empty(0, device=device)
    
    if target_dtype is None:
        target_dtype = torch.bfloat16 if use_amp else torch.float32
    
    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, total=num_batches):
            batch_texts = texts[i:i + batch_size]
            
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    batch_embeddings = model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=device
                    )
            else:
                batch_embeddings = model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=device
                )
            
            if batch_embeddings.dtype != target_dtype:
                batch_embeddings = batch_embeddings.to(dtype=target_dtype)
            
            all_embeddings.append(batch_embeddings)
    
    return torch.cat(all_embeddings, dim=0)