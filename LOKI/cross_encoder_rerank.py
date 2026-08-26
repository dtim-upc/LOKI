"""
cross_encoder_rerank.py
=======================

Pluggable zero-shot cross-encoder reranker for LOKI.

Used as Phase D.5 in materialize_joins.py — between LOKI atomic-link extraction
and clustering / labeling.  Re-scores (query, passage) pairs where the query is
the (diagnosis + medication) text and the passage is the candidate evidence
sentence.  LOKI provides recall; the cross-encoder provides precision.

Zero-shot only — annotations are never used for fine-tuning.

Default backend: cross-encoder/ettin-reranker-400m-v1 via sentence-transformers.
Any HF cross-encoder works (BGE-reranker, MS-MARCO MiniLM, etc.) by changing
the --cross_encoder_model flag at the call site.

Quick check
-----------
    conda activate THOR
    cd f:\\#LOKI_JOIN\\LOKI
    python cross_encoder_rerank.py --smoke_test
    python cross_encoder_rerank.py --smoke_test --model BAAI/bge-reranker-base
"""

from __future__ import annotations

import argparse
import hashlib
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch

from hf_model_resolver import load_hf_model_with_cache_fallback


DEFAULT_CROSS_ENCODER_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"


# ---------------------------------------------------------------------------
# Protocol — anything implementing this is a drop-in reranker
# ---------------------------------------------------------------------------

@runtime_checkable
class CrossEncoderReranker(Protocol):
    """Drop-in interface for any reranker backend."""

    name: str

    def score(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return one relevance score per (query, passage) pair, shape [N]."""
        ...


# ---------------------------------------------------------------------------
# HuggingFace / sentence-transformers backend
# ---------------------------------------------------------------------------

class HFCrossEncoder:
    """sentence-transformers.CrossEncoder backend.

    Works zero-shot with ettin-reranker, BGE-reranker, MS-MARCO MiniLM,
    Jina reranker, etc.  Output is sigmoid-normalized to [0, 1] by default so
    thresholds are portable across models.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
        fp16: bool = True,
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for HFCrossEncoder. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self.resolved_model_name, self.model_source = load_hf_model_with_cache_fallback(
            CrossEncoder,
            model_name,
            max_length=max_length,
            device=self.device,
        )

        # fp16 cuts memory ~2x with negligible quality loss for reranking
        if fp16 and str(self.device).startswith("cuda"):
            try:
                self._model.model.half()
            except Exception as exc:
                print(f"  [HFCrossEncoder] fp16 conversion skipped: {exc}")

    def score(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)

        raw = self._model.predict(
            list(pairs),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)

        if self.normalize:
            # Sigmoid so thresholds work the same across ettin / BGE / MS-MARCO
            raw = 1.0 / (1.0 + np.exp(-raw))
        return raw


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY = {
    "hf": HFCrossEncoder,
}


def build_reranker(
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    backend: str = "hf",
    **kwargs,
) -> CrossEncoderReranker:
    """Factory.  Add new backends (API rerankers, ColBERT, etc.) by registering
    them in _BACKEND_REGISTRY."""
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown cross-encoder backend '{backend}'. "
            f"Available: {sorted(_BACKEND_REGISTRY)}"
        )
    cls = _BACKEND_REGISTRY[backend]
    return cls(model_name=model_name, **kwargs)


def cache_key(model_name: str, query: str, passage: str) -> str:
    """Stable key for an on-disk score cache (Phase 6)."""
    h = hashlib.sha1(
        f"{model_name}\x00{query}\x00{passage}".encode("utf-8")
    ).hexdigest()
    return h[:32]


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

_SMOKE_PAIRS: List[Tuple[str, str]] = [
    # Strong treatment evidence — should score HIGH
    ("Diagnosis: type 2 diabetes mellitus | Medication: metformin 500 mg",
     "Patient continued on metformin for blood glucose control."),
    ("Diagnosis: hypertension | Medication: lisinopril 10 mg",
     "Started lisinopril for blood pressure management."),
    ("Diagnosis: depression | Medication: sertraline",
     "Sertraline was started as treatment for major depressive disorder."),
    ("Diagnosis: heart failure | Medication: furosemide",
     "Furosemide given for diuresis with good response."),
    ("Diagnosis: anemia | Medication: iron sulfate",
     "Iron sulfate prescribed for iron-deficiency anemia."),
    ("Diagnosis: COPD | Medication: albuterol nebulizer",
     "Albuterol nebulizers given q4h for wheezing."),
    # Discontinuation / adverse-effect evidence — should also score HIGH (still relevant)
    ("Diagnosis: atrial fibrillation | Medication: warfarin",
     "Patient developed bleeding; warfarin was held and discontinued."),
    ("Diagnosis: gout flare | Medication: prednisone",
     "Could not tolerate prednisone; discontinued due to hallucinations."),
    # Unrelated — should score LOW
    ("Diagnosis: pneumonia | Medication: aspirin 81 mg",
     "Patient eats a high-fiber breakfast every morning."),
    ("Diagnosis: insomnia | Medication: zolpidem",
     "The patient owns three cats and a dog at home."),
]


def _smoke_test(model_name: str, device: Optional[str]) -> None:
    print(f"\n  Loading cross-encoder: {model_name}")
    reranker = build_reranker(model_name=model_name, device=device)
    print(f"  Loaded on device: {reranker.device}")

    scores = reranker.score(_SMOKE_PAIRS, batch_size=8)

    print(f"\n  Scores (sigmoid-normalized to [0, 1]):\n")
    print(f"  {'idx':>3}  {'score':>7}  {'expect':>8}  query / passage")
    print(f"  {'---':>3}  {'-----':>7}  {'------':>8}  ----------------------------------")
    for i, ((q, p), s) in enumerate(zip(_SMOKE_PAIRS, scores)):
        expect = "LOW" if i >= 8 else "HIGH"
        print(f"  {i:>3}  {s:>7.4f}  {expect:>8}  Q: {q}")
        print(f"  {'':>3}  {'':>7}  {'':>8}  P: {p}")

    high = scores[:8]
    low = scores[8:]
    print(
        f"\n  Summary: mean(HIGH)={high.mean():.4f}  "
        f"mean(LOW)={low.mean():.4f}  "
        f"gap={(high.mean() - low.mean()):.4f}"
    )
    if high.mean() > low.mean():
        print("  Sanity check PASSED: related pairs score higher than unrelated.")
    else:
        print("  Sanity check FAILED: related pairs did not outscore unrelated. "
              "Investigate model output convention before wiring in.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pluggable zero-shot cross-encoder reranker for LOKI."
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run a 10-pair sanity check (8 related, 2 unrelated) and exit."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_CROSS_ENCODER_MODEL,
        help=f"HF cross-encoder model name (default: {DEFAULT_CROSS_ENCODER_MODEL})"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (e.g. 'cuda', 'cpu').  Default: cuda if available."
    )
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test(args.model, args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
