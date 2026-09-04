#!/usr/bin/env python3
"""Compare LOKI and prompt systems on Relationship Clustering.

The default output of this script is the relationship-clustering comparison:
LOKI keeps its stored cluster object, and prompt predictions are reconstructed
into synthetic clusters over GT-matched predicted pairs.

Inputs
- GT/Annotated_Test.json
- Pred/*.json
- ../Batch_Materialization/LOKI_Batch_mimic_GPT_OSS/materialized_batch_results_mimic.csv
- #Results/LOKI_Batch_mimic/materialized_batch_resume_state_mimic.json

Default outputs
- #Results/relationship_clustering_dashboard_summary.csv
- #Results/relationship_clustering_dashboard_per_admission.csv
- #Results/relationship_clustering_dashboard_report.md
- #Results/relationship_clustering_summary.csv
- #Results/relationship_clustering_per_admission.csv
- #Results/relationship_clustering_report.md
- #Results/relationship_clustering_fairness_summary.csv
- #Results/relationship_clustering_fairness_report.md
- #Results/relationship_clustering_visualizations.md
- Visualizations/relationship_clustering/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_GT_FILE = ROOT / "GT/Annotated_Test.json"
DEFAULT_PRED_DIR = ROOT / "Pred"
DEFAULT_LOKI_GPT_RESUME = ROOT / "#Results/LOKI_Batch_mimic_GPT_OSS/materialized_batch_resume_state_mimic.json"
DEFAULT_LOKI_GPT_RESULTS_CSV = ROOT.parent / "Batch_Materialization/LOKI_Batch_mimic_GPT_OSS/materialized_batch_results_mimic.csv"
DEFAULT_LOKI_QWEN_RESUME = ROOT / "#Results/loki_batch_mimic_Qwen-3.6/materialized_batch_resume_state_mimic.json"
DEFAULT_LOKI_QWEN_RESULTS_CSV = ROOT.parent / "Batch_Materialization/loki_batch_mimic_Qwen-3.6/materialized_batch_results_mimic.csv"
DEFAULT_OUTPUT_DIR = ROOT / "#Results"
DEFAULT_VIZ_DIR = ROOT / "Visualizations/relationship_clustering"

DEFAULT_REL_TYPES = ["TREATS", "ADVERSE_EFFECT", "DISCONTINUED", "CONTRAINDICATED", "NEGATIVE"]
REL_TYPES = list(DEFAULT_REL_TYPES)
_KNOWN_REL_TYPE_PRIORITY = {
    "TREATS": 0,
    "ADVERSE_EFFECT": 1,
    "DISCONTINUED": 2,
    "CONTRAINDICATED": 3,
    "NEGATIVE": 4,
    "OTHER": 5,
    "UNLABELED": 6,
}

COMPARISON_PROMPT_COLOR = "#9BB7AE"
COMPARISON_LOKI_COLOR = "#2F5D8A"
COMPARISON_DELTA_COLOR = "#4E6E8E"
COMPARISON_MODEL_PALETTE = [
    "#9BB7AE",
    "#C7886B",
    "#7E9CCB",
    "#B39BC8",
    "#C8A35F",
    "#6DA6A1",
]
COMPARISON_PROMPT_PALETTE = ["#9BB7AE", "#C98D75", "#8FA9C7", "#C7A86A", "#A18FC2", "#78B4AA"]
COMPARISON_LOKI_VARIANT_PALETTE = ["#2F5D8A", "#4C769D", "#6990B0", "#86ABC3", "#A3C5D6", "#C0DFE9"]


PairLabelRecord = Dict[str, Any]
ClusterLabelRecord = Dict[str, Any]


def _normalize_rel_type(rel_type: str) -> str:
    normalized = re.sub(r"\s+", "_", str(rel_type).strip().upper())
    if normalized == "CONTEXT":
        return "NEGATIVE"
    return normalized


def _rel_type_sort_key(rel_type: str) -> Tuple[int, str]:
    normalized = _normalize_rel_type(rel_type)
    return (_KNOWN_REL_TYPE_PRIORITY.get(normalized, len(_KNOWN_REL_TYPE_PRIORITY)), normalized)


def _set_active_rel_types(rel_types: Iterable[str]) -> List[str]:
    global REL_TYPES

    ordered: List[str] = []
    seen: Set[str] = set()
    for rel_type in rel_types:
        normalized = _normalize_rel_type(rel_type)
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)

    if not ordered:
        ordered = list(DEFAULT_REL_TYPES)
    REL_TYPES = sorted(ordered, key=_rel_type_sort_key)
    return REL_TYPES


def _resolve_rel_types_from_gt(gt_file: Path) -> List[str]:
    with gt_file.open("r", encoding="utf-8") as handle:
        annots = json.load(handle)

    discovered: List[str] = []
    for entry in annots.values():
        for rel in entry.get("relationships", []):
            rel_type = _normalize_rel_type(rel.get("relationship_type", ""))
            if rel_type:
                discovered.append(rel_type)
        for flag in entry.get("multi_relationship_flags", []):
            for rel_type in flag.get("relationship_types", []):
                normalized = _normalize_rel_type(rel_type)
                if normalized:
                    discovered.append(normalized)
    return _set_active_rel_types(discovered or DEFAULT_REL_TYPES)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: ("" if row.get(key) is None else row.get(key)) for key in fieldnames}
            writer.writerow(payload)


def _relative_markdown_path(target: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(target, start=base_dir)).as_posix()


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    resolved = [float(value) for value in values if value is not None]
    if not resolved:
        return None
    return round(sum(resolved) / len(resolved), 4)


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _first_present(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def _prf1(pred_items: Set[Any], gt_items: Set[Any]) -> Tuple[float, float, float]:
    tp = len(pred_items & gt_items)
    precision = tp / len(pred_items) if pred_items else 0.0
    recall = tp / len(gt_items) if gt_items else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _score_prediction(
    pred_items: Set[Tuple[int, int]],
    gt_items: Set[Tuple[int, int]],
) -> Tuple[int, float, float, float]:
    tp = len(pred_items & gt_items)
    precision, recall, f1 = _prf1(pred_items, gt_items)
    return tp, precision, recall, f1


def _build_gt_pair_type_sets(
    gt_relationships: Sequence[Dict[str, Any]],
) -> Tuple[Dict[Tuple[int, int], Set[str]], Dict[str, Set[Tuple[int, int]]]]:
    gt_pair_types: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
    gt_by_type: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
    for rel in gt_relationships:
        pair_key = (int(rel["diag_idx"]), int(rel["drug_idx"]))
        rel_type = _normalize_rel_type(str(rel.get("rel_type", "")))
        if not rel_type:
            continue
        gt_pair_types[pair_key].add(rel_type)
        gt_by_type[rel_type].add(pair_key)
    return gt_pair_types, gt_by_type


def _best_rel_type_match(
    pred_items: Set[Tuple[int, int]],
    gt_by_type: Dict[str, Set[Tuple[int, int]]],
) -> Tuple[str, Dict[str, float]]:
    best_type = REL_TYPES[0] if REL_TYPES else ""
    best_score: Optional[Tuple[float, int, float, float]] = None
    best_metrics: Dict[str, float] = {
        "tp": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "n_pred": len(pred_items),
        "n_gt": 0,
    }
    for rel_type in REL_TYPES:
        gt_items = gt_by_type.get(rel_type, set())
        tp, precision, recall, f1 = _score_prediction(pred_items, gt_items)
        score = (f1, tp, precision, recall)
        if best_score is None or score > best_score:
            best_score = score
            best_type = rel_type
            best_metrics = {
                "tp": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_pred": len(pred_items),
                "n_gt": len(gt_items),
            }
    return best_type, best_metrics


def _normalize_saved_cluster_label_record(
    admission_id: str,
    record: Dict[str, Any],
) -> Optional[ClusterLabelRecord]:
    gt_label = _normalize_rel_type(record.get("gt_label", ""))
    if not gt_label:
        return None

    predicted_label = _normalize_rel_type(record.get("predicted_label", "")) or "UNLABELED"
    raw_cluster_id = record.get("cluster_id")
    cluster_id: Any = raw_cluster_id
    if raw_cluster_id not in (None, ""):
        try:
            cluster_id = int(raw_cluster_id)
        except Exception:
            cluster_id = str(raw_cluster_id)

    return {
        "admission_id": str(record.get("admission_id", admission_id)),
        "cluster_id": cluster_id,
        "predicted_label": predicted_label,
        "gt_label": gt_label,
        "correct": bool(record.get("correct")) if record.get("correct") is not None else predicted_label == gt_label,
    }


def _cluster_label_metrics_from_records(
    records: Sequence[ClusterLabelRecord],
) -> Dict[str, Any]:
    pred_typed_clusters: Set[Tuple[Tuple[str, str], str]] = set()
    gt_typed_clusters: Set[Tuple[Tuple[str, str], str]] = set()
    pred_by_type: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    gt_by_type: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    n_correct = 0
    n_evaluated = 0

    for index, record in enumerate(records):
        gt_label = _normalize_rel_type(record.get("gt_label", ""))
        if not gt_label:
            continue
        admission_id = str(record.get("admission_id", ""))
        raw_cluster_id = record.get("cluster_id", index)
        cluster_key = (admission_id, str(raw_cluster_id))
        predicted_label = _normalize_rel_type(record.get("predicted_label", "")) or "UNLABELED"

        gt_typed_clusters.add((cluster_key, gt_label))
        gt_by_type[gt_label].add(cluster_key)
        if predicted_label in REL_TYPES:
            pred_typed_clusters.add((cluster_key, predicted_label))
            pred_by_type[predicted_label].add(cluster_key)

        is_correct = bool(record.get("correct")) if record.get("correct") is not None else predicted_label == gt_label
        if is_correct:
            n_correct += 1
        n_evaluated += 1

    if n_evaluated <= 0:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "n_pred": 0,
            "n_gt": 0,
            "n_evaluated": 0,
            "n_correct": 0,
            "per_type": {},
        }

    precision, recall, f1 = _prf1(pred_typed_clusters, gt_typed_clusters)
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    active_types: List[str] = []
    per_type: Dict[str, Dict[str, Any]] = {}
    for rel_type in REL_TYPES:
        pred_clusters = pred_by_type.get(rel_type, set())
        gt_clusters = gt_by_type.get(rel_type, set())
        type_precision, type_recall, type_f1 = _prf1(pred_clusters, gt_clusters)
        per_type[rel_type] = {
            "precision": type_precision,
            "recall": type_recall,
            "f1": type_f1,
            "n_pred": len(pred_clusters),
            "n_gt": len(gt_clusters),
        }
        if pred_clusters or gt_clusters:
            active_types.append(rel_type)
            macro_precision += type_precision
            macro_recall += type_recall
            macro_f1 += type_f1

    n_active = len(active_types)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": round(n_correct / max(n_evaluated, 1), 4),
        "macro_precision": round(macro_precision / n_active, 4) if n_active > 0 else None,
        "macro_recall": round(macro_recall / n_active, 4) if n_active > 0 else None,
        "macro_f1": round(macro_f1 / n_active, 4) if n_active > 0 else None,
        "n_pred": len(pred_typed_clusters),
        "n_gt": len(gt_typed_clusters),
        "n_evaluated": n_evaluated,
        "n_correct": n_correct,
        "per_type": per_type,
    }


def load_annotation_entries(gt_file: Path) -> Dict[str, Dict]:
    annots = _load_json(gt_file)
    return {str(admission_id): entry for admission_id, entry in annots.items()}


def load_ground_truth_for_admission(
    admission_id: str,
    annotation_entries: Dict[str, Dict],
) -> Tuple[List[Dict[str, Any]], Dict[int, List[int]], Dict[int, List[int]], set]:
    if admission_id not in annotation_entries:
        raise KeyError(f"Admission {admission_id} not found in annotation entries")

    entry = annotation_entries[admission_id]
    rg = entry["row_grounding"]
    gt_diag = {int(k) - 1: v["sentences"] for k, v in rg["diagnosis"].items()}
    gt_med = {int(k) - 1: v["sentences"] for k, v in rg["medication"].items()}

    gt_relationships: List[Dict[str, Any]] = []
    for rel in entry.get("relationships", []):
        gt_relationships.append({
            "diag_idx": int(rel["diagnosis_row"]) - 1,
            "drug_idx": int(rel["drug_row"]) - 1,
            "rel_type": _normalize_rel_type(rel.get("relationship_type", "")),
            "evidence_sents": list(rel.get("evidence_sentences", [])),
        })

    multi_pairs: set = set()
    for flag in entry.get("multi_relationship_flags", []):
        d_idx = int(flag["diagnosis_row"]) - 1
        m_idx = int(flag["drug_row"]) - 1
        multi_pairs.add((d_idx, m_idx))
        for rel_type in flag.get("relationship_types", []):
            gt_relationships.append({
                "diag_idx": d_idx,
                "drug_idx": m_idx,
                "rel_type": _normalize_rel_type(rel_type),
                "evidence_sents": [],
            })

    return gt_relationships, gt_diag, gt_med, multi_pairs


def _sanitize_ground_truth_indices(
    gt_relationships: List[Dict[str, Any]],
    gt_diag: Dict[int, List[int]],
    gt_med: Dict[int, List[int]],
    multi_pairs: set,
    n_diag_rows: int,
    n_med_rows: int,
    n_sentences: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[int]], Dict[int, List[int]], set]:
    def _filter_sent_ids(sent_ids: List[int]) -> List[int]:
        return [int(sent_idx) for sent_idx in sent_ids if 0 <= int(sent_idx) < n_sentences]

    sanitized_relationships: List[Dict[str, Any]] = []
    for rel in gt_relationships:
        diag_idx = int(rel["diag_idx"])
        med_idx = int(rel["drug_idx"])
        if not (0 <= diag_idx < n_diag_rows and 0 <= med_idx < n_med_rows):
            continue
        sanitized_rel = dict(rel)
        sanitized_rel["evidence_sents"] = _filter_sent_ids(list(rel.get("evidence_sents", [])))
        sanitized_relationships.append(sanitized_rel)

    sanitized_gt_diag = {
        int(row_idx): _filter_sent_ids(list(sent_ids))
        for row_idx, sent_ids in gt_diag.items()
        if 0 <= int(row_idx) < n_diag_rows
    }
    sanitized_gt_med = {
        int(row_idx): _filter_sent_ids(list(sent_ids))
        for row_idx, sent_ids in gt_med.items()
        if 0 <= int(row_idx) < n_med_rows
    }
    sanitized_multi_pairs = {
        (int(diag_idx), int(med_idx))
        for diag_idx, med_idx in multi_pairs
        if 0 <= int(diag_idx) < n_diag_rows and 0 <= int(med_idx) < n_med_rows
    }
    return sanitized_relationships, sanitized_gt_diag, sanitized_gt_med, sanitized_multi_pairs


def _build_gt_pair_type_lookup(
    gt_relationships: List[Dict[str, Any]],
) -> Dict[Tuple[int, int], Tuple[str, ...]]:
    pair_types: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
    for rel in gt_relationships:
        pair_types[(int(rel["diag_idx"]), int(rel["drug_idx"]))].add(
            _normalize_rel_type(str(rel.get("rel_type", "")))
        )
    return {
        pair_key: tuple(sorted(rel_types))
        for pair_key, rel_types in pair_types.items()
        if rel_types
    }


def _build_multi_valid(
    gt_relationships: List[Dict[str, Any]],
    multi_pairs: set,
) -> Dict[Tuple[int, int], Set[str]]:
    multi_valid: Dict[Tuple[int, int], Set[str]] = {}
    gt_pair_types = _build_gt_pair_type_lookup(gt_relationships)
    for pair_key in multi_pairs:
        valid_types = set(gt_pair_types.get(pair_key, ()))
        if len(valid_types) > 1:
            multi_valid[pair_key] = valid_types
    return multi_valid


def _select_supported_pair_label(
    label_scores: Dict[str, float],
    label_counts: Dict[str, int],
) -> str:
    if not label_scores:
        return ""
    return min(
        label_scores,
        key=lambda label: (
            -float(label_scores.get(label, 0.0)),
            -int(label_counts.get(label, 0)),
            _rel_type_sort_key(label),
        ),
    )


def _build_pair_label_record(
    admission_id: str,
    pair_key: Tuple[int, int],
    predicted_label_raw: str,
    gt_pair_types: Dict[Tuple[int, int], Tuple[str, ...]],
    multi_valid: Dict[Tuple[int, int], Set[str]],
) -> Optional[PairLabelRecord]:
    valid_gt_types = list(gt_pair_types.get(pair_key, ()))
    if not valid_gt_types:
        return None
    canonical_gt_label = valid_gt_types[0]
    is_multilabel_gt = len(valid_gt_types) > 1
    predicted_label = (
        canonical_gt_label
        if is_multilabel_gt and predicted_label_raw in multi_valid.get(pair_key, set())
        else predicted_label_raw
    )
    return {
        "admission_id": admission_id,
        "diag_row_idx": pair_key[0],
        "med_row_idx": pair_key[1],
        "predicted_label": predicted_label,
        "predicted_label_raw": predicted_label_raw,
        "gt_label": canonical_gt_label,
        "gt_valid_labels": valid_gt_types,
        "is_multilabel_gt": is_multilabel_gt,
    }


def _normalize_saved_pair_label_record(
    admission_id: str,
    record: Dict[str, Any],
) -> Optional[PairLabelRecord]:
    gt_label = _normalize_rel_type(record.get("gt_label", ""))
    if not gt_label:
        return None

    gt_valid_labels = sorted({
        _normalize_rel_type(rel_type)
        for rel_type in (record.get("gt_valid_labels", []) or [])
        if _normalize_rel_type(rel_type)
    })
    if gt_label not in gt_valid_labels:
        gt_valid_labels.append(gt_label)
        gt_valid_labels = sorted(set(gt_valid_labels))

    predicted_label_raw = _normalize_rel_type(
        record.get("predicted_label_raw", "") or record.get("predicted_label", "")
    )
    predicted_label = _normalize_rel_type(
        record.get("predicted_label_eval", "") or record.get("predicted_label", "")
    )
    if len(gt_valid_labels) > 1 and predicted_label_raw in gt_valid_labels:
        predicted_label = gt_label
    if not predicted_label:
        return None

    normalized_record: PairLabelRecord = {
        "admission_id": str(record.get("admission_id", admission_id)),
        "predicted_label": predicted_label,
        "predicted_label_raw": predicted_label_raw,
        "gt_label": gt_label,
        "gt_valid_labels": gt_valid_labels,
        "is_multilabel_gt": len(gt_valid_labels) > 1,
    }
    diag_row_idx = record.get("diag_row_idx")
    med_row_idx = record.get("med_row_idx")
    if diag_row_idx not in (None, ""):
        normalized_record["diag_row_idx"] = int(diag_row_idx)
    if med_row_idx not in (None, ""):
        normalized_record["med_row_idx"] = int(med_row_idx)
    return normalized_record


@dataclass
class PromptAdmission:
    admission_id: str
    patient_id: str
    source_file: str
    n_diag_rows: int
    n_med_rows: int
    n_sentences: int
    completed_entry_count: int
    relationships: List[Dict[str, Any]]
    multi_relationship_flags: List[Dict[str, Any]]


@dataclass
class AdmissionClusterEvaluation:
    system_name: str
    source_file: str
    admission_id: str
    patient_id: str
    gt_label_cardinality: int
    n_gt_matched_pairs: int
    n_clusters: int
    raw_pair_cluster_purity: Optional[float]
    raw_pair_oracle_precision: Optional[float]
    raw_pair_oracle_recall: Optional[float]
    raw_pair_oracle_f1: Optional[float]
    cluster_label_macro_precision: Optional[float]
    cluster_label_macro_recall: Optional[float]
    cluster_label_macro_f1: Optional[float]
    cluster_label_precision: Optional[float]
    cluster_label_recall: Optional[float]
    cluster_label_f1: Optional[float]
    cluster_label_accuracy: Optional[float]
    cluster_ari: Optional[float]
    cluster_label_n_evaluated: int
    cluster_label_n_correct: int
    cluster_label_records: List[ClusterLabelRecord]


def _anchor_parts(anchor_metadata: str) -> Tuple[str, str, str]:
    parts = str(anchor_metadata or "").split("-")
    if len(parts) < 3:
        return "", "", ""
    return parts[0].strip(), parts[1].strip(), "-".join(parts[2:]).strip()


def _canonical_prediction_signature(entry: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    relationships = []
    for rel in entry.get("relationships", []):
        relationships.append((
            int(rel.get("diagnosis_row", 0) or 0),
            int(rel.get("drug_row", 0) or 0),
            _normalize_rel_type(rel.get("relationship_type", "")),
            tuple(sorted(int(sent_idx) for sent_idx in rel.get("evidence_sentences", []))),
        ))
    flags = []
    for flag in entry.get("multi_relationship_flags", []):
        flags.append((
            int(flag.get("diagnosis_row", 0) or 0),
            int(flag.get("drug_row", 0) or 0),
            tuple(sorted(_normalize_rel_type(rel_type) for rel_type in flag.get("relationship_types", []))),
        ))
    return tuple(sorted(relationships)), tuple(sorted(flags))


def load_prompt_admissions(pred_file: Path) -> Dict[str, PromptAdmission]:
    payload = _load_json(pred_file)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in payload.get("annotations", []):
        _patient_id, admission_id, _table_type = _anchor_parts(entry.get("anchor_metadata", ""))
        if admission_id:
            grouped[admission_id].append(entry)

    admissions: Dict[str, PromptAdmission] = {}
    for admission_id, entries in sorted(grouped.items(), key=lambda item: int(item[0])):
        completed_entries = [
            entry for entry in entries if str(entry.get("status", "")).strip().lower() == "completed"
        ]
        if not completed_entries:
            continue

        baseline_signature = _canonical_prediction_signature(completed_entries[0])
        for entry in completed_entries[1:]:
            if _canonical_prediction_signature(entry) != baseline_signature:
                raise ValueError(
                    f"Completed prediction entries disagree for admission {admission_id} in {pred_file.name}"
                )

        diagnosis_entry = next((entry for entry in entries if str(entry.get("table_type", "")).strip().lower() == "diagnosis"), None)
        medication_entry = next((entry for entry in entries if str(entry.get("table_type", "")).strip().lower() == "medication"), None)
        source_entry = completed_entries[0]
        patient_id, _, _ = _anchor_parts(source_entry.get("anchor_metadata", ""))
        n_diag_rows = int((((diagnosis_entry or {}).get("reference_info") or {}).get("num_rows")) or 0)
        n_med_rows = int((((medication_entry or {}).get("reference_info") or {}).get("num_rows")) or 0)
        n_sentences = max(
            int((((entry.get("reference_info") or {}).get("num_sentences")) or 0))
            for entry in entries
        )
        if n_diag_rows <= 0 or n_med_rows <= 0 or n_sentences <= 0:
            continue

        admissions[admission_id] = PromptAdmission(
            admission_id=admission_id,
            patient_id=patient_id,
            source_file=pred_file.name,
            n_diag_rows=n_diag_rows,
            n_med_rows=n_med_rows,
            n_sentences=n_sentences,
            completed_entry_count=len(completed_entries),
            relationships=list(source_entry.get("relationships", [])),
            multi_relationship_flags=list(source_entry.get("multi_relationship_flags", [])),
        )
    return admissions


def build_prompt_pair_label_records(
    prompt_admission: PromptAdmission,
    annotation_entries: Dict[str, Dict],
) -> List[PairLabelRecord]:
    gt_relationships, gt_diag, gt_med, multi_pairs = load_ground_truth_for_admission(
        prompt_admission.admission_id,
        annotation_entries,
    )
    gt_relationships, gt_diag, gt_med, multi_pairs = _sanitize_ground_truth_indices(
        gt_relationships,
        gt_diag,
        gt_med,
        multi_pairs,
        prompt_admission.n_diag_rows,
        prompt_admission.n_med_rows,
        prompt_admission.n_sentences,
    )
    gt_pair_types = _build_gt_pair_type_lookup(gt_relationships)
    multi_valid = _build_multi_valid(gt_relationships, multi_pairs)

    predicted_pair_labels = _build_prompt_predicted_pair_labels(prompt_admission)

    records: List[PairLabelRecord] = []
    for pair_key in sorted(predicted_pair_labels):
        predicted_label_raw = predicted_pair_labels[pair_key]
        record = _build_pair_label_record(
            prompt_admission.admission_id,
            pair_key,
            predicted_label_raw,
            gt_pair_types,
            multi_valid,
        )
        if record is not None:
            records.append(record)
    return records


def _build_prompt_predicted_pair_labels(
    prompt_admission: PromptAdmission,
) -> Dict[Tuple[int, int], str]:
    predicted_pair_scores: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    predicted_pair_counts: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rel in prompt_admission.relationships:
        diag_idx = int(rel.get("diagnosis_row", 0) or 0) - 1
        med_idx = int(rel.get("drug_row", 0) or 0) - 1
        rel_type = _normalize_rel_type(rel.get("relationship_type", ""))
        pair_key = (diag_idx, med_idx)
        if rel_type == "":
            continue
        if not (0 <= diag_idx < prompt_admission.n_diag_rows and 0 <= med_idx < prompt_admission.n_med_rows):
            continue
        predicted_pair_scores[pair_key][rel_type] += 1.0
        predicted_pair_counts[pair_key][rel_type] += 1

    for flag in prompt_admission.multi_relationship_flags:
        diag_idx = int(flag.get("diagnosis_row", 0) or 0) - 1
        med_idx = int(flag.get("drug_row", 0) or 0) - 1
        pair_key = (diag_idx, med_idx)
        if not (0 <= diag_idx < prompt_admission.n_diag_rows and 0 <= med_idx < prompt_admission.n_med_rows):
            continue
        for rel_type_raw in flag.get("relationship_types", []):
            rel_type = _normalize_rel_type(rel_type_raw)
            if rel_type:
                predicted_pair_scores[pair_key][rel_type] += 1.0
                predicted_pair_counts[pair_key][rel_type] += 1

    predicted_pair_labels: Dict[Tuple[int, int], str] = {}
    for pair_key in sorted(predicted_pair_scores):
        predicted_label_raw = _select_supported_pair_label(
            predicted_pair_scores[pair_key],
            predicted_pair_counts[pair_key],
        )
        if not predicted_label_raw:
            continue
        predicted_pair_labels[pair_key] = predicted_label_raw
    return predicted_pair_labels


def _pair_label_cluster_quality(
    pair_records: Sequence[PairLabelRecord],
) -> Dict[str, Any]:
    pred_labels: List[str] = []
    true_labels: List[str] = []

    for record in pair_records:
        predicted_label_raw = _normalize_rel_type(
            record.get("predicted_label_raw", "") or record.get("predicted_label", "")
        )
        if predicted_label_raw not in REL_TYPES:
            continue

        gt_valid_labels = [
            _normalize_rel_type(rel_type)
            for rel_type in (record.get("gt_valid_labels", []) or [])
            if _normalize_rel_type(rel_type)
        ]
        if not gt_valid_labels:
            gt_label = _normalize_rel_type(record.get("gt_label", ""))
            if gt_label:
                gt_valid_labels = [gt_label]
        if not gt_valid_labels:
            continue

        true_label = (
            predicted_label_raw
            if predicted_label_raw in gt_valid_labels
            else gt_valid_labels[0]
        )
        pred_labels.append(predicted_label_raw)
        true_labels.append(true_label)

    if not pred_labels:
        return {"purity": None, "ari": None, "n_evaluated": 0}

    type_buckets: Dict[str, List[str]] = defaultdict(list)
    for predicted_label, true_label in zip(pred_labels, true_labels):
        type_buckets[predicted_label].append(true_label)

    purity = (
        sum(Counter(bucket).most_common(1)[0][1] for bucket in type_buckets.values())
        / len(pred_labels)
    )

    ari = None
    try:
        from sklearn.metrics import adjusted_rand_score  # type: ignore

        all_labels = sorted(set(pred_labels + true_labels), key=_rel_type_sort_key)
        label_to_index = {label: index for index, label in enumerate(all_labels)}
        ari = adjusted_rand_score(
            [label_to_index[label] for label in true_labels],
            [label_to_index[label] for label in pred_labels],
        )
    except Exception:
        ari = None

    return {
        "purity": round(purity, 4),
        "ari": round(float(ari), 4) if ari is not None else None,
        "n_evaluated": len(pred_labels),
    }


def _evaluate_cluster_membership_from_pair_records(
    admission_id: str,
    pair_records: Sequence[PairLabelRecord],
    gt_relationships: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[ClusterLabelRecord]]:
    gt_pair_types, gt_by_type = _build_gt_pair_type_sets(gt_relationships)
    cluster_members: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
    for record in pair_records:
        diag_row_idx = record.get("diag_row_idx")
        med_row_idx = record.get("med_row_idx")
        if diag_row_idx in (None, "") or med_row_idx in (None, ""):
            continue
        pair_key = (int(diag_row_idx), int(med_row_idx))
        if pair_key not in gt_pair_types:
            continue
        predicted_label = _normalize_rel_type(
            record.get("predicted_label_raw", "") or record.get("predicted_label", "")
        ) or "UNLABELED"
        cluster_members[predicted_label].add(pair_key)

    if not cluster_members:
        return {
            "n_clusters": 0,
            "n_gt_matched_pairs": 0,
            "raw_pair_cluster_purity": None,
            "raw_pair_oracle_precision": None,
            "raw_pair_oracle_recall": None,
            "raw_pair_oracle_f1": None,
        }, []

    cluster_label_records: List[ClusterLabelRecord] = []
    pred_by_type: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
    gt_matched_pairs: Set[Tuple[int, int]] = set()
    dominant_sum = 0

    for cluster_index, predicted_label in enumerate(sorted(cluster_members, key=_rel_type_sort_key)):
        members = {pair for pair in cluster_members[predicted_label] if pair in gt_pair_types}
        if not members:
            continue
        gt_matched_pairs.update(members)
        dominant_sum += max(
            (
                sum(1 for pair in members if rel_type in gt_pair_types[pair])
                for rel_type in REL_TYPES
            ),
            default=0,
        )
        oracle_type, oracle_metrics = _best_rel_type_match(members, gt_by_type)
        pred_by_type[oracle_type].update(members)
        cluster_label_records.append({
            "admission_id": admission_id,
            "cluster_id": cluster_index,
            "predicted_label": predicted_label,
            "gt_label": oracle_type,
            "correct": predicted_label == oracle_type,
            "n_pairs": len(members),
            "oracle_tp": oracle_metrics.get("tp", 0),
            "oracle_precision": oracle_metrics.get("precision"),
            "oracle_recall": oracle_metrics.get("recall"),
            "oracle_f1": oracle_metrics.get("f1"),
        })

    if not gt_matched_pairs:
        return {
            "n_clusters": 0,
            "n_gt_matched_pairs": 0,
            "raw_pair_cluster_purity": None,
            "raw_pair_oracle_precision": None,
            "raw_pair_oracle_recall": None,
            "raw_pair_oracle_f1": None,
        }, []

    pred_typed_pairs = {
        (*pair, rel_type)
        for rel_type, members in pred_by_type.items()
        for pair in members
    }
    gt_typed_pairs = {
        (*pair, rel_type)
        for pair, rel_types in gt_pair_types.items()
        for rel_type in rel_types
    }
    oracle_precision, oracle_recall, oracle_f1 = _prf1(pred_typed_pairs, gt_typed_pairs)
    return {
        "n_clusters": len(cluster_label_records),
        "n_gt_matched_pairs": len(gt_matched_pairs),
        "raw_pair_cluster_purity": round(dominant_sum / max(len(gt_matched_pairs), 1), 4),
        "raw_pair_oracle_precision": oracle_precision,
        "raw_pair_oracle_recall": oracle_recall,
        "raw_pair_oracle_f1": oracle_f1,
    }, cluster_label_records


def build_prompt_cluster_evaluations(
    pred_file: Path,
    annotation_entries: Dict[str, Dict],
) -> List[AdmissionClusterEvaluation]:
    admissions = load_prompt_admissions(pred_file)
    evaluations: List[AdmissionClusterEvaluation] = []
    for admission_id in sorted(admissions, key=int):
        if admission_id not in annotation_entries:
            continue
        prompt_admission = admissions[admission_id]
        gt_relationships, gt_diag, gt_med, multi_pairs = load_ground_truth_for_admission(
            admission_id,
            annotation_entries,
        )
        gt_relationships, gt_diag, gt_med, multi_pairs = _sanitize_ground_truth_indices(
            gt_relationships,
            gt_diag,
            gt_med,
            multi_pairs,
            prompt_admission.n_diag_rows,
            prompt_admission.n_med_rows,
            prompt_admission.n_sentences,
        )
        pair_records = build_prompt_pair_label_records(prompt_admission, annotation_entries)
        if not pair_records:
            continue

        raw_cluster_metrics, cluster_label_records = _evaluate_cluster_membership_from_pair_records(
            admission_id,
            pair_records,
            gt_relationships,
        )
        cluster_label_metrics = _cluster_label_metrics_from_records(cluster_label_records)
        cluster_quality = _pair_label_cluster_quality(pair_records)
        gt_label_cardinality = len({
            str(record.get("gt_label", ""))
            for record in pair_records
            if str(record.get("gt_label", ""))
        })

        evaluations.append(AdmissionClusterEvaluation(
            system_name=pred_file.stem,
            source_file=pred_file.name,
            admission_id=admission_id,
            patient_id=prompt_admission.patient_id,
            gt_label_cardinality=gt_label_cardinality,
            n_gt_matched_pairs=int(raw_cluster_metrics.get("n_gt_matched_pairs") or 0),
            n_clusters=int(raw_cluster_metrics.get("n_clusters") or 0),
            raw_pair_cluster_purity=_safe_float(raw_cluster_metrics.get("raw_pair_cluster_purity")),
            raw_pair_oracle_precision=_safe_float(raw_cluster_metrics.get("raw_pair_oracle_precision")),
            raw_pair_oracle_recall=_safe_float(raw_cluster_metrics.get("raw_pair_oracle_recall")),
            raw_pair_oracle_f1=_safe_float(raw_cluster_metrics.get("raw_pair_oracle_f1")),
            cluster_label_macro_precision=_safe_float(cluster_label_metrics.get("macro_precision")),
            cluster_label_macro_recall=_safe_float(cluster_label_metrics.get("macro_recall")),
            cluster_label_macro_f1=_safe_float(cluster_label_metrics.get("macro_f1")),
            cluster_label_precision=_safe_float(cluster_label_metrics.get("precision")),
            cluster_label_recall=_safe_float(cluster_label_metrics.get("recall")),
            cluster_label_f1=_safe_float(cluster_label_metrics.get("f1")),
            cluster_label_accuracy=_safe_float(cluster_label_metrics.get("accuracy")),
            cluster_ari=_safe_float(cluster_quality.get("ari")),
            cluster_label_n_evaluated=int(cluster_label_metrics.get("n_evaluated") or 0),
            cluster_label_n_correct=int(cluster_label_metrics.get("n_correct") or 0),
            cluster_label_records=cluster_label_records,
        ))
    return evaluations


def load_loki_cluster_evaluations(resume_state_file: Path) -> List[AdmissionClusterEvaluation]:
    payload = _load_json(resume_state_file)
    evaluations: List[AdmissionClusterEvaluation] = []
    for item in payload:
        admission_id = str(item.get("admission_id", "")).strip()
        patient_id = str(item.get("patient_id", "")).strip()
        if not admission_id:
            continue

        pair_records = [
            record
            for record in item.get("pair_label_records", [])
            if isinstance(record, dict)
        ]
        cluster_label_records = []
        for record in item.get("cluster_label_records", []):
            if not isinstance(record, dict):
                continue
            normalized_record = _normalize_saved_cluster_label_record(admission_id, record)
            if normalized_record is not None:
                cluster_label_records.append(normalized_record)

        batch_row = item.get("batch_row", {}) if isinstance(item.get("batch_row"), dict) else {}
        cluster_label_metrics = _cluster_label_metrics_from_records(cluster_label_records)
        gt_label_cardinality = len({
            _normalize_rel_type(record.get("gt_label", ""))
            for record in pair_records
            if _normalize_rel_type(record.get("gt_label", ""))
        })
        n_clusters = len({record.get("cluster_id") for record in cluster_label_records})
        if n_clusters <= 0:
            n_clusters = int(_safe_float(batch_row.get("cluster_label_n_evaluated")) or 0)

        n_gt_matched_pairs = len(pair_records)
        if n_gt_matched_pairs <= 0 and not cluster_label_records:
            continue

        evaluations.append(AdmissionClusterEvaluation(
            system_name="LOKI",
            source_file=resume_state_file.name,
            admission_id=admission_id,
            patient_id=patient_id,
            gt_label_cardinality=gt_label_cardinality,
            n_gt_matched_pairs=n_gt_matched_pairs,
            n_clusters=n_clusters,
            raw_pair_cluster_purity=_safe_float(batch_row.get("raw_pair_cluster_purity")),
            raw_pair_oracle_precision=_safe_float(batch_row.get("raw_pair_oracle_precision")),
            raw_pair_oracle_recall=_safe_float(batch_row.get("raw_pair_oracle_recall")),
            raw_pair_oracle_f1=_safe_float(batch_row.get("raw_pair_oracle_f1")),
            cluster_label_macro_precision=_first_present(
                _safe_float(cluster_label_metrics.get("macro_precision")),
                _safe_float(batch_row.get("cluster_label_macro_precision")),
            ),
            cluster_label_macro_recall=_first_present(
                _safe_float(cluster_label_metrics.get("macro_recall")),
                _safe_float(batch_row.get("cluster_label_macro_recall")),
            ),
            cluster_label_macro_f1=_first_present(
                _safe_float(cluster_label_metrics.get("macro_f1")),
                _safe_float(batch_row.get("cluster_label_macro_f1")),
            ),
            cluster_label_precision=_first_present(
                _safe_float(cluster_label_metrics.get("precision")),
                _safe_float(batch_row.get("cluster_label_precision")),
            ),
            cluster_label_recall=_first_present(
                _safe_float(cluster_label_metrics.get("recall")),
                _safe_float(batch_row.get("cluster_label_recall")),
            ),
            cluster_label_f1=_first_present(
                _safe_float(cluster_label_metrics.get("f1")),
                _safe_float(batch_row.get("cluster_label_f1")),
            ),
            cluster_label_accuracy=_first_present(
                _safe_float(cluster_label_metrics.get("accuracy")),
                _safe_float(batch_row.get("cluster_label_accuracy")),
            ),
            cluster_ari=_safe_float(batch_row.get("cluster_ari")),
            cluster_label_n_evaluated=int(
                cluster_label_metrics.get("n_evaluated")
                or _safe_float(batch_row.get("cluster_label_n_evaluated"))
                or 0
            ),
            cluster_label_n_correct=int(
                cluster_label_metrics.get("n_correct")
                or _safe_float(batch_row.get("cluster_label_n_correct"))
                or 0
            ),
            cluster_label_records=cluster_label_records,
        ))
    return evaluations


def _dashboard_cluster_metric(value: Any, n_evaluated: int) -> Optional[float]:
    resolved = _safe_float(value)
    if resolved is None and n_evaluated <= 0:
        return 0.0
    return resolved


def _conservative_qwen_metric(
    row: Dict[str, Any],
    metric_key: str,
    coverage_key: str = "raw_pair_oracle_recall",
) -> Optional[float]:
    metric_value = _safe_float(row.get(metric_key))
    coverage_value = _safe_float(row.get(coverage_key))
    if metric_value is None or coverage_value is None:
        return None
    return round(float(metric_value) * float(coverage_value), 4)


def load_dashboard_rows(results_csv_file: Path) -> List[Dict[str, Any]]:
    rows = _load_csv_rows(results_csv_file)
    return [row for row in rows if str(row.get("admission_id", "")).strip()]


def build_prompt_dashboard_rows(
    pred_file: Path,
    annotation_entries: Dict[str, Dict],
) -> List[Dict[str, Any]]:
    admissions = load_prompt_admissions(pred_file)
    rows: List[Dict[str, Any]] = []

    for admission_id in sorted(admissions, key=int):
        if admission_id not in annotation_entries:
            continue

        prompt_admission = admissions[admission_id]
        gt_relationships, gt_diag, gt_med, multi_pairs = load_ground_truth_for_admission(
            admission_id,
            annotation_entries,
        )
        gt_relationships, gt_diag, gt_med, multi_pairs = _sanitize_ground_truth_indices(
            gt_relationships,
            gt_diag,
            gt_med,
            multi_pairs,
            prompt_admission.n_diag_rows,
            prompt_admission.n_med_rows,
            prompt_admission.n_sentences,
        )

        gt_pair_types = _build_gt_pair_type_lookup(gt_relationships)
        multi_valid = _build_multi_valid(gt_relationships, multi_pairs)
        predicted_pair_labels = _build_prompt_predicted_pair_labels(prompt_admission)

        pair_records: List[PairLabelRecord] = []
        for pair_key in sorted(predicted_pair_labels):
            record = _build_pair_label_record(
                admission_id,
                pair_key,
                predicted_pair_labels[pair_key],
                gt_pair_types,
                multi_valid,
            )
            if record is not None:
                pair_records.append(record)

        raw_cluster_metrics, cluster_label_records = _evaluate_cluster_membership_from_pair_records(
            admission_id,
            pair_records,
            gt_relationships,
        )
        cluster_label_metrics = _cluster_label_metrics_from_records(cluster_label_records)
        cluster_quality = _pair_label_cluster_quality(pair_records)

        n_evaluated = int(cluster_label_metrics.get("n_evaluated") or 0)
        n_correct = int(cluster_label_metrics.get("n_correct") or 0)
        n_pred_pairs = len(predicted_pair_labels)
        n_gt_pairs = len(gt_pair_types)
        n_final_clusters = len({label for label in predicted_pair_labels.values() if label})

        rows.append({
            "dataset": "mimic",
            "evaluation_profile": "prompt_reconstructed_relationship_clustering",
            "admission_id": admission_id,
            "patient_id": prompt_admission.patient_id,
            "runtime_sec": None,
            "n_diag_rows": prompt_admission.n_diag_rows,
            "n_med_rows": prompt_admission.n_med_rows,
            "n_sentences": prompt_admission.n_sentences,
            "n_paths": len(prompt_admission.relationships),
            "n_pred_pairs": n_pred_pairs,
            "n_gt_pairs": n_gt_pairs,
            "n_final_clusters": n_final_clusters,
            "cluster_label_backend": pred_file.stem,
            "gliner2_label_input_mode": None,
            "pair_average_precision": None,
            "cluster_label_macro_precision": _dashboard_cluster_metric(cluster_label_metrics.get("macro_precision"), n_evaluated),
            "cluster_label_macro_recall": _dashboard_cluster_metric(cluster_label_metrics.get("macro_recall"), n_evaluated),
            "cluster_label_macro_f1": _dashboard_cluster_metric(cluster_label_metrics.get("macro_f1"), n_evaluated),
            "cluster_label_precision": _dashboard_cluster_metric(cluster_label_metrics.get("precision"), n_evaluated),
            "cluster_label_recall": _dashboard_cluster_metric(cluster_label_metrics.get("recall"), n_evaluated),
            "cluster_label_f1": _dashboard_cluster_metric(cluster_label_metrics.get("f1"), n_evaluated),
            "cluster_label_accuracy": _dashboard_cluster_metric(cluster_label_metrics.get("accuracy"), n_evaluated),
            "cluster_label_n_evaluated": n_evaluated,
            "cluster_label_n_correct": n_correct,
            "raw_pair_cluster_purity": _safe_float(raw_cluster_metrics.get("raw_pair_cluster_purity")),
            "raw_pair_oracle_precision": _safe_float(raw_cluster_metrics.get("raw_pair_oracle_precision")),
            "raw_pair_oracle_recall": _safe_float(raw_cluster_metrics.get("raw_pair_oracle_recall")),
            "raw_pair_oracle_f1": _safe_float(raw_cluster_metrics.get("raw_pair_oracle_f1")),
            "cluster_purity": _safe_float(cluster_quality.get("purity")),
            "cluster_ari": _safe_float(cluster_quality.get("ari")),
            "cluster_silhouette": None,
        })

    return rows


def summarize_dashboard_rows(
    system_name: str,
    source_file: str,
    scope: str,
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def _mean_metric(key: str) -> Optional[float]:
        return _mean(_safe_float(row.get(key)) for row in rows)

    def _sum_int_metric(key: str) -> int:
        total = 0
        for row in rows:
            value = _safe_float(row.get(key))
            if value is not None:
                total += int(round(value))
        return total

    def _sum_optional_int_metric(key: str) -> Optional[int]:
        total = 0
        found = False
        for row in rows:
            value = _safe_float(row.get(key))
            if value is None:
                continue
            total += int(round(value))
            found = True
        return total if found else None

    return {
        "system_name": system_name,
        "source_file": source_file,
        "scope": scope,
        "n_admissions": len(rows),
        "n_pred_pairs": _sum_int_metric("n_pred_pairs"),
        "n_gt_pairs": _sum_int_metric("n_gt_pairs"),
        "n_final_clusters": _sum_int_metric("n_final_clusters"),
        "cluster_label_n_evaluated": _sum_optional_int_metric("cluster_label_n_evaluated"),
        "cluster_label_n_correct": _sum_optional_int_metric("cluster_label_n_correct"),
        "pair_average_precision": _mean_metric("pair_average_precision"),
        "cluster_label_macro_precision": _mean_metric("cluster_label_macro_precision"),
        "cluster_label_macro_recall": _mean_metric("cluster_label_macro_recall"),
        "cluster_label_macro_f1": _mean_metric("cluster_label_macro_f1"),
        "cluster_label_precision": _mean_metric("cluster_label_precision"),
        "cluster_label_recall": _mean_metric("cluster_label_recall"),
        "cluster_label_f1": _mean_metric("cluster_label_f1"),
        "cluster_label_accuracy": _mean_metric("cluster_label_accuracy"),
        "raw_pair_cluster_purity": _mean_metric("raw_pair_cluster_purity"),
        "raw_pair_oracle_precision": _mean_metric("raw_pair_oracle_precision"),
        "raw_pair_oracle_recall": _mean_metric("raw_pair_oracle_recall"),
        "raw_pair_oracle_f1": _mean_metric("raw_pair_oracle_f1"),
        "cluster_ari": _mean_metric("cluster_ari"),
        "cluster_silhouette": _mean_metric("cluster_silhouette"),
    }


def build_relationship_dashboard_report(summary_rows: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Relationship Clustering Dashboards")
    lines.append("")
    lines.append("These summaries use LOKI's original per-admission dashboard aggregation semantics.")
    lines.append("")
    lines.append("- LOKI is loaded directly from the original batch-results CSV.")
    lines.append("- Prompt systems are reconstructed into the same per-admission row shape before aggregation.")
    lines.append("- Admissions with zero evaluated clusters contribute 0.0 to macro P/R/F1 and mean accuracy, matching the original dashboard behavior.")
    lines.append("- Prompt cluster silhouette is unavailable because the prompt JSON does not expose the embedding space used by LOKI's silhouette computation.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| System | Admissions | Pred pairs | GT pairs | Final clusters | Evaluated clusters | Correct clusters | Macro P | Macro R | Macro F1 | Mean accuracy | Raw purity | Raw oracle F1 | Cluster ARI | Cluster silhouette |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            "| {system} | {n_adm} | {n_pred_pairs} | {n_gt_pairs} | {n_final_clusters} | {n_eval} | {n_correct} | {macro_p} | {macro_r} | {macro_f1} | {accuracy} | {raw_purity} | {raw_f1} | {cluster_ari} | {cluster_silhouette} |".format(
                system=row.get("system_name", ""),
                n_adm=row.get("n_admissions", 0),
                n_pred_pairs=row.get("n_pred_pairs", 0),
                n_gt_pairs=row.get("n_gt_pairs", 0),
                n_final_clusters=row.get("n_final_clusters", 0),
                n_eval=row.get("cluster_label_n_evaluated", 0),
                n_correct=row.get("cluster_label_n_correct", 0),
                macro_p=_fmt(_safe_float(row.get("cluster_label_macro_precision"))),
                macro_r=_fmt(_safe_float(row.get("cluster_label_macro_recall"))),
                macro_f1=_fmt(_safe_float(row.get("cluster_label_macro_f1"))),
                accuracy=_fmt(_safe_float(row.get("cluster_label_accuracy"))),
                raw_purity=_fmt(_safe_float(row.get("raw_pair_cluster_purity"))),
                raw_f1=_fmt(_safe_float(row.get("raw_pair_oracle_f1"))),
                cluster_ari=_fmt(_safe_float(row.get("cluster_ari"))),
                cluster_silhouette=_fmt(_safe_float(row.get("cluster_silhouette"))),
            )
        )
    return "\n".join(lines) + "\n"


def summarize_cluster_system(
    system_name: str,
    source_file: str,
    scope: str,
    evaluations: Sequence[AdmissionClusterEvaluation],
) -> Dict[str, Any]:
    all_cluster_label_records = [
        record
        for evaluation in evaluations
        for record in evaluation.cluster_label_records
    ]
    corpus_cluster_label_metrics = _cluster_label_metrics_from_records(all_cluster_label_records)
    return {
        "system_name": system_name,
        "source_file": source_file,
        "scope": scope,
        "n_admissions": len(evaluations),
        "n_gt_matched_pairs": sum(evaluation.n_gt_matched_pairs for evaluation in evaluations),
        "n_clusters": sum(evaluation.n_clusters for evaluation in evaluations),
        "raw_pair_cluster_purity": _mean(evaluation.raw_pair_cluster_purity for evaluation in evaluations),
        "raw_pair_oracle_precision": _mean(evaluation.raw_pair_oracle_precision for evaluation in evaluations),
        "raw_pair_oracle_recall": _mean(evaluation.raw_pair_oracle_recall for evaluation in evaluations),
        "raw_pair_oracle_f1": _mean(evaluation.raw_pair_oracle_f1 for evaluation in evaluations),
        "cluster_label_macro_precision": _mean(evaluation.cluster_label_macro_precision for evaluation in evaluations),
        "cluster_label_macro_recall": _mean(evaluation.cluster_label_macro_recall for evaluation in evaluations),
        "cluster_label_macro_f1": _mean(evaluation.cluster_label_macro_f1 for evaluation in evaluations),
        "cluster_label_precision": _safe_float(corpus_cluster_label_metrics.get("precision")),
        "cluster_label_recall": _safe_float(corpus_cluster_label_metrics.get("recall")),
        "cluster_label_f1": _safe_float(corpus_cluster_label_metrics.get("f1")),
        "cluster_label_accuracy": _safe_float(corpus_cluster_label_metrics.get("accuracy")),
        "cluster_ari": _mean(evaluation.cluster_ari for evaluation in evaluations),
        "n_evaluated_clusters": int(corpus_cluster_label_metrics.get("n_evaluated") or 0),
        "n_correct_clusters": int(corpus_cluster_label_metrics.get("n_correct") or 0),
    }


def _corpus_cluster_label_per_type_rows(
    system_name: str,
    scope: str,
    source_file: str,
    records: Sequence[ClusterLabelRecord],
) -> List[Dict[str, Any]]:
    metrics = _cluster_label_metrics_from_records(records)
    per_type = metrics.get("per_type", {}) if isinstance(metrics.get("per_type"), dict) else {}
    rows: List[Dict[str, Any]] = []
    for rel_type in REL_TYPES:
        metric = per_type.get(rel_type, {})
        n_pred = int(metric.get("n_pred") or 0)
        n_gt = int(metric.get("n_gt") or 0)
        if n_pred <= 0 and n_gt <= 0:
            continue
        rows.append({
            "system_name": system_name,
            "scope": scope,
            "source_file": source_file,
            "rel_type": rel_type,
            "precision": _safe_float(metric.get("precision")),
            "recall": _safe_float(metric.get("recall")),
            "f1": _safe_float(metric.get("f1")),
            "n_pred": n_pred,
            "n_gt": n_gt,
        })
    return rows


def _cluster_support_overlap(
    prompt_by_admission: Dict[str, AdmissionClusterEvaluation],
    loki_by_admission: Dict[str, AdmissionClusterEvaluation],
    admission_ids: Sequence[str],
    field_name: str,
) -> Tuple[int, int, int]:
    prompt_more = 0
    equal = 0
    prompt_less = 0
    for admission_id in admission_ids:
        prompt_value = int(getattr(prompt_by_admission[admission_id], field_name, 0) or 0)
        loki_value = int(getattr(loki_by_admission[admission_id], field_name, 0) or 0)
        if prompt_value > loki_value:
            prompt_more += 1
        elif prompt_value < loki_value:
            prompt_less += 1
        else:
            equal += 1
    return prompt_more, equal, prompt_less


def build_cluster_fairness_row(
    prompt_name: str,
    prompt_source_file: str,
    scope: str,
    admission_ids: Sequence[str],
    prompt_by_admission: Dict[str, AdmissionClusterEvaluation],
    loki_by_admission: Dict[str, AdmissionClusterEvaluation],
    loki_system_name: str = "LOKI",
    loki_resume_name: str = "loki_resume.json",
) -> Dict[str, Any]:
    prompt_slice = [prompt_by_admission[admission_id] for admission_id in admission_ids]
    loki_slice = [loki_by_admission[admission_id] for admission_id in admission_ids]
    prompt_summary = summarize_cluster_system(prompt_name, prompt_source_file, scope, prompt_slice)
    loki_summary = summarize_cluster_system(loki_system_name, loki_resume_name, scope, loki_slice)
    prompt_more_pairs, equal_pairs, prompt_less_pairs = _cluster_support_overlap(
        prompt_by_admission,
        loki_by_admission,
        admission_ids,
        "n_gt_matched_pairs",
    )
    prompt_more_clusters, equal_clusters, prompt_less_clusters = _cluster_support_overlap(
        prompt_by_admission,
        loki_by_admission,
        admission_ids,
        "n_clusters",
    )
    prompt_single_type = sum(1 for evaluation in prompt_slice if evaluation.gt_label_cardinality <= 1)
    loki_single_type = sum(1 for evaluation in loki_slice if evaluation.gt_label_cardinality <= 1)
    return {
        "prompt_system_name": prompt_name,
        "prompt_source_file": prompt_source_file,
        "loki_system_name": loki_system_name,
        "scope": scope,
        "n_admissions": len(admission_ids),
        "prompt_n_gt_matched_pairs": prompt_summary.get("n_gt_matched_pairs"),
        "loki_n_gt_matched_pairs": loki_summary.get("n_gt_matched_pairs"),
        "prompt_n_clusters": prompt_summary.get("n_clusters"),
        "loki_n_clusters": loki_summary.get("n_clusters"),
        "prompt_raw_pair_cluster_purity": prompt_summary.get("raw_pair_cluster_purity"),
        "loki_raw_pair_cluster_purity": loki_summary.get("raw_pair_cluster_purity"),
        "prompt_raw_pair_oracle_precision": prompt_summary.get("raw_pair_oracle_precision"),
        "prompt_raw_pair_oracle_recall": prompt_summary.get("raw_pair_oracle_recall"),
        "prompt_raw_pair_oracle_f1": prompt_summary.get("raw_pair_oracle_f1"),
        "loki_raw_pair_oracle_f1": loki_summary.get("raw_pair_oracle_f1"),
        "prompt_cluster_label_macro_precision": prompt_summary.get("cluster_label_macro_precision"),
        "loki_cluster_label_macro_precision": loki_summary.get("cluster_label_macro_precision"),
        "prompt_cluster_label_macro_recall": prompt_summary.get("cluster_label_macro_recall"),
        "loki_cluster_label_macro_recall": loki_summary.get("cluster_label_macro_recall"),
        "prompt_cluster_label_macro_f1": prompt_summary.get("cluster_label_macro_f1"),
        "loki_cluster_label_macro_f1": loki_summary.get("cluster_label_macro_f1"),
        "prompt_cluster_label_accuracy": prompt_summary.get("cluster_label_accuracy"),
        "loki_cluster_label_accuracy": loki_summary.get("cluster_label_accuracy"),
        "prompt_cluster_ari": prompt_summary.get("cluster_ari"),
        "loki_cluster_ari": loki_summary.get("cluster_ari"),
        "delta_raw_pair_oracle_f1": None if prompt_summary.get("raw_pair_oracle_f1") is None or loki_summary.get("raw_pair_oracle_f1") is None else round(float(prompt_summary["raw_pair_oracle_f1"]) - float(loki_summary["raw_pair_oracle_f1"]), 4),
        "delta_cluster_label_macro_f1": None if prompt_summary.get("cluster_label_macro_f1") is None or loki_summary.get("cluster_label_macro_f1") is None else round(float(prompt_summary["cluster_label_macro_f1"]) - float(loki_summary["cluster_label_macro_f1"]), 4),
        "delta_cluster_label_accuracy": None if prompt_summary.get("cluster_label_accuracy") is None or loki_summary.get("cluster_label_accuracy") is None else round(float(prompt_summary["cluster_label_accuracy"]) - float(loki_summary["cluster_label_accuracy"]), 4),
        "prompt_single_type_admissions": prompt_single_type,
        "loki_single_type_admissions": loki_single_type,
        "prompt_more_pairs_admissions": prompt_more_pairs,
        "equal_pair_count_admissions": equal_pairs,
        "prompt_fewer_pairs_admissions": prompt_less_pairs,
        "prompt_more_clusters_admissions": prompt_more_clusters,
        "equal_cluster_count_admissions": equal_clusters,
        "prompt_fewer_clusters_admissions": prompt_less_clusters,
    }


def build_cluster_report(summary_rows: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Relationship Clustering")
    lines.append("")
    lines.append("This report compares Qwen and LOKI on Relationship Clustering rather than the old pair-label proxy surface.")
    lines.append("")
    lines.append("- LOKI uses its stored cluster artifacts: raw pair-cluster purity / oracle-pair F1 plus cluster-label metrics from the batch resume-state.")
    lines.append("- Qwen is reconstructed as synthetic predicted clusters by grouping GT-matched predicted pairs into one bucket per predicted relation type within each admission.")
    lines.append("- Primary comparator: raw pair-cluster purity and raw oracle-pair F1. Cluster-label metrics are secondary for prompts because the synthetic cluster identity is derived from the predicted relation type itself.")
    lines.append("- This is still conditional on GT-matched predicted pairs; it is a clustering-task comparison, not full end-to-end retrieval evaluation.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| System | Scope | Admissions | GT-matched pairs | Clusters | Raw purity | Raw oracle P/R/F1 | Cluster-label macro P/R/F1 | Corpus cluster-label P/R/F1 | Cluster accuracy |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |")
    for row in summary_rows:
        lines.append(
            "| {system} | {scope} | {n_adm} | {n_pairs} | {n_clusters} | {raw_purity} | {raw_p} / {raw_r} / {raw_f1} | {macro_p} / {macro_r} / {macro_f1} | {cluster_p} / {cluster_r} / {cluster_f1} | {cluster_acc} |".format(
                system=row.get("system_name", ""),
                scope=row.get("scope", ""),
                n_adm=row.get("n_admissions", 0),
                n_pairs=row.get("n_gt_matched_pairs", 0),
                n_clusters=row.get("n_clusters", 0),
                raw_purity=_fmt(_safe_float(row.get("raw_pair_cluster_purity"))),
                raw_p=_fmt(_safe_float(row.get("raw_pair_oracle_precision"))),
                raw_r=_fmt(_safe_float(row.get("raw_pair_oracle_recall"))),
                raw_f1=_fmt(_safe_float(row.get("raw_pair_oracle_f1"))),
                macro_p=_fmt(_safe_float(row.get("cluster_label_macro_precision"))),
                macro_r=_fmt(_safe_float(row.get("cluster_label_macro_recall"))),
                macro_f1=_fmt(_safe_float(row.get("cluster_label_macro_f1"))),
                cluster_p=_fmt(_safe_float(row.get("cluster_label_precision"))),
                cluster_r=_fmt(_safe_float(row.get("cluster_label_recall"))),
                cluster_f1=_fmt(_safe_float(row.get("cluster_label_f1"))),
                cluster_acc=_fmt(_safe_float(row.get("cluster_label_accuracy"))),
            )
        )

    return "\n".join(lines) + "\n"


def build_cluster_fairness_report(fairness_rows: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Relationship Clustering Audit")
    lines.append("")
    lines.append("This audit compares Qwen and LOKI on the same recovered-pair clustering task rather than the pair-label proxy.")
    lines.append("Primary comparable view: Qwen raw-oracle pair P/R/F1 against LOKI cluster-macro P/R/F1.")
    lines.append("Secondary view: Qwen Accuracy, Cluster Purity, and ARI are scaled by raw-oracle recall before being compared against LOKI's original clustering metrics.")
    lines.append("")
    lines.append("## Audit Summary")
    lines.append("")
    lines.append("| Prompt | Slice | Admissions | Prompt pairs | LOKI pairs | Prompt clusters | LOKI clusters | Qwen comparable P/R/F1 | LOKI macro P/R/F1 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in fairness_rows:
        lines.append(
            "| {prompt} | {scope} | {n_adm} | {prompt_pairs} | {loki_pairs} | {prompt_clusters} | {loki_clusters} | {prompt_p} / {prompt_r} / {prompt_f1} | {loki_p} / {loki_r} / {loki_f1} |".format(
                prompt=row.get("prompt_system_name", ""),
                scope=row.get("scope", ""),
                n_adm=row.get("n_admissions", 0),
                prompt_pairs=row.get("prompt_n_gt_matched_pairs", 0),
                loki_pairs=row.get("loki_n_gt_matched_pairs", 0),
                prompt_clusters=row.get("prompt_n_clusters", 0),
                loki_clusters=row.get("loki_n_clusters", 0),
                prompt_p=_fmt(_safe_float(row.get("prompt_raw_pair_oracle_precision"))),
                prompt_r=_fmt(_safe_float(row.get("prompt_raw_pair_oracle_recall"))),
                prompt_f1=_fmt(_safe_float(row.get("prompt_raw_pair_oracle_f1"))),
                loki_p=_fmt(_safe_float(row.get("loki_cluster_label_macro_precision"))),
                loki_r=_fmt(_safe_float(row.get("loki_cluster_label_macro_recall"))),
                loki_f1=_fmt(_safe_float(row.get("loki_cluster_label_macro_f1"))),
            )
        )

    lines.append("")
    lines.append("## Secondary Metrics")
    lines.append("")
    lines.append("Qwen secondary metrics below are multiplied by raw-oracle recall so unsupported pair coverage does not get full credit.")
    lines.append("")
    lines.append("| Prompt | Slice | Qwen Accuracy | LOKI Accuracy | Qwen Purity | LOKI Purity | Qwen ARI | LOKI ARI |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in fairness_rows:
        lines.append(
            "| {prompt} | {scope} | {prompt_acc} | {loki_acc} | {prompt_purity} | {loki_purity} | {prompt_ari} | {loki_ari} |".format(
                prompt=row.get("prompt_system_name", ""),
                scope=row.get("scope", ""),
                prompt_acc=_fmt(_conservative_qwen_metric(row, "prompt_cluster_label_accuracy", "prompt_raw_pair_oracle_recall")),
                loki_acc=_fmt(_safe_float(row.get("loki_cluster_label_accuracy"))),
                prompt_purity=_fmt(_conservative_qwen_metric(row, "prompt_raw_pair_cluster_purity", "prompt_raw_pair_oracle_recall")),
                loki_purity=_fmt(_safe_float(row.get("loki_raw_pair_cluster_purity"))),
                prompt_ari=_fmt(_conservative_qwen_metric(row, "prompt_cluster_ari", "prompt_raw_pair_oracle_recall")),
                loki_ari=_fmt(_safe_float(row.get("loki_cluster_ari"))),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for prompt_name in sorted({str(row.get("prompt_system_name", "")) for row in fairness_rows}):
        matched_all = next((row for row in fairness_rows if row.get("prompt_system_name") == prompt_name and row.get("scope") == "matched_all"), None)
        hard_overlap = next((row for row in fairness_rows if row.get("prompt_system_name") == prompt_name and row.get("scope") == "matched_multitype_overlap"), None)
        if matched_all is None:
            continue
        lines.append(f"### {prompt_name}")
        lines.append("")
        lines.append(
            "- On the all-overlap slice, the prompt has more GT-matched pairs than LOKI in "
            f"{int(matched_all.get('prompt_more_pairs_admissions') or 0)} admissions, fewer in "
            f"{int(matched_all.get('prompt_fewer_pairs_admissions') or 0)}, and equal support in "
            f"{int(matched_all.get('equal_pair_count_admissions') or 0)}."
        )
        lines.append(
            "- Cluster granularity is different as well: the prompt has more predicted clusters than LOKI in "
            f"{int(matched_all.get('prompt_more_clusters_admissions') or 0)} admissions, fewer in "
            f"{int(matched_all.get('prompt_fewer_clusters_admissions') or 0)}, and equal cluster counts in "
            f"{int(matched_all.get('equal_cluster_count_admissions') or 0)}."
        )
        lines.append(
            "- Secondary all-overlap diagnostics are "
            f"Accuracy={_fmt(_conservative_qwen_metric(matched_all, 'prompt_cluster_label_accuracy', 'prompt_raw_pair_oracle_recall'))} vs {_fmt(_safe_float(matched_all.get('loki_cluster_label_accuracy')))}, "
            f"Purity={_fmt(_conservative_qwen_metric(matched_all, 'prompt_raw_pair_cluster_purity', 'prompt_raw_pair_oracle_recall'))} vs {_fmt(_safe_float(matched_all.get('loki_raw_pair_cluster_purity')))}, "
            f"ARI={_fmt(_conservative_qwen_metric(matched_all, 'prompt_cluster_ari', 'prompt_raw_pair_oracle_recall'))} vs {_fmt(_safe_float(matched_all.get('loki_cluster_ari')))}."
        )
        lines.append(
            "- Easy single-type admissions still inflate this surface for both systems: "
            f"{int(matched_all.get('prompt_single_type_admissions') or 0)}/{int(matched_all.get('n_admissions') or 0)} for the prompt and "
            f"{int(matched_all.get('loki_single_type_admissions') or 0)}/{int(matched_all.get('n_admissions') or 0)} for LOKI."
        )
        if hard_overlap is not None:
            lines.append(
                "- On the harder multi-type overlap slice, the comparable F1 view is "
                f"{_fmt(_safe_float(hard_overlap.get('prompt_raw_pair_oracle_f1')))} for the prompt "
                f"against {_fmt(_safe_float(hard_overlap.get('loki_cluster_label_macro_f1')))} for LOKI."
            )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This is a closer clustering-task comparison than the old type-bucket pair-label proxy because it preserves LOKI's actual cluster object and scores Qwen at the cluster level.")
    lines.append("- It is still conditional on GT-matched predicted pairs, so it should be read alongside pair-retrieval and typed materialization metrics rather than replacing them.")
    return "\n".join(lines) + "\n"


def build_relationship_visualization_gallery(
    dashboard_summary_rows: Sequence[Dict[str, Any]],
    fairness_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    viz_dir: Path,
) -> str:
    lines: List[str] = []
    lines.append("# Relationship Clustering and Typed Relationship Materialization Visualizations")
    lines.append("")
    lines.append("This page collects the default Relationship Clustering and Typed Relationship Materialization plots for quick inspection.")
    lines.append("Relationship Clustering figures focus on cluster-quality diagnostics, while Typed Relationship Materialization figures isolate the comparable type-assignment P/R/F1 scores.")
    lines.append("Shared comparison figures aggregate every prediction JSON under Pred/, and every figure is saved as both PNG and PDF with the same basename.")
    lines.append("Model-specific diagnostics remain separate where the admission alignment itself depends on the prediction file.")
    lines.append("")

    dashboard_by_system = {
        str(row.get("system_name", "")): row
        for row in dashboard_summary_rows
        if row.get("scope") == "full"
    }
    
    # Exclude LOKI systems from prompt_names
    prompt_names = sorted(
        system_name
        for system_name in dashboard_by_system
        if system_name and "loki" not in system_name.lower()
    )
    
    loki_names = sorted(
        system_name
        for system_name in dashboard_by_system
        if system_name and "loki" in system_name.lower()
    )
    
    for loki_name in loki_names:
        loki_dashboard = dashboard_by_system.get(loki_name)
        loki_dashboard_path = _relative_markdown_path(
            viz_dir / f"{_slug(loki_name)}_relationship_clustering_dashboard.png",
            output_dir,
        )
        if loki_dashboard is not None:
            lines.append(f"## {loki_name}")
            lines.append("")
            lines.append(
                f"- Original full-dashboard macro F1: "
                f"{_fmt(_safe_float(loki_dashboard.get('cluster_label_macro_f1')))}."
            )
            lines.append(
                f"- Original full-dashboard mean accuracy: "
                f"{_fmt(_safe_float(loki_dashboard.get('cluster_label_accuracy')))}."
            )
            lines.append(
                f"- Original final predicted clusters: "
                f"{int(loki_dashboard.get('n_final_clusters') or 0)} across "
                f"{int(loki_dashboard.get('n_admissions') or 0)} admissions."
            )
            lines.append("")
            lines.append(f"![{loki_name} Relationship Clustering dashboard]({loki_dashboard_path})")
            lines.append("")

    loki_cluster_quality_path = _relative_markdown_path(
        viz_dir / "loki_per_admission_relationship_clustering_quality.png",
        output_dir,
    )
    if len(loki_names) >= 2:
        lines.append("## LOKI Per-Admission Cluster Quality")
        lines.append("")
        lines.append(
            "- This companion figure keeps the original per-admission scatter style and combines the two LOKI labelers into one figure as separate plots."
        )
        lines.append(
            "- The two plots show per-admission cluster precision and recall, with one shared colorbar for cluster macro F1."
        )
        lines.append("")
        lines.append(f"![LOKI per-admission cluster quality]({loki_cluster_quality_path})")
        lines.append("")

    main_metrics_path = _relative_markdown_path(
        viz_dir / "all_models_main_comparison_metrics.png",
        output_dir,
    )
    semantic_metrics_path = _relative_markdown_path(
        viz_dir / "all_models_semantic_integration_metrics.png",
        output_dir,
    )
    semantic_slices_path = _relative_markdown_path(
        viz_dir / "all_models_semantic_integration_slices.png",
        output_dir,
    )
    relationship_metrics_path = _relative_markdown_path(
        viz_dir / "all_models_relationship_clustering_metrics.png",
        output_dir,
    )
    relationship_slices_path = _relative_markdown_path(
        viz_dir / "all_models_relationship_clustering_slices.png",
        output_dir,
    )
    compute_cost_path = _relative_markdown_path(
        viz_dir / "all_models_compute_cost.png",
        output_dir,
    )
    compute_cost_flat_path = _relative_markdown_path(
        viz_dir / "all_models_compute_cost_flat.png",
        output_dir,
    )
    compute_cost_broken_path = _relative_markdown_path(
        viz_dir / "all_models_compute_cost_broken_axis.png",
        output_dir,
    )
    compute_cost_side_by_side_path = _relative_markdown_path(
        viz_dir / "all_models_compute_cost_side_by_side.png",
        output_dir,
    )
    compute_cost_half_circle_path = _relative_markdown_path(
        viz_dir / "all_models_compute_cost_half_circle.png",
        output_dir,
    )
    data_quality_path = _relative_markdown_path(
        viz_dir / "all_models_data_quality.png",
        output_dir,
    )

    if prompt_names:
        lines.append("## Combined Comparison")
        lines.append("")
        lines.append(
            "- The main paper-ready comparison figure places Relationship Clustering Quality and Typed Relationship Materialization side by side in one compact layout with a shared legend."
        )
        lines.append("")
        lines.append(f"![Main comparison metrics for all models]({main_metrics_path})")
        lines.append("")
        lines.append("### Typed Relationship Materialization")
        lines.append("")
        lines.append(
            "- These figures isolate the comparable type-assignment metrics only: raw-oracle pair P/R/F1 for prompt-only systems versus cluster-label macro P/R/F1 for LOKI."
        )
        lines.append(
            "- Overlap slices use one shared admission intersection across all plotted models, so the typed relationship materialization comparison is shown on exactly the same support."
        )
        lines.append(
            "- Models included: " + ", ".join(prompt_names) + "."
        )
        lines.append("")
        lines.append(f"![Typed Relationship Materialization metrics for all models]({semantic_metrics_path})")
        lines.append("")
        if any(str(row.get("scope", "")) == "matched_all" or "matched_all" in str(row.get("scope", "")) for row in fairness_rows):
            lines.append(f"![Typed Relationship Materialization overlap slices for all models]({semantic_slices_path})")
            lines.append("")

        lines.append("### Relationship Clustering Quality")
        lines.append("")
        lines.append(
            "- These figures keep the secondary clustering diagnostics separate from typed relationship materialization: label accuracy, cluster purity, and ARI."
        )
        lines.append(
            "- Prompt-only systems remain conservatively scaled by raw-oracle recall for these quality diagnostics, matching the existing comparison policy."
        )
        lines.append("")
        lines.append(f"![Relationship Clustering quality metrics for all models]({relationship_metrics_path})")
        lines.append("")
        if any(str(row.get("scope", "")) == "matched_all" or "matched_all" in str(row.get("scope", "")) for row in fairness_rows):
            lines.append(f"![Relationship Clustering quality overlap slices for all models]({relationship_slices_path})")
            lines.append("")
        
        lines.append("### Resource Allocation & Compute Cost")
        lines.append("")
        lines.append(
            "- Average latency (seconds per admission) and token footprint (prompt vs completion tokens) for each model."
        )
        lines.append("")
        lines.append(
            "- The half-circle companion view compresses the radial design into four stacked execution profiles, preserving the inward token markers while saving horizontal space."
        )
        lines.append("")
        lines.append(f"![Half-circle compute cost comparison for all models]({compute_cost_half_circle_path})")
        lines.append("")
        lines.append(
            "- The broken-axis companion view separates LOKI's small preprocessing stages from the much larger labeling/runtime regime, while keeping token footprint in a dedicated lower band."
        )
        lines.append("")
        lines.append(f"![Broken-axis compute cost comparison for all models]({compute_cost_broken_path})")
        lines.append("")
        lines.append(
            "- The side-by-side companion view separates latency and token footprint into two coordinated panels, keeping the latency scale logarithmic while showing token usage directly by model."
        )
        lines.append("")
        lines.append(f"![Side-by-side compute cost comparison for all models]({compute_cost_side_by_side_path})")
        lines.append("")
        lines.append(
            "- The flattened companion view stacks the LOKI pipeline stages into one bar per model, giving a simpler paper-friendly comparison across the four systems."
        )
        lines.append("")
        lines.append(f"![Flattened compute cost comparison for all models]({compute_cost_flat_path})")
        lines.append("")
        lines.append(f"![Resource and Compute Cost for all models]({compute_cost_path})")
        lines.append("")
        lines.append("### Data Quality & Relational Integrity")
        lines.append("")
        lines.append(
            "- Relational integrity violations (Out-of-Bounds row/sentence references) and schema anomalies (dropped empty rows) across the corpus."
        )
        lines.append("")
        lines.append(f"![Data Quality and Relational Integrity for all models]({data_quality_path})")
        lines.append("")

    loki_dashboard = dashboard_by_system.get("LOKI+GPT-OSS 20B")
    for prompt_name in prompt_names:
        prompt_dashboard = dashboard_by_system.get(prompt_name)
        matched_all = next((row for row in fairness_rows if row.get("prompt_system_name") == prompt_name and "matched_all" in str(row.get("scope", ""))), None)
        hard_overlap = next((row for row in fairness_rows if row.get("prompt_system_name") == prompt_name and "multitype_overlap" in str(row.get("scope", ""))), None)
        if prompt_dashboard is None:
            continue

        slug = _slug(prompt_name)
        dashboard_path = _relative_markdown_path(viz_dir / f"{slug}_relationship_clustering_dashboard.png", output_dir)
        counts_path = _relative_markdown_path(viz_dir / f"{slug}_relationship_clustering_cluster_counts.png", output_dir)
        delta_path = _relative_markdown_path(viz_dir / f"{slug}_relationship_clustering_raw_oracle_f1_delta.png", output_dir)

        lines.append(f"## {prompt_name}")
        lines.append("")
        lines.append(
            "- Comparable P/R/F1 uses Qwen raw-oracle pair metrics against LOKI cluster-macro metrics: "
            f"{_fmt(_safe_float(prompt_dashboard.get('raw_pair_oracle_precision')))} / "
            f"{_fmt(_safe_float(prompt_dashboard.get('raw_pair_oracle_recall')))} / "
            f"{_fmt(_safe_float(prompt_dashboard.get('raw_pair_oracle_f1')))} for {prompt_name} vs "
            f"{_fmt(_safe_float((loki_dashboard or {}).get('cluster_label_macro_precision')))} / "
            f"{_fmt(_safe_float((loki_dashboard or {}).get('cluster_label_macro_recall')))} / "
            f"{_fmt(_safe_float((loki_dashboard or {}).get('cluster_label_macro_f1')))} for LOKI (GPT-OSS)."
        )
        lines.append(
            "- Secondary diagnostics scale Qwen by raw-oracle recall before comparing against LOKI: "
            f"Accuracy={_fmt(_conservative_qwen_metric(prompt_dashboard, 'cluster_label_accuracy'))} vs {_fmt(_safe_float((loki_dashboard or {}).get('cluster_label_accuracy')))}, "
            f"Purity={_fmt(_conservative_qwen_metric(prompt_dashboard, 'raw_pair_cluster_purity'))} vs {_fmt(_safe_float((loki_dashboard or {}).get('raw_pair_cluster_purity')))}, "
            f"ARI={_fmt(_conservative_qwen_metric(prompt_dashboard, 'cluster_ari'))} vs {_fmt(_safe_float((loki_dashboard or {}).get('cluster_ari')))}."
        )
        if matched_all is not None:
            lines.append(
                "- Overlap slices use the same comparable mapping, with hard-slice F1 at "
                f"{_fmt(_safe_float((hard_overlap or {}).get('prompt_raw_pair_oracle_f1')))} for {prompt_name} vs "
                f"{_fmt(_safe_float((hard_overlap or {}).get('loki_cluster_label_macro_f1')))} for LOKI."
            )
        lines.append("")
        lines.append("### Prompt Dashboard Diagnostic")
        lines.append("")
        lines.append(f"![Relationship Clustering dashboard for {prompt_name}]({dashboard_path})")
        lines.append("")
        lines.append("### Cluster Count Diagnostic")
        lines.append("")
        lines.append(f"![Relationship Clustering cluster counts for {prompt_name}]({counts_path})")
        lines.append("")
        lines.append("### Raw Oracle F1 Admission Deltas")
        lines.append("")
        lines.append(f"![Relationship Clustering raw oracle F1 deltas for {prompt_name}]({delta_path})")
        lines.append("")

    return "\n".join(lines) + "\n"


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        print(f"Visualization skipped (missing library): {exc}")
        return None


def _save_plot_outputs(
    fig: Any,
    out_path: Path,
    dpi: int = 220,
    bbox_inches: str = "tight",
    facecolor: str = "white",
) -> None:
    png_path = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
    fig.savefig(pdf_path, bbox_inches=bbox_inches, facecolor=facecolor)


def _save_plot_outputs_crop_vertical_whitespace(
    fig: Any,
    out_path: Path,
    dpi: int = 220,
    artists: Optional[Sequence[Any]] = None,
    facecolor: str = "white",
) -> None:
    from matplotlib.transforms import Bbox

    png_path = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    pdf_path = png_path.with_suffix(".pdf")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if artists:
        content_bboxes = []
        for artist in artists:
            if artist is None:
                continue
            try:
                bbox = artist.get_tightbbox(renderer)
            except Exception:
                bbox = None
            if bbox is None:
                try:
                    bbox = artist.get_window_extent(renderer)
                except Exception:
                    bbox = None
            if bbox is not None:
                content_bboxes.append(bbox)
        if content_bboxes:
            tight_bbox = Bbox.union(content_bboxes).transformed(fig.dpi_scale_trans.inverted())
        else:
            tight_bbox = fig.get_tightbbox(renderer)
    else:
        tight_bbox = fig.get_tightbbox(renderer)
    fig_width, fig_height = fig.get_size_inches()
    pad_x = 0
    pad_y = 0
    cropped_bbox = Bbox.from_extents(
        max(0.0, tight_bbox.x0 - pad_x),
        max(0.0, tight_bbox.y0 - pad_y),
        min(fig_width, tight_bbox.x1 + pad_x),
        min(fig_height, tight_bbox.y1 + pad_y),
    )

    fig.savefig(png_path, dpi=dpi, bbox_inches=cropped_bbox, facecolor=facecolor)
    fig.savefig(pdf_path, bbox_inches=cropped_bbox, facecolor=facecolor)


def _remove_plot_outputs(out_path: Path) -> None:
    png_path = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    pdf_path = png_path.with_suffix(".pdf")
    for path in (png_path, pdf_path):
        if path.exists():
            path.unlink()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "plot"


def _annotate_bars(ax: Any, bars: Sequence[Any]) -> None:
    _annotate_bars_with_values(ax, bars)


def _annotate_bars_with_values(
    ax: Any,
    bars: Sequence[Any],
    values: Optional[Sequence[Optional[float]]] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    fontsize: int = 8,
    text_color: str = "#1f2937",
    rotation: float = 0.0,
    inside_bar: bool = False,
) -> None:
    axis_y_min, axis_y_max = ax.get_ylim()
    resolved_y_min = float(axis_y_min if y_min is None else y_min)
    resolved_y_max = float(axis_y_max if y_max is None else y_max)
    y_range = max(resolved_y_max - resolved_y_min, 1e-6)
    label_margin = 0.05 * y_range

    for index, bar in enumerate(bars):
        value = float(bar.get_height()) if values is None else values[index]
        x_pos = bar.get_x() + bar.get_width() / 2.0
        if value is None:
            ax.text(
                x_pos,
                max(0.02, resolved_y_min + label_margin),
                "N/A",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="#6b7280",
                style="italic",
            )
            continue

        numeric_value = float(value)
        if inside_bar and numeric_value >= 0:
            label_y = max(resolved_y_min + label_margin, numeric_value * 0.5)
            vertical_alignment = "center"
        else:
            label_y = min(numeric_value + 0.02, resolved_y_max - label_margin) if numeric_value >= 0 else max(numeric_value - 0.02, resolved_y_min + label_margin)
            vertical_alignment = "bottom" if numeric_value >= 0 else "top"
        ax.text(
            x_pos,
            label_y,
            _fmt(numeric_value),
            ha="center",
            va=vertical_alignment,
            fontsize=fontsize,
            color=text_color,
            rotation=rotation,
        )


def _safe_row_metric_values(
    rows: Sequence[Dict[str, Any]],
    key: str,
    default: float = 0.0,
) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = _safe_float(row.get(key))
        values.append(default if value is None else float(value))
    return values


def _spread_overlapping_points(
    x_values: Sequence[float],
    y_values: Sequence[float],
    radius: float = 0.025,
) -> Tuple[List[float], List[float]]:
    spread_x = [float(value) for value in x_values]
    spread_y = [float(value) for value in y_values]
    duplicate_groups: Dict[Tuple[float, float], List[int]] = defaultdict(list)

    for index, (x_value, y_value) in enumerate(zip(spread_x, spread_y)):
        duplicate_groups[(round(x_value, 4), round(y_value, 4))].append(index)

    for indices in duplicate_groups.values():
        if len(indices) <= 1:
            continue
        group_radius = min(radius, 0.01 + 0.004 * len(indices))
        base_x = min(max(spread_x[indices[0]], group_radius), 1.0 - group_radius)
        base_y = min(max(spread_y[indices[0]], group_radius), 1.0 - group_radius)
        for offset_index, point_index in enumerate(indices):
            angle = (2.0 * math.pi * offset_index) / len(indices)
            spread_x[point_index] = base_x + group_radius * math.cos(angle)
            spread_y[point_index] = base_y + group_radius * math.sin(angle)

    return spread_x, spread_y


def _iter_f1_curve_points(target_f1: float, n_points: int = 200) -> List[Tuple[float, float]]:
    start = max(target_f1 / 2.0 + 1e-3, 0.01)
    points: List[Tuple[float, float]] = []
    for index in range(n_points):
        x_value = start + ((1.0 - start) * index / max(n_points - 1, 1))
        denominator = max((2.0 * x_value) - target_f1, 1e-6)
        y_value = (target_f1 * x_value) / denominator
        if 0.0 <= y_value <= 1.0:
            points.append((x_value, y_value))
    return points


def visualize_loki_per_admission_semantic_integration(
    system_dashboards: Sequence[Tuple[str, Sequence[Dict[str, Any]], Dict[str, Any], str]],
    out_path: Path,
) -> None:
    plt = _load_pyplot()
    if plt is None or not system_dashboards:
        return

    non_empty_dashboards = [
        (system_name, rows, summary_row, cmap_name)
        for system_name, rows, summary_row, cmap_name in system_dashboards
        if rows
    ]
    if not non_empty_dashboards:
        return

    figure_width = 6.5 if len(non_empty_dashboards) >= 2 else 3.55
    figure_height = 3.4
    fig, axes = plt.subplots(1, len(non_empty_dashboards), figsize=(figure_width, figure_height), sharex=True, sharey=True)
    if len(non_empty_dashboards) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    point_color = "#2F5D8A"
    for axis_index, (ax, (system_name, rows, summary_row, cmap_name)) in enumerate(zip(axes, non_empty_dashboards)):
        display_name = _display_system_name(system_name)
        cluster_label_precisions = _safe_row_metric_values(rows, "cluster_label_macro_precision", default=0.0)
        cluster_label_recalls = _safe_row_metric_values(rows, "cluster_label_macro_recall", default=0.0)
        scatter_recalls, scatter_precisions = _spread_overlapping_points(
            cluster_label_recalls,
            cluster_label_precisions,
        )
        sizes = [max(12.0, 3.3 * int(_safe_float(row.get("n_gt_pairs")) or 0.0)) for row in rows]

        ax.set_facecolor("white")
        ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.85)
        scatter = ax.scatter(
            scatter_recalls,
            scatter_precisions,
            color=point_color,
            s=sizes,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
        )
        for target_f1 in (0.2, 0.4, 0.6):
            curve_points = _iter_f1_curve_points(target_f1)
            if not curve_points:
                continue
            x_values = [point[0] for point in curve_points]
            y_values = [point[1] for point in curve_points]
            ax.plot(x_values, y_values, linestyle="--", linewidth=1.0, color="#cbd5e1")
            ax.text(
                min(x_values[-1], 0.78),
                y_values[-1],
                f"F1={target_f1:.1f}",
                fontsize=9,
                color="#64748b",
            )

        ax.set_title(
            f"{display_name}",
            # f"{system_name}\Avg. Cluster Macro F1={_fmt(_safe_float(summary_row.get('cluster_label_macro_f1')))}",
            color="#111827",
            fontsize=10.5,
        )
        ax.set_xlabel("Cluster Recall", color="#111827", fontsize=10.5)
        ax.tick_params(colors="#4b5563", labelsize=8.5)
        if axis_index > 0:
            ax.tick_params(axis="y", left=True, labelleft=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.02)

    axes[0].set_ylabel("Cluster Precision", color="#111827", fontsize=10.5)
    # fig.suptitle(
    #     "LOKI Per-Admission Cluster Quality",
    #     color="#111827",
    #     fontsize=12,
    #     y=0.985,
    # )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.79, bottom=0.16, wspace=0.10)
    _save_plot_outputs_crop_vertical_whitespace(fig, out_path, dpi=220, facecolor="white")
    plt.close(fig)


def visualize_relationship_clustering_dashboard(
    system_name: str,
    rows: Sequence[Dict[str, Any]],
    summary_row: Dict[str, Any],
    out_path: Path,
    color_metric_key: str,
    color_metric_label: str,
) -> None:
    plt = _load_pyplot()
    if plt is None or not rows:
        return
    display_name = _display_system_name(system_name)

    cluster_label_precisions = _safe_row_metric_values(rows, "cluster_label_macro_precision", default=0.0)
    cluster_label_recalls = _safe_row_metric_values(rows, "cluster_label_macro_recall", default=0.0)
    color_values = _safe_row_metric_values(rows, color_metric_key, default=0.0)
    scatter_recalls, scatter_precisions = _spread_overlapping_points(
        cluster_label_recalls,
        cluster_label_precisions,
    )
    sizes = [max(36.0, 9.0 * int(_safe_float(row.get("n_gt_pairs")) or 0.0)) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(16.2, 6.4))
    fig.patch.set_facecolor("white")
    ax_scatter, ax_bar = axes

    ax_scatter.set_facecolor("white")
    ax_scatter.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.85)
    scatter = ax_scatter.scatter(
        scatter_recalls,
        scatter_precisions,
        c=color_values,
        s=sizes,
        cmap="viridis",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.6,
    )
    for target_f1 in (0.2, 0.4, 0.6):
        curve_points = _iter_f1_curve_points(target_f1)
        if not curve_points:
            continue
        x_values = [point[0] for point in curve_points]
        y_values = [point[1] for point in curve_points]
        ax_scatter.plot(x_values, y_values, linestyle="--", linewidth=1.0, color="#cbd5e1")
        ax_scatter.text(
            min(x_values[-1], 0.78),
            y_values[-1],
            f"F1={target_f1:.1f}",
            fontsize=9,
            color="#64748b",
        )

    ax_scatter.set_title("P/R (Macro) Per Admission for Relationship Clustering", color="#111827", fontsize=14)
    ax_scatter.set_xlabel("Cluster macro recall", color="#111827", fontsize=12)
    ax_scatter.set_ylabel("Cluster macro precision", color="#111827", fontsize=12)
    ax_scatter.tick_params(colors="#4b5563", labelsize=10)
    for spine in ax_scatter.spines.values():
        spine.set_edgecolor("#d1d5db")
    cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04)
    cbar.set_label(color_metric_label, color="#111827", fontsize=9)

    metric_names = [
        "Macro\nP",
        "Macro\nR",
        "Macro\nF1",
        "Mean\nAccuracy",
        "Cluster\nPurity",
        "Cluster\nARI",
        "Cluster\nSilhouette",
    ]
    metric_values = [
        _safe_float(summary_row.get("cluster_label_macro_precision")),
        _safe_float(summary_row.get("cluster_label_macro_recall")),
        _safe_float(summary_row.get("cluster_label_macro_f1")),
        _safe_float(summary_row.get("cluster_label_accuracy")),
        _safe_float(summary_row.get("raw_pair_cluster_purity")),
        _safe_float(summary_row.get("cluster_ari")),
        _safe_float(summary_row.get("cluster_silhouette")),
    ]
    plot_values = [0.0 if value is None else float(value) for value in metric_values]
    min_value = min(plot_values) if plot_values else 0.0
    max_value = max(plot_values) if plot_values else 0.0
    y_min = min(-1.0, min_value - 0.03) if min_value < 0.0 else 0.0
    y_max = max(1.08, max_value + 0.08)
    positions = list(range(len(metric_names)))

    ax_bar.set_facecolor("white")
    bars = ax_bar.bar(
        positions,
        plot_values,
        width=0.62,
        color=["#ef4444", "#dc2626", "#b91c1c", "#fb923c", "#14b8a6", "#334155", "#6366f1"],
        alpha=0.9,
    )
    ax_bar.set_ylim(y_min, y_max)
    ax_bar.set_title("Relationship Clustering Metrics", color="#111827", fontsize=14)
    ax_bar.set_ylabel("Score", color="#111827", fontsize=12)
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(metric_names, fontsize=10, rotation=0, ha="center")
    ax_bar.tick_params(axis="y", colors="#4b5563", labelsize=10)
    ax_bar.grid(True, axis="y", color="#e5e7eb", linewidth=0.8, alpha=0.85)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#d1d5db")

    y_range = max(y_max - y_min, 1e-6)
    label_margin = 0.05 * y_range
    for index, (bar, value) in enumerate(zip(bars, metric_values)):
        if value is None:
            ax_bar.text(index, max(0.02, y_min + label_margin), "N/A", ha="center", va="bottom", fontsize=9, color="#6b7280", style="italic")
            continue
        label_y = min(value + 0.02, y_max - label_margin) if value >= 0 else max(value - 0.02, y_min + label_margin)
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
            color="#111827",
        )

    title_counts = [f"Admissions={int(summary_row.get('n_admissions') or 0)}"]
    evaluated_clusters = summary_row.get("cluster_label_n_evaluated")
    correctly_labeled_clusters = summary_row.get("cluster_label_n_correct")
    if evaluated_clusters is not None:
        title_counts.append(f"Evaluated clusters={int(evaluated_clusters or 0)}")
    if correctly_labeled_clusters is not None:
        title_counts.append(f"Correctly labeled clusters={int(correctly_labeled_clusters or 0)}")
    title_counts.append(f"Final predicted clusters={int(summary_row.get('n_final_clusters') or 0)}")

    fig.suptitle(
        f"{display_name} Relationship Clustering Dashboard\n" + "  ".join(title_counts),
        color="#111827",
        fontsize=16,
        y=1.02,
    )
    fig.tight_layout()
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_ranked_delta_plot(
    prompt_name: str,
    rows: Sequence[Dict[str, Any]],
    metric_key: str,
    title: str,
    out_path: Path,
) -> None:
    plt = _load_pyplot()
    if plt is None or not rows:
        return

    filtered = [row for row in rows if _safe_float(row.get(metric_key)) is not None]
    if not filtered:
        return
    ranked = sorted(filtered, key=lambda row: float(_safe_float(row.get(metric_key)) or 0.0), reverse=True)
    y_vals = [float(_safe_float(row.get(metric_key)) or 0.0) for row in ranked]
    x_vals = list(range(len(ranked)))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    fig.patch.set_facecolor("white")
    ax.axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
    ax.plot(x_vals, y_vals, color=COMPARISON_DELTA_COLOR, linewidth=1.5)
    ax.scatter(x_vals, y_vals, color=COMPARISON_DELTA_COLOR, s=18)
    ax.set_xlabel("Admissions ranked by delta", fontsize=10, color="#1f2937")
    ax.set_ylabel(metric_key.replace("delta_", "Delta ").replace("_", " ").title(), fontsize=10, color="#1f2937")
    ax.set_title(f"{title}: {prompt_name} vs LOKI", fontsize=12, color="#111827")
    ax.grid(alpha=0.2)

    for index in list(range(min(3, len(ranked)))) + list(range(max(len(ranked) - 3, 0), len(ranked))):
        row = ranked[index]
        ax.text(x_vals[index], y_vals[index], str(row.get("admission_id", "")), fontsize=7, color="#374151", ha="center", va="bottom")

    fig.tight_layout()
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _system_styling(system_name: str) -> Tuple[str, str]:
    name_lower = system_name.lower()
    if "loki+gpt" in name_lower or (name_lower == "loki" and "gpt" in name_lower):
        return r"LOKI ($G$=GPT-OSS)", "#2563EB"
    elif "loki+qwen" in name_lower or (name_lower == "loki" and "qwen" in name_lower):
        return r"LOKI ($G$=Qwen-3.6)", "#7C3AED"
    elif "qwen-3.7" in name_lower or "qwen3.7" in name_lower:
        return "Qwen-3.7 (API)", "#DC2626"
    elif "qwen3.6-local" in name_lower:
        return "Qwen-3.6 (Local)", "#EA580C"
    return system_name, "#64748B"


def _display_system_name(system_name: str) -> str:
    return _system_styling(system_name)[0]


def _load_prompt_timing_summary(output_dir: Path, model_name: str) -> Dict[str, Any]:
    summary_path = output_dir / "Compute_Cost" / model_name / "inference_timing_summary.json"
    if summary_path.exists():
        try:
            return _load_json(summary_path)
        except Exception as e:
            print(f"Error loading timing summary for {model_name}: {e}")
    return {}


def _load_loki_materialization_summary(output_dir: Path, variant: str) -> Dict[str, Any]:
    candidate_paths = [
        output_dir / variant / "materialized_batch_summary_mimic.csv",
        ROOT.parent / "Batch_Materialization" / variant / "materialized_batch_summary_mimic.csv",
    ]
    for summary_path in candidate_paths:
        if not summary_path.exists():
            continue
        try:
            rows = _load_csv_rows(summary_path)
        except Exception as e:
            print(f"Error loading LOKI materialization summary from {summary_path}: {e}")
            continue
        if rows:
            return rows[0]
    return {}


def _loki_first_pass_labeling_seconds(
    phase_summary: Dict[str, Any],
    runtime_sec: float,
    join_path_sec: float,
    hdbscan_sec: float,
    fallback_first_pass_ratio: Optional[float] = None,
) -> float:
    first_pass = _safe_float(phase_summary.get("phase_e_cluster_labeling_sec"))
    if first_pass is not None:
        return float(first_pass)
    total_labeling = max(0.1, float(runtime_sec) - float(join_path_sec) - float(hdbscan_sec))
    if fallback_first_pass_ratio is not None:
        return max(0.1, total_labeling * float(fallback_first_pass_ratio))
    return total_labeling


def _load_prompt_data_quality(output_dir: Path, model_name: str) -> Dict[str, Any]:
    if "3.7" in model_name:
        path = output_dir / "Data_Quality" / "Qwen3.7-API" / "oob_repair_report.json"
    else:
        path = output_dir / "Data_Quality" / "Qwen3.6-Local" / "oob_repair_report_Qwen3.6-Local.json"
    if path.exists():
        try:
            data = _load_json(path)
            if isinstance(data, dict) and "annotators" in data:
                return data["annotators"][0].get("overall_stats", {})
        except Exception as e:
            print(f"Error loading data quality report for {model_name}: {e}")
    return {}


def _ordered_full_dashboard_summary_rows(
    summary_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ordered_rows: List[Dict[str, Any]] = []
    for sys_key in ["loki+gpt", "loki+qwen", "qwen-3.7", "qwen3.6-local"]:
        for row in summary_rows:
            name = str(row.get("system_name", "")).lower()
            scope = str(row.get("scope", "")).lower()
            if scope == "full" and (sys_key in name or (sys_key == "qwen-3.7" and "qwen3.7" in name)):
                ordered_rows.append(row)
                break
    for row in summary_rows:
        if row.get("scope") == "full" and row not in ordered_rows:
            ordered_rows.append(row)
    return ordered_rows


def _system_display_rank(system_name: str) -> int:
    name = system_name.lower()
    if "loki+gpt" in name:
        return 0
    if "loki+qwen" in name:
        return 1
    if "qwen-3.7" in name:
        return 2
    if "qwen3.6" in name:
        return 3
    return 4


def _collect_cluster_metric_bar_systems(
    summary_rows: Sequence[Dict[str, Any]],
    metrics: Sequence[Tuple[str, str, str, bool]],
) -> List[Tuple[str, List[Optional[float]], str]]:
    systems: List[Tuple[str, List[Optional[float]], str]] = []
    ordered_rows = _ordered_full_dashboard_summary_rows(summary_rows)

    for row in ordered_rows:
        sys_name = str(row.get("system_name", ""))
        label, color = _system_styling(sys_name)
        is_loki = "loki" in sys_name.lower()

        values: List[Optional[float]] = []
        for _, prompt_key, loki_key, scale_prompt in metrics:
            key_to_use = loki_key if is_loki else prompt_key
            if not is_loki and scale_prompt:
                values.append(_conservative_qwen_metric(row, key_to_use))
            else:
                values.append(_safe_float(row.get(key_to_use)))
        systems.append((label, values, color))

    return systems


def _render_cluster_metric_bar_panel(
    ax: Any,
    systems: Sequence[Tuple[str, List[Optional[float]], str]],
    metrics: Sequence[Tuple[str, str, str, bool]],
    panel_title: str,
    show_ylabel: bool,
    title_fontsize: float = 10.5,
    title_fontweight: str = "normal",
    tick_fontsize: float = 9.0,
    label_fontsize: float = 10.0,
    annotation_fontsize: float = 7.5,
    annotation_text_color: str = "#1f2937",
    annotation_rotation: float = 0.0,
    annotation_inside_bar: bool = False,
    metric_spacing: float = 0.82,
) -> Tuple[List[Any], List[str]]:
    if not systems:
        return [], []

    x = [index * metric_spacing for index in range(len(metrics))]
    width = min(0.68 / max(len(systems), 1), 0.13)
    offset_center = (len(systems) - 1) / 2.0

    all_plot_values: List[float] = []
    bar_groups: List[Tuple[Any, List[Optional[float]]]] = []
    for system_index, (system_name, actual_values, color) in enumerate(systems):
        plot_values = [0.0 if value is None else float(value) for value in actual_values]
        positions = [idx + (system_index - offset_center) * width for idx in x]
        bars = ax.bar(positions, plot_values, width=width * 0.92, color=color, label=system_name, zorder=3)
        bar_groups.append((bars, actual_values))
        all_plot_values.extend(plot_values)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _, _ in metrics], fontsize=tick_fontsize, color="#1f2937")
    min_value = min(all_plot_values) if all_plot_values else 0.0
    max_value = max(all_plot_values) if all_plot_values else 0.0
    y_min = min(-1.0, min_value - 0.03) if min_value < 0.0 else 0.0
    y_max = max(1.08, max_value + 0.08)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="y", labelsize=tick_fontsize, colors="#374151")
    if show_ylabel:
        ax.set_ylabel("Score", fontsize=label_fontsize, color="#1f2937")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.28, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_title(panel_title, fontsize=title_fontsize, fontweight=title_fontweight, color="#111827", pad=5)
    handles, labels = ax.get_legend_handles_labels()
    for bars, actual_values in bar_groups:
        _annotate_bars_with_values(
            ax,
            bars,
            actual_values,
            y_min=y_min,
            y_max=y_max,
            fontsize=annotation_fontsize,
            text_color=annotation_text_color,
            rotation=annotation_rotation,
            inside_bar=annotation_inside_bar,
        )
    return handles, labels


def _visualize_cluster_metric_bars(
    summary_rows: Sequence[Dict[str, Any]],
    metrics: Sequence[Tuple[str, str, str, bool]],
    title: str,
    out_path: Path,
) -> None:
    plt = _load_pyplot()
    if plt is None or not summary_rows:
        return

    systems = _collect_cluster_metric_bar_systems(summary_rows, metrics)
    if not systems:
        return

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    fig.patch.set_facecolor("white")
    handles, labels = _render_cluster_metric_bar_panel(
        ax,
        systems,
        metrics,
        title,
        show_ylabel=True,
        title_fontsize=15,
        tick_fontsize=13,
        label_fontsize=13,
        annotation_fontsize=10,
    )
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=len(systems),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        fontsize=11,
        handlelength=1.5,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_combined_main_metric_bars(
    summary_rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    plt = _load_pyplot()
    if plt is None or not summary_rows:
        return

    semantic_metrics = [
        ("Macro P", "raw_pair_oracle_precision", "cluster_label_macro_precision", False),
        ("Macro R", "raw_pair_oracle_recall", "cluster_label_macro_recall", False),
        ("Macro F1", "raw_pair_oracle_f1", "cluster_label_macro_f1", False),
    ]
    clustering_metrics = [
        ("Label Acc.", "cluster_label_accuracy", "cluster_label_accuracy", True),
        ("Cluster Purity", "raw_pair_cluster_purity", "raw_pair_cluster_purity", True),
        ("ARI", "cluster_ari", "cluster_ari", True),
    ]
    semantic_systems = _collect_cluster_metric_bar_systems(summary_rows, semantic_metrics)
    clustering_systems = _collect_cluster_metric_bar_systems(summary_rows, clustering_metrics)
    if not semantic_systems or not clustering_systems:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.45), sharey=True)
    fig.patch.set_facecolor("white")
    handles, labels = _render_cluster_metric_bar_panel(
        axes[0],
        clustering_systems,
        clustering_metrics,
        "Relationship Clustering Quality",
        show_ylabel=True,
        title_fontsize=11.1,
        title_fontweight="bold",
        tick_fontsize=10.0,
        label_fontsize=10.8,
        annotation_fontsize=7.8,
        annotation_text_color="#FFFFFF",
        annotation_rotation=90.0,
        annotation_inside_bar=True,
        metric_spacing=0.66,
    )
    _render_cluster_metric_bar_panel(
        axes[1],
        semantic_systems,
        semantic_metrics,
        "Typed Relationship Materialization",
        show_ylabel=False,
        title_fontsize=11.1,
        title_fontweight="bold",
        tick_fontsize=10.0,
        label_fontsize=10.8,
        annotation_fontsize=7.8,
        annotation_text_color="#FFFFFF",
        annotation_rotation=90.0,
        annotation_inside_bar=True,
        metric_spacing=0.66,
    )
    axes[1].tick_params(axis="y", labelleft=False)
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        fontsize=9.2,
        handlelength=1.1,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    # fig.suptitle("LOKI vs Frontier LLMs", fontsize=11.6, color="#111827", y=0.975)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.20, top=0.80, wspace=0.10)
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_semantic_integration_metric_bars(
    summary_rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    _visualize_cluster_metric_bars(
        summary_rows,
        [
            ("Macro\nP", "raw_pair_oracle_precision", "cluster_label_macro_precision", False),
            ("Macro\nR", "raw_pair_oracle_recall", "cluster_label_macro_recall", False),
            ("Macro\nF1", "raw_pair_oracle_f1", "cluster_label_macro_f1", False),
        ],
        "Typed Relationship Materialization Comparison: LOKI vs Frontier LLMs",
        out_path,
    )


def visualize_relationship_clustering_metric_bars(
    summary_rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    _visualize_cluster_metric_bars(
        summary_rows,
        [
            ("Label\nAccuracy", "cluster_label_accuracy", "cluster_label_accuracy", True),
            ("Cluster\nPurity", "raw_pair_cluster_purity", "raw_pair_cluster_purity", True),
            ("ARI", "cluster_ari", "cluster_ari", True),
        ],
        "Relationship Clustering Quality Comparison: LOKI vs Frontier LLMs",
        out_path,
    )


def _visualize_cluster_fairness_slices(
    fairness_rows: Sequence[Dict[str, Any]],
    metrics: Sequence[Tuple[str, str, str, bool]],
    title: str,
    out_path: Path,
) -> None:
    plt = _load_pyplot()
    if plt is None or not fairness_rows:
        return

    slices = [
        ("All overlap", "combined_matched_all"),
        ("Hard multitype overlap", "combined_matched_multitype_overlap"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.0), sharey=True)
    fig.patch.set_facecolor("white")
    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    for ax, (slice_name, scope_name) in zip(axes, slices):
        slice_rows = [row for row in fairness_rows if str(row.get("scope", "")) == scope_name]
        if not slice_rows:
            ax.set_visible(False)
            continue

        x = list(range(len(metrics)))
        systems: List[Tuple[str, List[Optional[float]], str, bool]] = []

        loki_seen = set()
        for row in slice_rows:
            loki_name = str(row.get("loki_system_name", ""))
            if loki_name and loki_name not in loki_seen:
                loki_seen.add(loki_name)
                label, color = _system_styling(loki_name)
                values = [_safe_float(row.get(loki_key)) for _, _, loki_key, _ in metrics]
                systems.append((label, values, color, True))

        prompt_seen = set()
        for row in slice_rows:
            prompt_name = str(row.get("prompt_system_name", ""))
            if prompt_name and prompt_name not in prompt_seen:
                prompt_seen.add(prompt_name)
                label, color = _system_styling(prompt_name)
                values = []
                for _, prompt_key, _, scale_prompt in metrics:
                    if scale_prompt:
                        values.append(_conservative_qwen_metric(row, prompt_key, "prompt_raw_pair_oracle_recall"))
                    else:
                        values.append(_safe_float(row.get(prompt_key)))
                systems.append((label, values, color, False))

        systems.sort(key=lambda item: _system_display_rank(item[0]))
        n_series = len(systems)
        width = min(0.84 / max(n_series, 1), 0.18)
        offset_center = (n_series - 1) / 2.0

        for system_index, (system_name, actual_values, color, _is_loki) in enumerate(systems):
            plot_values = [0.0 if value is None else float(value) for value in actual_values]
            positions = [idx + (system_index - offset_center) * width for idx in x]
            bars = ax.bar(positions, plot_values, width=width * 0.94, color=color, label=system_name)
            _annotate_bars_with_values(ax, bars, actual_values, y_min=0.0, y_max=1.08, fontsize=8 if n_series > 4 else 9)

            if system_name not in legend_labels and len(bars) > 0:
                legend_handles.append(bars[0])
                legend_labels.append(system_name)

        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _, _, _ in metrics], fontsize=11, color="#1f2937")
        ax.set_ylim(0.0, 1.08)
        n_admissions_val = int(slice_rows[0].get("n_admissions") or 0) if slice_rows else 0
        ax.set_title(
            f"{slice_name}\n(shared admissions across all models, n={n_admissions_val})",
            fontsize=13,
            color="#111827",
        )
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Score", fontsize=12, color="#1f2937")
    if legend_handles:
        sorted_legend = sorted(
            zip(legend_labels, legend_handles),
            key=lambda item: _system_display_rank(item[0]),
        )
        fig.legend(
            [item[1] for item in sorted_legend],
            [item[0] for item in sorted_legend],
            frameon=False,
            ncol=len(sorted_legend),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            fontsize=11,
        )
    fig.suptitle(title, fontsize=15, color="#111827", y=1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_semantic_integration_fairness_slices(
    fairness_rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    _visualize_cluster_fairness_slices(
        fairness_rows,
        [
            ("Macro\nP", "prompt_raw_pair_oracle_precision", "loki_cluster_label_macro_precision", False),
            ("Macro\nR", "prompt_raw_pair_oracle_recall", "loki_cluster_label_macro_recall", False),
            ("Macro\nF1", "prompt_raw_pair_oracle_f1", "loki_cluster_label_macro_f1", False),
        ],
        "Typed Relationship Materialization Slices: All Models vs shared LOKI baselines",
        out_path,
    )


def visualize_relationship_clustering_fairness_slices(
    fairness_rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    _visualize_cluster_fairness_slices(
        fairness_rows,
        [
            ("Label\nAccuracy", "prompt_cluster_label_accuracy", "loki_cluster_label_accuracy", True),
            ("Cluster\nPurity", "prompt_raw_pair_cluster_purity", "loki_raw_pair_cluster_purity", True),
            ("ARI", "prompt_cluster_ari", "loki_cluster_ari", True),
        ],
        "Relationship Clustering Quality Slices: All Models vs shared LOKI baselines",
        out_path,
    )


def visualize_all_models_compute_cost(
    loki_gpt_dashboard_rows: List[Dict[str, Any]],
    loki_qwen_dashboard_rows: List[Dict[str, Any]],
    output_dir: Path,
    out_path: Path,
) -> None:
    import numpy as np
    plt = _load_pyplot()
    if plt is None:
        return

    loki_gpt_runtime = _mean(_safe_float(row.get("runtime_sec")) for row in loki_gpt_dashboard_rows) or 179.83
    loki_qwen_runtime = _mean(_safe_float(row.get("runtime_sec")) for row in loki_qwen_dashboard_rows) or 1842.35

    qwen_local_summary = _load_prompt_timing_summary(output_dir, "Qwen3.6-Local")
    qwen_api_summary = _load_prompt_timing_summary(output_dir, "Qwen-3.7")

    qwen_local_runtime = qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_inference_seconds", 89.86)
    qwen_api_runtime = qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_inference_seconds", 175.92)

    t_join_paths = 9.18
    t_hdbscan = 0.10
    t_labeling_gpt = max(0.1, loki_gpt_runtime - t_join_paths - t_hdbscan)
    t_labeling_qwen = max(0.1, loki_qwen_runtime - t_join_paths - t_hdbscan)

    loki_gpt_prompt_tokens = 7000
    loki_gpt_comp_tokens = 150
    loki_qwen_prompt_tokens = 7000
    loki_qwen_comp_tokens = 150

    qwen_local_prompt_tokens = qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_prompt_tokens", 10080)
    qwen_local_comp_tokens = qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_completion_tokens", 11999)

    qwen_api_prompt_tokens = qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_prompt_tokens", 10361)
    qwen_api_comp_tokens = qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_completion_tokens", 12726)

    # Categories list with multiline names for clean horizontal alignment
    categories = [
        {
            "name": "LOKI\nJoin-Path",
            "time": t_join_paths,
            "tokens": 0,
            "color": "#64b5f6",
            "group": "loki",
            "angle": 25
        },
        {
            "name": "HDBSCAN\nClustering",
            "time": t_hdbscan,
            "tokens": 0,
            "color": "#ba68c8",
            "group": "loki",
            "angle": 70
        },
        {
            "name": "GPT-OSS\n(Labeling)",
            "time": t_labeling_gpt,
            "tokens": loki_gpt_prompt_tokens + loki_gpt_comp_tokens,
            "color": "#2F5D8A",
            "group": "loki",
            "angle": 115
        },
        {
            "name": "Qwen-3.6\n(Labeling)",
            "time": t_labeling_qwen,
            "tokens": loki_qwen_prompt_tokens + loki_qwen_comp_tokens,
            "color": "#5C85AD",
            "group": "loki",
            "angle": 160
        },
        {
            "name": "Qwen-3.6\n(Local)",
            "time": qwen_local_runtime,
            "tokens": qwen_local_prompt_tokens + qwen_local_comp_tokens,
            "color": "#9BB7AE",
            "group": "baselines",
            "angle": 240
        },
        {
            "name": "Qwen-3.7\n(API)",
            "time": qwen_api_runtime,
            "tokens": qwen_api_prompt_tokens + qwen_api_comp_tokens,
            "color": "#C7886B",
            "group": "baselines",
            "angle": 295
        }
    ]

    base_r = 5.0
    h_max = 4.0
    
    def latency_to_height(t):
        t_min = 0.05
        t_max = 2000.0
        val = math.log10(max(t, t_min)) - math.log10(t_min)
        denom = math.log10(t_max) - math.log10(t_min)
        return (val / denom) * h_max

    def tokens_to_height(tok):
        tok_max = 30000.0
        return (tok / tok_max) * h_max

    fig = plt.figure(figsize=(7.5, 7.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("white")

    # Group background shading (spans full radius for neat sector design)
    theta_loki = np.linspace(10 * np.pi / 180, 175 * np.pi / 180, 200)
    ax.fill_between(theta_loki, 0, base_r + h_max + 0.3, color="#e0f2fe", alpha=0.35, zorder=1)
    
    theta_baselines = np.linspace(220 * np.pi / 180, 315 * np.pi / 180, 200)
    ax.fill_between(theta_baselines, 0, base_r + h_max + 0.3, color="#ffe4e6", alpha=0.35, zorder=1)

    # Label the sectors/groups text at the outer bounds
    ax.text(90 * np.pi / 180, base_r + h_max + 1.05, "LOKI", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#1e3a8a', zorder=2)
    ax.text(267.5 * np.pi / 180, base_r + h_max + 1.15, "Frontier LLMs", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#9f1239', zorder=2)

    # Label the innermost empty circle center
    ax.text(0, 0, "Token\nUsed", ha='center', va='center', fontsize=11, fontweight='bold', color='#475569', zorder=5)

    # Draw solid separating circle at base_r (separating inward tokens from outward latency) - 1pt thick
    ax.plot(
        np.linspace(0, 2 * np.pi, 200),
        [base_r] * 200,
        color="#1e293b",
        linestyle="-",
        linewidth=1.0,
        zorder=4
    )

    # Draw radial bars for Latency
    bar_width = 0.28  # width of radial bars in radians

    # Small angle offsets to separate the two Frontier model markers (avoid overlap)
    baseline_angle_offsets = {
        "Qwen3.6-Local": -5.0,
        "Qwen-3.7 (API)": 5.0,
    }

    # Small label-angle offsets for baseline latency labels to avoid overlap
    label_angle_offsets = {
        "Qwen3.6-Local": -3.0,
        "Qwen-3.7 (API)": 3.0,
    }

    for cat in categories:
        # apply a tiny angle offset for baseline categories to reduce overlap
        angle_deg = float(cat.get("angle", 0.0)) + baseline_angle_offsets.get(cat.get("name", ""), 0.0)
        angle_rad = angle_deg * np.pi / 180
        h_lat = latency_to_height(cat["time"])
        
        # Plot latency bar (outward)
        ax.bar(
            x=angle_rad,
            height=h_lat,
            bottom=base_r,
            width=bar_width,
            color=cat["color"],
            edgecolor="none",
            alpha=0.9,
            zorder=3
        )
        
        # Add latency text value near the bar tip, rotated parallel to the top of the bar
        val_str = f"{cat['time']:.2f}s" if cat['time'] < 1.0 else f"{cat['time']:.1f}s"
        r_text_lat = base_r + h_lat + 0.35

        # For baselines, nudge the label radially outward a bit and apply a tiny angular shift
        if cat.get("group") == "baselines":
            r_text_lat += 0.18

        # Calculate rotation to be parallel to the bar's top (perpendicular to radial line)
        rot = angle_deg - 90
        if rot > 90:
            rot -= 180
        elif rot < -90:
            rot += 180
        
        # compute optional small angular offset for the text (keeps text close to its bar)
        label_offset_deg = label_angle_offsets.get(cat.get("name", ""), 0.0)
        label_angle_rad = angle_rad + (label_offset_deg * np.pi / 180)
        label_fontsize = 10.0 if cat.get("group") == "baselines" else 11
        ax.text(
            label_angle_rad, r_text_lat, val_str,
            ha='center', va='center', fontsize=label_fontsize, fontweight='bold', color='#1e393b',
            rotation=rot, zorder=5
        )

        # Plot Token Footprint line and marker (inward into middle circle)
        h_tok = tokens_to_height(cat["tokens"])

        if cat["tokens"] > 0:
            # Draw dotted line from base_r inward to base_r - h_tok
            ax.plot(
                [angle_rad, angle_rad],
                [base_r, base_r - h_tok],
                color="#475569",
                linestyle=":",
                linewidth=1.8,
                zorder=4
            )

            # Draw circular dot marker. Use a smaller dot for Frontier baselines to reduce overlap.
            dot_color = "#ff7043"  # Vibrant Coral-Orange
            dot_size = 120
            if cat.get("group") == "baselines":
                dot_size = 90
            ax.scatter(
                angle_rad,
                base_r - h_tok,
                color=dot_color,
                s=dot_size,
                edgecolors="white",
                linewidths=2,
                zorder=5
            )

            tok_str = f"{cat['tokens']/1000:.1f}K"
            # Reduce padding for baseline token labels so they sit closer to markers
            r_text_tok = base_r - h_tok - (0.6 if cat.get("group") == "baselines" else 1.0)

            ax.text(
                angle_rad, r_text_tok, tok_str,
                ha='center', va='center', fontsize=10.5, fontweight='bold', color='#c2410c',
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=5
            )

        # Add outer category name labels (rotated parallel to the top of the bar)
        # Use the original angle (cat["angle"]) for names, not the offset angle_deg
        # Special padding for Qwen-3.6 (Labeling) which sits too close; others use standard padding
        if "Qwen-3.6\n(Labeling)" in cat.get("name", ""):
            r_name = base_r + h_max + 0.95
        else:
            r_name = base_r + h_max + 0.30
        
        original_angle_deg = cat['angle']
        original_angle_rad = original_angle_deg * np.pi / 180

        # Calculate rotation to be parallel to the bar's top (use original angle, not offset)
        rot_name = original_angle_deg - 90
        if rot_name > 90:
            rot_name -= 180
        elif rot_name < -90:
            rot_name += 180

        # Dynamically set vertical alignment based on quadrant to push text outwards
        if 0 <= original_angle_deg < 180:
            name_va = 'bottom'
        else:
            name_va = 'top'

        ax.text(
            original_angle_rad, r_name, cat["name"],
            ha='center', va=name_va, fontsize=11, fontweight='semibold', color='#334155',
            rotation=rot_name, zorder=5
        )

    # Set up radial grid for Latency (Outward, using 10^n scale with hidden 1s and 1000s labels)
    tick_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    tick_positions = [base_r + latency_to_height(t) for t in tick_values]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([])  # Hide default labels to prevent automatic placement
    ax.tick_params(axis='y', grid_color='#cbd5e1', grid_linestyle='--', grid_alpha=0.7)

    # Manually place y-tick labels (latency scale) rotated parallel to and inside the grid circles
    scale_angle_deg = 0.0
    scale_angle_rad = scale_angle_deg * np.pi / 180
    scale_rot = scale_angle_deg - 90
    if scale_rot > 90:
        scale_rot -= 180
    elif scale_rot < -90:
        scale_rot += 180
        
    tick_labels = ["0.1s", "1s", "10s", "100s", ""]
    for pos, label in zip(tick_positions, tick_labels):
        if label:
            # Place slightly inside the grid circle (subtracting 0.22 from the radius)
            ax.text(
                scale_angle_rad, pos + 0.22, label,
                ha='center', va='center', fontsize=8.0, color='#475569', fontweight='semibold',
                rotation=scale_rot, zorder=5
            )

    # Set up radial grid for Tokens (Inward, compact scale up to 30K)
    scale_tok_angle_deg = 180.0
    scale_tok_angle_rad = scale_tok_angle_deg * np.pi / 180
    scale_tok_rot = scale_tok_angle_deg - 90
    if scale_tok_rot > 90:
        scale_tok_rot -= 180
    elif scale_tok_rot < -90:
        scale_tok_rot += 180

    for tok_val in [10000, 20000, 30000]:
        r_val = base_r - tokens_to_height(tok_val)
        # Draw a thin circular grid line for this token value
        ax.plot(np.linspace(0, 2 * np.pi, 200), [r_val] * 200, color='#cbd5e1', linestyle=':', linewidth=0.8, zorder=2)
        # Add grid tick label rotated parallel and inside the circle (closer to center)
        ax.text(
            scale_tok_angle_rad, r_val + 0.7, f"{tok_val//1000}K",
            ha='center', va='center', fontsize=8.0, color='#94a3b8', fontweight='semibold',
            rotation=scale_tok_rot, zorder=5
        )

    # Hide angular ticks since we label categories directly
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_rlabel_position(0)  # Put radial labels on the 0 degrees line

    # Set radial limits (with generous margin at the outer edge for labels)
    ax.set_ylim(0, base_r + h_max + 1.45)

    # Create dummy elements for Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor='#2F5D8A', edgecolor='none', label='Execution Latency (Outward, Log-Scale Arc)'),
        Line2D([0], [0], marker='o', color='#475569', markerfacecolor='#ff7043', markeredgecolor='white',
               markersize=10, markeredgewidth=1.5, linestyle=':', label='Token Footprint (Inward, Linear-Scale Dot)')
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=1,
        frameon=False,
        fontsize=11.5,
        prop={'weight': 'semibold'}
    )

    # Title
    # plt.title("Compute Latency & LLM Token Consumption Profile", 
    #           fontsize=15, fontweight='bold', color='#0f172a', pad=25)

    fig.tight_layout()
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_all_models_compute_cost_half_circle(
    loki_gpt_dashboard_rows: List[Dict[str, Any]],
    loki_qwen_dashboard_rows: List[Dict[str, Any]],
    output_dir: Path,
    out_path: Path,
) -> None:
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt = _load_pyplot()
    if plt is None:
        return

    loki_gpt_phase_summary = _load_loki_materialization_summary(output_dir, "LOKI_Batch_mimic_GPT_OSS")
    loki_qwen_phase_summary = _load_loki_materialization_summary(output_dir, "loki_batch_mimic_Qwen-3.6")

    loki_gpt_runtime = _mean(_safe_float(row.get("runtime_sec")) for row in loki_gpt_dashboard_rows) or 179.83
    loki_qwen_runtime = _mean(_safe_float(row.get("runtime_sec")) for row in loki_qwen_dashboard_rows) or 1842.35

    qwen_local_summary = _load_prompt_timing_summary(output_dir, "Qwen3.6-Local")
    qwen_api_summary = _load_prompt_timing_summary(output_dir, "Qwen-3.7")

    qwen_local_runtime = qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_inference_seconds", 89.86)
    qwen_api_runtime = qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_inference_seconds", 175.92)

    t_join_paths = 9.18
    t_hdbscan = 0.10
    qwen_total_labeling_runtime = max(0.1, loki_qwen_runtime - t_join_paths - t_hdbscan)
    qwen_first_pass_ratio = (
        float(_safe_float(loki_qwen_phase_summary.get("phase_e_cluster_labeling_sec")) or 0.0) / qwen_total_labeling_runtime
        if _safe_float(loki_qwen_phase_summary.get("phase_e_cluster_labeling_sec")) is not None
        else None
    )
    t_labeling_gpt = _loki_first_pass_labeling_seconds(
        loki_gpt_phase_summary,
        loki_gpt_runtime,
        t_join_paths,
        t_hdbscan,
        fallback_first_pass_ratio=qwen_first_pass_ratio,
    )
    t_labeling_qwen = _loki_first_pass_labeling_seconds(
        loki_qwen_phase_summary,
        loki_qwen_runtime,
        t_join_paths,
        t_hdbscan,
    )
    loki_gpt_total_time = t_hdbscan + t_join_paths + t_labeling_gpt
    loki_qwen_total_time = t_hdbscan + t_join_paths + t_labeling_qwen

    loki_gpt_tokens = 7000 + 150
    loki_qwen_tokens = 7000 + 150
    qwen_local_tokens = (
        qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_prompt_tokens", 10080)
        + qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_completion_tokens", 11999)
    )
    qwen_api_tokens = (
        qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_prompt_tokens", 10361)
        + qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_completion_tokens", 12726)
    )

    profiles = [
        {
            "name": "LOKI\n(G=GPT-OSS)",
            "group": "loki",
            "angle": 150,
            "token_label_pad": 0.95,
            "stages": [
                ("HDBSCAN", t_hdbscan, "#A78BFA"),
                ("Join-Path", t_join_paths, "#93C5FD"),
                ("GPT-OSS Labeling", t_labeling_gpt, "#2563EB"),
            ],
            "total_time": loki_gpt_total_time,
            "tokens": loki_gpt_tokens,
        },
        {
            "name": "LOKI\n(G=Qwen-3.6)",
            "group": "loki",
            "angle": 118,
            "token_label_pad": 0.95,
            "name_pad": 0.18,
            "stages": [
                ("HDBSCAN", t_hdbscan, "#A78BFA"),
                ("Join-Path", t_join_paths, "#93C5FD"),
                ("Qwen-3.6 Labeling", t_labeling_qwen, "#2563EB"),
            ],
            "total_time": loki_qwen_total_time,
            "tokens": loki_qwen_tokens,
        },
        {
            "name": "Qwen-3.6",
            "group": "frontier",
            "angle": 62,
            "token_label_pad": 0.62,
            "token_angle_offset": -11.0,
            "token_label_fraction": 0.52,
            "stages": [
                ("Total Runtime", qwen_local_runtime, "#EA580C"),
            ],
            "total_time": qwen_local_runtime,
            "tokens": qwen_local_tokens,
        },
        {
            "name": "Qwen-3.7",
            "group": "frontier",
            "angle": 30,
            "token_label_pad": 0.62,
            "token_angle_offset": -11.0,
            "token_label_fraction": 0.52,
            "stages": [
                ("Total Runtime", qwen_api_runtime, "#DC2626"),
            ],
            "total_time": qwen_api_runtime,
            "tokens": qwen_api_tokens,
        },
    ]

    base_r = 5.0
    h_max = 4.0
    bar_width = 0.35

    def latency_to_height(t: float) -> float:
        t_min = 0.05
        t_max = 2000.0
        val = math.log10(max(t, t_min)) - math.log10(t_min)
        denom = math.log10(t_max) - math.log10(t_min)
        return (val / denom) * h_max

    def tokens_to_height(tok: float) -> float:
        tok_max = 30000.0
        return (tok / tok_max) * h_max

    fig = plt.figure(figsize=(8.0, 4.9))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("white")
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    theta_loki = np.linspace(95 * np.pi / 180, 180 * np.pi / 180, 200)
    ax.fill_between(theta_loki, 0, base_r + h_max + 0.35, color="#e0f2fe", alpha=0.35, zorder=1)

    theta_frontier = np.linspace(0, 85 * np.pi / 180, 200)
    ax.fill_between(theta_frontier, 0, base_r + h_max + 0.35, color="#ffe4e6", alpha=0.35, zorder=1)

    ax.text(135 * np.pi / 180, base_r + h_max + 3.35, "LOKI",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1e3a8a", rotation=50, rotation_mode="anchor", zorder=2)
    ax.text(50 * np.pi / 180, base_r + h_max + 2.50, "LLMs",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#9f1239", rotation=-45, rotation_mode="anchor", zorder=2)

    ax.plot(
        np.linspace(0, np.pi, 200),
        [base_r] * 200,
        color="#1e293b",
        linestyle="-",
        linewidth=1.0,
        zorder=4,
    )

    loki_token_label_specs: List[Tuple[float, float, float]] = []

    for profile in profiles:
        angle_deg = float(profile["angle"])
        angle_rad = math.radians(angle_deg)
        rot = angle_deg - 90
        if rot > 90:
            rot -= 180
        elif rot < -90:
            rot += 180

        cumulative_time = 0.0
        for stage_index, (_stage_name, stage_time, stage_color) in enumerate(profile["stages"]):
            lower_h = latency_to_height(cumulative_time)
            cumulative_time += float(stage_time)
            upper_h = latency_to_height(cumulative_time)
            segment_h = max(upper_h - lower_h, 0.0)
            if segment_h <= 0:
                continue
            ax.bar(
                x=angle_rad,
                height=segment_h,
                bottom=base_r + lower_h,
                width=bar_width,
                color=stage_color,
                edgecolor="white",
                linewidth=0.9,
                alpha=0.95,
                zorder=3,
            )

            if profile["group"] == "loki" and stage_index < len(profile["stages"]) - 1:
                ax.text(
                    angle_rad,
                    base_r + upper_h + (0.22 if stage_index == 0 else 0.3),
                    f"{cumulative_time:.1f}s",
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    fontweight="bold",
                    color=("white" if stage_index == 1 else "#1e393b"),
                    rotation=rot,
                    zorder=5,
                )

        h_total = latency_to_height(float(profile["total_time"]))
        runtime_text = (
            f"{float(profile['total_time']):.2f}s"
            if float(profile["total_time"]) < 1.0
            else f"{float(profile['total_time']):.1f}s"
        )

        ax.text(
            angle_rad,
            base_r + h_total + 0.34,
            runtime_text,
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color="#1e393b",
            rotation=rot,
            zorder=5,
        )

        h_tok = tokens_to_height(float(profile["tokens"]))
        ax.plot(
            [angle_rad, angle_rad],
            [base_r, base_r - h_tok],
            color="#475569",
            linestyle=":",
            linewidth=1.8,
            zorder=4,
        )
        ax.scatter(
            angle_rad,
            base_r - h_tok,
            color="#ff7043",
            s=104 if profile["group"] == "loki" else 92,
            edgecolors="white",
            linewidths=2,
            zorder=5,
        )
        if profile["group"] == "loki":
            loki_token_label_specs.append((angle_deg, h_tok, float(profile["token_label_pad"])))
        else:
            token_label_angle_deg = angle_deg + float(profile.get("token_angle_offset", 0.0))
            token_label_rot = token_label_angle_deg
            if token_label_rot > 180:
                token_label_rot -= 180
            elif token_label_rot < 0:
                token_label_rot += 180
            ax.text(
                math.radians(token_label_angle_deg),
                base_r - (h_tok * float(profile.get("token_label_fraction", 1.0))),
                f"{float(profile['tokens']) / 1000.0:.1f}K",
                ha="center",
                va="center",
                fontsize=10.2,
                fontweight="bold",
                color="#c2410c",
                rotation=token_label_rot,
                rotation_mode="anchor",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=5,
            )

        name_radius = base_r + h_max + (0.76 if profile["group"] == "loki" else 0.46) + float(profile.get("name_pad", 0.0))
        ax.text(
            angle_rad,
            name_radius,
            str(profile["name"]),
            ha="center",
            va="bottom",
            multialignment="center",
            fontsize=10.8,
            fontweight="semibold",
            color="#334155",
            rotation=rot,
            rotation_mode="anchor",
            zorder=5,
        )

    if loki_token_label_specs:
        loki_mid_angle_deg = sum(spec[0] for spec in loki_token_label_specs) / len(loki_token_label_specs)
        loki_h_tok = sum(spec[1] for spec in loki_token_label_specs) / len(loki_token_label_specs)
        loki_pad = sum(spec[2] for spec in loki_token_label_specs) / len(loki_token_label_specs)
        loki_rot = loki_mid_angle_deg - 90
        if loki_rot > 90:
            loki_rot -= 180
        elif loki_rot < -90:
            loki_rot += 180

        ax.text(
            math.radians(loki_mid_angle_deg),
            base_r - loki_h_tok - loki_pad,
            f"{7150 / 1000.0:.1f}K",
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color="#c2410c",
            rotation=loki_rot,
            rotation_mode="anchor",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            zorder=5,
        )

    tick_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    tick_positions = [base_r + latency_to_height(t) for t in tick_values]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", grid_color="#cbd5e1", grid_linestyle="--", grid_alpha=0.7)

    tick_labels = ["0.1s", "1s", "10s", "100s", ""]
    for pos, label in zip(tick_positions, tick_labels):
        if label:
            ax.text(
                np.pi / 2,
                pos + 0.18,
                label,
                ha="center",
                va="center",
                fontsize=8.0,
                color="#475569",
                fontweight="semibold",
                zorder=5,
            )

    for tok_val in [10000, 20000, 30000]:
        r_val = base_r - tokens_to_height(tok_val)
        ax.plot(np.linspace(0, np.pi, 200), [r_val] * 200, color="#cbd5e1", linestyle=":", linewidth=0.8, zorder=2)
        ax.text(
            np.pi / 2,
            r_val + 0.44,
            f"{tok_val // 1000}K",
            ha="center",
            va="center",
            fontsize=8.0,
            color="#94a3b8",
            fontweight="semibold",
            zorder=5,
        )

    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_ylim(0, base_r + h_max + 1.15)

    legend_elements = [
        Patch(facecolor="#93C5FD", edgecolor="none", label="Join-Path Representation"),
        Patch(facecolor="#A78BFA", edgecolor="none", label="Clustering"),
        Patch(facecolor="#2563EB", edgecolor="none", label="Labeling"),
        Line2D([0], [0], marker="o", color="#475569", markerfacecolor="#ff7043", markeredgecolor="white",
               markersize=9, markeredgewidth=1.5, linestyle=":", label="Token Usage (Inner Circle)"),
    ]
    stage_legend = ax.legend(
        handles=legend_elements[:3],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.18),
        ncol=3,
        frameon=False,
        fontsize=9.9,
        borderaxespad=0.0,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    ax.add_artist(stage_legend)
    token_legend = ax.legend(
        handles=legend_elements[3:],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.12),
        ncol=1,
        frameon=False,
        fontsize=9.9,
        borderaxespad=0.0,
        handletextpad=0.5,
    )

    ax.set_position([0.04, 0.06, 0.92, 0.90])
    _save_plot_outputs_crop_vertical_whitespace(
        fig,
        out_path,
        dpi=220,
        artists=[ax, stage_legend, token_legend],
        facecolor="white",
    )
    plt.close(fig)


def visualize_all_models_compute_cost_flat(
    loki_gpt_dashboard_rows: List[Dict[str, Any]],
    loki_qwen_dashboard_rows: List[Dict[str, Any]],
    output_dir: Path,
    out_path: Path,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt = _load_pyplot()
    if plt is None:
        return

    loki_gpt_phase_summary = _load_loki_materialization_summary(output_dir, "LOKI_Batch_mimic_GPT_OSS")
    loki_qwen_phase_summary = _load_loki_materialization_summary(output_dir, "loki_batch_mimic_Qwen-3.6")

    loki_gpt_runtime = _mean(_safe_float(row.get("runtime_sec")) for row in loki_gpt_dashboard_rows) or 179.83
    loki_qwen_runtime = _mean(_safe_float(row.get("runtime_sec")) for row in loki_qwen_dashboard_rows) or 1842.35

    qwen_local_summary = _load_prompt_timing_summary(output_dir, "Qwen3.6-Local")
    qwen_api_summary = _load_prompt_timing_summary(output_dir, "Qwen-3.7")

    qwen_local_runtime = qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_inference_seconds", 89.86)
    qwen_api_runtime = qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_inference_seconds", 175.92)

    gpt_join_path = (
        float(_safe_float(loki_gpt_phase_summary.get("phase_d_join_path_extraction_sec")) or 0.0249)
        + float(_safe_float(loki_gpt_phase_summary.get("phase_c_joint_encoding_sec")) or 6.5338)
    )
    gpt_hdbscan = float(_safe_float(loki_gpt_phase_summary.get("phase_e_hdbscan_clustering_sec")) or 0.0990)
    gpt_labeling = float(_safe_float(loki_gpt_phase_summary.get("phase_e_semantic_materialization_sec")) or max(0.1, loki_gpt_runtime - gpt_join_path - gpt_hdbscan))

    qwen_join_path = (
        float(_safe_float(loki_qwen_phase_summary.get("phase_d_join_path_extraction_sec")) or 0.0249)
        + float(_safe_float(loki_qwen_phase_summary.get("phase_c_joint_encoding_sec")) or 6.5338)
    )
    qwen_hdbscan = float(_safe_float(loki_qwen_phase_summary.get("phase_e_hdbscan_clustering_sec")) or 0.0990)
    qwen_labeling = float(_safe_float(loki_qwen_phase_summary.get("phase_e_semantic_materialization_sec")) or max(0.1, loki_qwen_runtime - qwen_join_path - qwen_hdbscan))

    loki_gpt_tokens = 7000 + 150
    loki_qwen_tokens = 7000 + 150
    qwen_local_tokens = (
        qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_prompt_tokens", 10080)
        + qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_completion_tokens", 11999)
    )
    qwen_api_tokens = (
        qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_prompt_tokens", 10361)
        + qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_completion_tokens", 12726)
    )

    loki_gpt_label, loki_gpt_color = _system_styling("LOKI+GPT-OSS 20B")
    loki_qwen_label, loki_qwen_color = _system_styling("LOKI+Qwen-3.6")
    qwen_api_label, qwen_api_color = _system_styling("Qwen-3.7")
    qwen_local_label, qwen_local_color = _system_styling("Qwen3.6-Local")

    model_profiles = [
        {
            "name": "LOKI\n($G$=GPT-OSS)",
            "group": "loki",
            "stages": [
                ("Join-Path Candidate Generation", gpt_join_path, "#BFDBFE"),
                ("HDBSCAN Clustering", gpt_hdbscan, "#60A5FA"),
                ("LLM Labeling", gpt_labeling, loki_gpt_color),
            ],
            "total_time": loki_gpt_runtime,
            "tokens": loki_gpt_tokens,
        },
        {
            "name": "LOKI\n($G$=Qwen-3.6)",
            "group": "loki",
            "stages": [
                ("Join-Path Candidate Generation", qwen_join_path, "#DDD6FE"),
                ("HDBSCAN Clustering", qwen_hdbscan, "#A78BFA"),
                ("LLM Labeling", qwen_labeling, loki_qwen_color),
            ],
            "total_time": loki_qwen_runtime,
            "tokens": loki_qwen_tokens,
        },
        {
            "name": "Qwen-3.7\n(API)",
            "group": "frontier",
            "stages": [
                ("Total Runtime", qwen_api_runtime, qwen_api_color),
            ],
            "total_time": qwen_api_runtime,
            "tokens": qwen_api_tokens,
        },
        {
            "name": "Qwen-3.6\n(Local)",
            "group": "frontier",
            "stages": [
                ("Total Runtime", qwen_local_runtime, qwen_local_color),
            ],
            "total_time": qwen_local_runtime,
            "tokens": qwen_local_tokens,
        },
    ]

    fig, ax = plt.subplots(figsize=(7.75, 5.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax2 = ax.twinx()

    ax.axvspan(-0.5, 1.5, color="#eff6ff", alpha=0.75, zorder=0)
    ax.axvspan(1.5, 3.5, color="#fff7ed", alpha=0.55, zorder=0)

    x_positions = list(range(len(model_profiles)))
    bar_width = 0.50
    runtime_base = 0.0

    for index, profile in enumerate(model_profiles):
        cumulative_bottom = runtime_base
        for stage_name, stage_time, stage_color in profile["stages"]:
            ax.bar(
                index,
                stage_time,
                width=bar_width,
                bottom=cumulative_bottom,
                color=stage_color,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
                alpha=0.94,
            )
            if profile["group"] == "loki":
                boundary_y = cumulative_bottom + stage_time
                ax.plot(
                    [index - (bar_width * 0.42), index + (bar_width * 0.42)],
                    [boundary_y, boundary_y],
                    color="white",
                    linewidth=1.2,
                    solid_capstyle="round",
                    zorder=4,
                )
            cumulative_bottom += stage_time

        runtime_text = f"{profile['total_time']:.2f}s" if profile["total_time"] < 1.0 else f"{profile['total_time']:.1f}s"
        ax.text(
            index,
            profile["total_time"] * 1.08,
            runtime_text,
            ha="center",
            va="bottom",
            fontsize=10.2,
            fontweight="semibold",
            color="#334155",
            rotation=18,
            zorder=5,
        )

        ax2.plot(
            [index, index],
            [0, float(profile["tokens"])],
            color="#475569",
            linestyle=":",
            linewidth=1.6,
            zorder=2,
        )
        ax2.scatter(
            index,
            float(profile["tokens"]),
            color="#ff7043",
            s=90,
            edgecolors="white",
            linewidths=1.8,
            zorder=5,
        )
        ax2.text(
            index,
            float(profile["tokens"]) + 1200,
            f"{float(profile['tokens'])/1000.0:.1f}K",
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color="#c2410c",
            zorder=5,
        )

    ax.set_yscale("symlog", linthresh=0.1, linscale=0.9, base=10)
    ax.set_ylim(0, 3000)
    latency_ticks = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    latency_tick_labels = ["0.01s", "0.03s", "0.1s", "0.3s", "1s", "3s", "10s", "30s", "100s", "300s", "1000s"]
    ax.set_yticks(latency_ticks)
    ax.set_yticklabels(latency_tick_labels, fontsize=9.8, color="#475569")
    ax.set_ylabel("Execution Latency", fontsize=11, color="#1f2937")
    ax.grid(axis="y", linestyle="--", color="#cbd5e1", alpha=0.82, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    ax2.set_ylim(0, 40000)
    ax2.set_yticks([10000, 20000, 30000, 40000])
    ax2.set_yticklabels(["10K", "20K", "30K", "40K"], fontsize=10.2, color="#94a3b8")
    ax2.set_ylabel("Token Footprint", fontsize=11, color="#475569", labelpad=22)
    ax2.tick_params(axis="y", pad=6)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([profile["name"] for profile in model_profiles], fontsize=10.8, color="#334155", fontweight="semibold")

    ax.text(0.25, 1.05, "LOKI", transform=ax.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#1e3a8a")
    ax.text(0.75, 1.05, "Frontier LLMs", transform=ax.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#9f1239")

    legend_elements = [
        Patch(facecolor=qwen_api_color, edgecolor="none", label="Frontier Total Runtime"),
        Line2D([0], [0], marker="o", color="#475569", markerfacecolor="#ff7043", markeredgecolor="white",
               markersize=9, markeredgewidth=1.5, linestyle=":", label="Token Footprint"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=10.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    fig.tight_layout()
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_all_models_compute_cost_broken_axis(
    loki_gpt_dashboard_rows: List[Dict[str, Any]],
    loki_qwen_dashboard_rows: List[Dict[str, Any]],
    output_dir: Path,
    out_path: Path,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt = _load_pyplot()
    if plt is None:
        return

    loki_gpt_phase_summary = _load_loki_materialization_summary(output_dir, "LOKI_Batch_mimic_GPT_OSS")
    loki_qwen_phase_summary = _load_loki_materialization_summary(output_dir, "loki_batch_mimic_Qwen-3.6")

    loki_gpt_runtime = _mean(_safe_float(row.get("runtime_sec")) for row in loki_gpt_dashboard_rows) or 179.83
    loki_qwen_runtime = _mean(_safe_float(row.get("runtime_sec")) for row in loki_qwen_dashboard_rows) or 1842.35

    qwen_local_summary = _load_prompt_timing_summary(output_dir, "Qwen3.6-Local")
    qwen_api_summary = _load_prompt_timing_summary(output_dir, "Qwen-3.7")

    qwen_local_runtime = qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_inference_seconds", 89.86)
    qwen_api_runtime = qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_inference_seconds", 175.92)

    gpt_join_path = (
        float(_safe_float(loki_gpt_phase_summary.get("phase_d_join_path_extraction_sec")) or 0.0249)
        + float(_safe_float(loki_gpt_phase_summary.get("phase_c_joint_encoding_sec")) or 6.5338)
    )
    gpt_hdbscan = float(_safe_float(loki_gpt_phase_summary.get("phase_e_hdbscan_clustering_sec")) or 0.0990)
    gpt_labeling = float(_safe_float(loki_gpt_phase_summary.get("phase_e_semantic_materialization_sec")) or max(0.1, loki_gpt_runtime - gpt_join_path - gpt_hdbscan))

    qwen_join_path = (
        float(_safe_float(loki_qwen_phase_summary.get("phase_d_join_path_extraction_sec")) or 0.0249)
        + float(_safe_float(loki_qwen_phase_summary.get("phase_c_joint_encoding_sec")) or 6.5338)
    )
    qwen_hdbscan = float(_safe_float(loki_qwen_phase_summary.get("phase_e_hdbscan_clustering_sec")) or 0.0990)
    qwen_labeling = float(_safe_float(loki_qwen_phase_summary.get("phase_e_semantic_materialization_sec")) or max(0.1, loki_qwen_runtime - qwen_join_path - qwen_hdbscan))

    loki_gpt_tokens = 7000 + 150
    loki_qwen_tokens = 7000 + 150
    qwen_local_tokens = (
        qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_prompt_tokens", 10080)
        + qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_completion_tokens", 11999)
    )
    qwen_api_tokens = (
        qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_prompt_tokens", 10361)
        + qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_completion_tokens", 12726)
    )

    loki_gpt_label, loki_gpt_color = _system_styling("LOKI+GPT-OSS 20B")
    loki_qwen_label, loki_qwen_color = _system_styling("LOKI+Qwen-3.6")
    _qwen_api_label, qwen_api_color = _system_styling("Qwen-3.7")
    _qwen_local_label, qwen_local_color = _system_styling("Qwen3.6-Local")

    model_profiles = [
        {
            "name": "LOKI\n($G$=GPT-OSS)",
            "group": "loki",
            "stages": [
                ("Join-Path Candidate Generation", gpt_join_path, "#BFDBFE"),
                ("HDBSCAN Clustering", gpt_hdbscan, "#60A5FA"),
                ("LLM Labeling", gpt_labeling, loki_gpt_color),
            ],
            "total_time": loki_gpt_runtime,
            "tokens": loki_gpt_tokens,
        },
        {
            "name": "LOKI\n($G$=Qwen-3.6)",
            "group": "loki",
            "stages": [
                ("Join-Path Candidate Generation", qwen_join_path, "#DDD6FE"),
                ("HDBSCAN Clustering", qwen_hdbscan, "#A78BFA"),
                ("LLM Labeling", qwen_labeling, loki_qwen_color),
            ],
            "total_time": loki_qwen_runtime,
            "tokens": loki_qwen_tokens,
        },
        {
            "name": "Qwen-3.7\n(API)",
            "group": "frontier",
            "stages": [
                ("Total Runtime", qwen_api_runtime, qwen_api_color),
            ],
            "total_time": qwen_api_runtime,
            "tokens": qwen_api_tokens,
        },
        {
            "name": "Qwen-3.6\n(Local)",
            "group": "frontier",
            "stages": [
                ("Total Runtime", qwen_local_runtime, qwen_local_color),
            ],
            "total_time": qwen_local_runtime,
            "tokens": qwen_local_tokens,
        },
    ]

    fig = plt.figure(figsize=(7.8, 6.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.1, 1.35, 1.15], hspace=0.06)
    ax_high = fig.add_subplot(gs[0])
    ax_low = fig.add_subplot(gs[1], sharex=ax_high)
    ax_tok = fig.add_subplot(gs[2], sharex=ax_high)
    fig.patch.set_facecolor("white")

    for axis in (ax_high, ax_low, ax_tok):
        axis.set_facecolor("white")
        axis.axvspan(-0.5, 1.5, color="#eff6ff", alpha=0.75, zorder=0)
        axis.axvspan(1.5, 3.5, color="#fff7ed", alpha=0.55, zorder=0)

    x_positions = list(range(len(model_profiles)))
    bar_width = 0.50

    def _draw_latency_panel(axis: Any) -> None:
        for index, profile in enumerate(model_profiles):
            cumulative_bottom = 0.0
            for _stage_name, stage_time, stage_color in profile["stages"]:
                axis.bar(
                    index,
                    stage_time,
                    width=bar_width,
                    bottom=cumulative_bottom,
                    color=stage_color,
                    edgecolor="white",
                    linewidth=0.9,
                    alpha=0.95,
                    zorder=3,
                )
                if profile["group"] == "loki":
                    boundary_y = cumulative_bottom + stage_time
                    axis.plot(
                        [index - (bar_width * 0.42), index + (bar_width * 0.42)],
                        [boundary_y, boundary_y],
                        color="white",
                        linewidth=1.2,
                        solid_capstyle="round",
                        zorder=4,
                    )
                cumulative_bottom += stage_time

    _draw_latency_panel(ax_high)
    _draw_latency_panel(ax_low)

    for index, profile in enumerate(model_profiles):
        runtime_text = f"{profile['total_time']:.2f}s" if profile["total_time"] < 1.0 else f"{profile['total_time']:.1f}s"
        ax_high.text(
            index,
            profile["total_time"] * 1.08,
            runtime_text,
            ha="center",
            va="bottom",
            fontsize=10.1,
            fontweight="semibold",
            color="#334155",
            rotation=18,
            zorder=5,
        )
        ax_tok.plot(
            [index, index],
            [0, float(profile["tokens"])],
            color="#475569",
            linestyle=":",
            linewidth=1.6,
            zorder=2,
        )
        ax_tok.scatter(
            index,
            float(profile["tokens"]),
            color="#ff7043",
            s=88,
            edgecolors="white",
            linewidths=1.8,
            zorder=5,
        )
        ax_tok.text(
            index,
            float(profile["tokens"]) + 1300,
            f"{float(profile['tokens'])/1000.0:.1f}K",
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color="#c2410c",
            zorder=5,
        )

    ax_high.set_yscale("log")
    ax_high.set_ylim(30, 3000)
    ax_high.set_yticks([30, 100, 300, 1000])
    ax_high.set_yticklabels(["30s", "100s", "300s", "1000s"], fontsize=10.0, color="#475569")
    ax_high.grid(axis="y", linestyle="--", color="#cbd5e1", alpha=0.82, linewidth=0.9, zorder=0)
    ax_high.set_axisbelow(True)
    ax_high.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax_low.set_ylim(0, 12)
    ax_low.set_yticks([0, 2, 5, 10])
    ax_low.set_yticklabels(["0s", "2s", "5s", "10s"], fontsize=10.0, color="#475569")
    ax_low.grid(axis="y", linestyle="--", color="#cbd5e1", alpha=0.82, linewidth=0.9, zorder=0)
    ax_low.set_axisbelow(True)
    ax_low.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_low.set_ylabel("Execution Latency", fontsize=11, color="#1f2937")

    ax_tok.set_ylim(0, 40000)
    ax_tok.set_yticks([0, 10000, 20000, 30000, 40000])
    ax_tok.set_yticklabels(["0", "10K", "20K", "30K", "40K"], fontsize=10.0, color="#94a3b8")
    ax_tok.grid(axis="y", linestyle=":", color="#e2e8f0", alpha=0.85, linewidth=0.85, zorder=0)
    ax_tok.set_axisbelow(True)
    ax_tok.set_ylabel("Token Footprint", fontsize=11, color="#475569")
    ax_tok.set_xticks(x_positions)
    ax_tok.set_xticklabels([profile["name"] for profile in model_profiles], fontsize=10.8, color="#334155", fontweight="semibold")

    ax_high.text(0.25, 1.05, "LOKI", transform=ax_high.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#1e3a8a")
    ax_high.text(0.75, 1.05, "Frontier LLMs", transform=ax_high.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#9f1239")

    # Broken-axis diagonal marks between high and low latency panels.
    d = 0.012
    kwargs_high = dict(transform=ax_high.transAxes, color="#475569", clip_on=False, linewidth=1.1)
    ax_high.plot((-d, +d), (-d, +d), **kwargs_high)
    ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs_high)
    kwargs_low = dict(transform=ax_low.transAxes, color="#475569", clip_on=False, linewidth=1.1)
    ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs_low)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_low)

    legend_elements = [
        Patch(facecolor="#BFDBFE", edgecolor="none", label="Join-Path Candidate Generation"),
        Patch(facecolor="#60A5FA", edgecolor="none", label="HDBSCAN Clustering"),
        Patch(facecolor=loki_gpt_color, edgecolor="none", label="LLM Labeling"),
        Patch(facecolor=qwen_api_color, edgecolor="none", label="Frontier Total Runtime"),
        Line2D([0], [0], marker="o", color="#475569", markerfacecolor="#ff7043", markeredgecolor="white",
               markersize=9, markeredgewidth=1.5, linestyle=":", label="Token Footprint"),
    ]
    ax_tok.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        ncol=3,
        frameon=False,
        fontsize=9.8,
    )

    for axis in (ax_high, ax_low, ax_tok):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_all_models_compute_cost_side_by_side(
    loki_gpt_dashboard_rows: List[Dict[str, Any]],
    loki_qwen_dashboard_rows: List[Dict[str, Any]],
    output_dir: Path,
    out_path: Path,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt = _load_pyplot()
    if plt is None:
        return

    loki_gpt_phase_summary = _load_loki_materialization_summary(output_dir, "LOKI_Batch_mimic_GPT_OSS")
    loki_qwen_phase_summary = _load_loki_materialization_summary(output_dir, "loki_batch_mimic_Qwen-3.6")

    loki_gpt_runtime = _mean(_safe_float(row.get("runtime_sec")) for row in loki_gpt_dashboard_rows) or 179.83
    loki_qwen_runtime = _mean(_safe_float(row.get("runtime_sec")) for row in loki_qwen_dashboard_rows) or 1842.35

    qwen_local_summary = _load_prompt_timing_summary(output_dir, "Qwen3.6-Local")
    qwen_api_summary = _load_prompt_timing_summary(output_dir, "Qwen-3.7")

    qwen_local_runtime = qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_inference_seconds", 89.86)
    qwen_api_runtime = qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_inference_seconds", 175.92)

    gpt_join_path = (
        float(_safe_float(loki_gpt_phase_summary.get("phase_d_join_path_extraction_sec")) or 0.0249)
        + float(_safe_float(loki_gpt_phase_summary.get("phase_c_joint_encoding_sec")) or 6.5338)
    )
    gpt_hdbscan = float(_safe_float(loki_gpt_phase_summary.get("phase_e_hdbscan_clustering_sec")) or 0.0990)
    gpt_labeling = float(_safe_float(loki_gpt_phase_summary.get("phase_e_semantic_materialization_sec")) or max(0.1, loki_gpt_runtime - gpt_join_path - gpt_hdbscan))

    qwen_join_path = (
        float(_safe_float(loki_qwen_phase_summary.get("phase_d_join_path_extraction_sec")) or 0.0249)
        + float(_safe_float(loki_qwen_phase_summary.get("phase_c_joint_encoding_sec")) or 6.5338)
    )
    qwen_hdbscan = float(_safe_float(loki_qwen_phase_summary.get("phase_e_hdbscan_clustering_sec")) or 0.0990)
    qwen_labeling = float(_safe_float(loki_qwen_phase_summary.get("phase_e_semantic_materialization_sec")) or max(0.1, loki_qwen_runtime - qwen_join_path - qwen_hdbscan))

    loki_gpt_tokens = 7000 + 150
    loki_qwen_tokens = 7000 + 150
    qwen_local_tokens = (
        qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_prompt_tokens", 10080)
        + qwen_local_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_completion_tokens", 11999)
    )
    qwen_api_tokens = (
        qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_prompt_tokens", 10361)
        + qwen_api_summary.get("summary_views", {}).get("successful_attempts", {}).get("average_completion_tokens", 12726)
    )

    loki_gpt_label, loki_gpt_color = _system_styling("LOKI+GPT-OSS 20B")
    loki_qwen_label, loki_qwen_color = _system_styling("LOKI+Qwen-3.6")
    qwen_api_label, qwen_api_color = _system_styling("Qwen-3.7")
    qwen_local_label, qwen_local_color = _system_styling("Qwen3.6-Local")

    model_profiles = [
        {
            "name": "LOKI\n($G$=GPT-OSS)",
            "group": "loki",
            "stages": [
                ("Join-Path Candidate Generation", gpt_join_path, "#BFDBFE"),
                ("HDBSCAN Clustering", gpt_hdbscan, "#60A5FA"),
                ("LLM Labeling", gpt_labeling, loki_gpt_color),
            ],
            "total_time": loki_gpt_runtime,
            "tokens": loki_gpt_tokens,
            "token_color": loki_gpt_color,
        },
        {
            "name": "LOKI\n($G$=Qwen-3.6)",
            "group": "loki",
            "stages": [
                ("Join-Path Candidate Generation", qwen_join_path, "#DDD6FE"),
                ("HDBSCAN Clustering", qwen_hdbscan, "#A78BFA"),
                ("LLM Labeling", qwen_labeling, loki_qwen_color),
            ],
            "total_time": loki_qwen_runtime,
            "tokens": loki_qwen_tokens,
            "token_color": loki_qwen_color,
        },
        {
            "name": "Qwen-3.7\n(API)",
            "group": "frontier",
            "stages": [("Total Runtime", qwen_api_runtime, qwen_api_color)],
            "total_time": qwen_api_runtime,
            "tokens": qwen_api_tokens,
            "token_color": qwen_api_color,
        },
        {
            "name": "Qwen-3.6\n(Local)",
            "group": "frontier",
            "stages": [("Total Runtime", qwen_local_runtime, qwen_local_color)],
            "total_time": qwen_local_runtime,
            "tokens": qwen_local_tokens,
            "token_color": qwen_local_color,
        },
    ]

    fig, (ax_latency, ax_tokens) = plt.subplots(1, 2, figsize=(8.6, 4.9))
    fig.patch.set_facecolor("white")
    for axis in (ax_latency, ax_tokens):
        axis.set_facecolor("white")
        axis.axvspan(-0.5, 1.5, color="#eff6ff", alpha=0.75, zorder=0)
        axis.axvspan(1.5, 3.5, color="#fff7ed", alpha=0.55, zorder=0)

    x_positions = list(range(len(model_profiles)))
    bar_width = 0.54

    for index, profile in enumerate(model_profiles):
        cumulative_bottom = 0.0
        for _stage_name, stage_time, stage_color in profile["stages"]:
            ax_latency.bar(
                index,
                stage_time,
                width=bar_width,
                bottom=cumulative_bottom,
                color=stage_color,
                edgecolor="white",
                linewidth=0.9,
                alpha=0.95,
                zorder=3,
            )
            cumulative_bottom += stage_time
        runtime_text = f"{profile['total_time']:.2f}s" if profile["total_time"] < 1.0 else f"{profile['total_time']:.1f}s"
        ax_latency.text(
            index,
            profile["total_time"] * 1.07,
            runtime_text,
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="semibold",
            color="#334155",
            rotation=18,
            zorder=5,
        )

        ax_tokens.bar(
            index,
            float(profile["tokens"]),
            width=bar_width,
            color=profile["token_color"],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
            zorder=3,
        )
        ax_tokens.text(
            index,
            float(profile["tokens"]) + 900,
            f"{float(profile['tokens'])/1000.0:.1f}K",
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color="#334155",
            zorder=5,
        )

    ax_latency.set_yscale("log")
    ax_latency.set_ylim(0.01, 3000)
    ax_latency.set_yticks([0.01, 0.1, 1, 10, 100, 1000])
    ax_latency.set_yticklabels(["0.01s", "0.1s", "1s", "10s", "100s", "1000s"], fontsize=10.0, color="#475569")
    ax_latency.set_ylabel("Execution Latency", fontsize=11, color="#1f2937")
    ax_latency.set_title("Latency (log scale)", fontsize=12, fontweight="bold", color="#111827", pad=10)
    ax_latency.grid(axis="y", linestyle="--", color="#cbd5e1", alpha=0.82, linewidth=0.9, zorder=0)
    ax_latency.set_axisbelow(True)

    ax_tokens.set_ylim(0, 40000)
    ax_tokens.set_yticks([0, 10000, 20000, 30000, 40000])
    ax_tokens.set_yticklabels(["0", "10K", "20K", "30K", "40K"], fontsize=10.0, color="#475569")
    ax_tokens.set_ylabel("Token Footprint", fontsize=11, color="#1f2937")
    ax_tokens.set_title("Token Usage", fontsize=12, fontweight="bold", color="#111827", pad=10)
    ax_tokens.grid(axis="y", linestyle=":", color="#dbe4ee", alpha=0.9, linewidth=0.9, zorder=0)
    ax_tokens.set_axisbelow(True)

    for axis in (ax_latency, ax_tokens):
        axis.set_xticks(x_positions)
        axis.set_xticklabels([profile["name"] for profile in model_profiles], fontsize=10.2, color="#334155", fontweight="semibold")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    ax_latency.text(0.25, 1.06, "LOKI", transform=ax_latency.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#1e3a8a")
    ax_latency.text(0.75, 1.06, "Frontier LLMs", transform=ax_latency.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#9f1239")
    ax_tokens.text(0.25, 1.06, "LOKI", transform=ax_tokens.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#1e3a8a")
    ax_tokens.text(0.75, 1.06, "Frontier LLMs", transform=ax_tokens.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#9f1239")

    legend_elements = [
        Patch(facecolor="#BFDBFE", edgecolor="none", label="Join-Path Candidate Generation"),
        Patch(facecolor="#60A5FA", edgecolor="none", label="HDBSCAN Clustering"),
        Patch(facecolor=loki_gpt_color, edgecolor="none", label="LLM Labeling"),
        Patch(facecolor=qwen_api_color, edgecolor="none", label="Frontier Total Runtime"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=9.8,
    )

    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.20, wspace=0.20)
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_all_models_data_quality(
    output_dir: Path,
    out_path: Path,
) -> None:
    plt = _load_pyplot()
    if plt is None:
        return

    qwen_local_stats = _load_prompt_data_quality(output_dir, "Qwen3.6-Local")
    qwen_api_stats = _load_prompt_data_quality(output_dir, "Qwen-3.7")

    qwen_local_oob = qwen_local_stats.get("total_oob_refs", 51)
    qwen_local_dropped = (
        int(qwen_local_stats.get("diag_rows_empty_dropped") or 74) +
        int(qwen_local_stats.get("med_rows_empty_dropped") or 454) +
        int(qwen_local_stats.get("rel_empty_evidence_dropped") or 24)
    )

    qwen_api_oob = qwen_api_stats.get("total_oob_refs", 62)
    qwen_api_dropped = (
        int(qwen_api_stats.get("diag_rows_empty_dropped") or 96) +
        int(qwen_api_stats.get("med_rows_empty_dropped") or 471) +
        int(qwen_api_stats.get("rel_empty_evidence_dropped") or 97)
    )

    system_names = [
        _display_system_name("LOKI+GPT-OSS 20B"),
        _display_system_name("LOKI+Qwen-3.6"),
        _display_system_name("Qwen-3.7 (API)"),
        _display_system_name("Qwen3.6-Local"),
    ]
    oob_violations = [0, 0, qwen_api_oob, qwen_local_oob]
    dropped_anomalies = [0, 0, qwen_api_dropped, qwen_local_dropped]
    
    colors = ["#2F5D8A", "#5C85AD", "#C7886B", "#9BB7AE"]

    fig, (ax_oob, ax_dropped) = plt.subplots(1, 2, figsize=(15.2, 5.8))
    fig.patch.set_facecolor("white")

    ax_oob.set_facecolor("white")
    ax_oob.grid(axis="y", color="#e5e7eb", linewidth=0.8, alpha=0.85)
    bars_oob = ax_oob.bar(system_names, oob_violations, color=colors, width=0.55, edgecolor="none", alpha=0.9)
    ax_oob.set_ylabel("Total Out-of-Bounds Row References", fontsize=11, color="#1f2937")
    ax_oob.set_title("Relational Integrity Violations (lower is better)", fontsize=13, color="#111827", pad=12)
    ax_oob.tick_params(axis="x", labelsize=10, rotation=15)
    ax_oob.tick_params(axis="y", labelsize=10)
    for spine in ax_oob.spines.values():
        spine.set_edgecolor("#d1d5db")

    for bar in bars_oob:
        height = bar.get_height()
        ax_oob.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1.2 if height > 0 else 1.2,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1f2937",
            weight="bold"
        )

    ax_dropped.set_facecolor("white")
    ax_dropped.grid(axis="y", color="#e5e7eb", linewidth=0.8, alpha=0.85)
    bars_dropped = ax_dropped.bar(system_names, dropped_anomalies, color=colors, width=0.55, edgecolor="none", alpha=0.9)
    ax_dropped.set_ylabel("Total Dropped Empty Rows (Anomalies)", fontsize=11, color="#1f2937")
    ax_dropped.set_title("Formatting Schema Anomalies (lower is better)", fontsize=13, color="#111827", pad=12)
    ax_dropped.tick_params(axis="x", labelsize=10, rotation=15)
    ax_dropped.tick_params(axis="y", labelsize=10)
    for spine in ax_dropped.spines.values():
        spine.set_edgecolor("#d1d5db")

    for bar in bars_dropped:
        height = bar.get_height()
        ax_dropped.text(
            bar.get_x() + bar.get_width()/2.,
            height + 15 if height > 0 else 10,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1f2937",
            weight="bold"
        )

    fig.suptitle("Data Quality & Relational Integrity: LOKI vs Frontier LLMs", fontsize=15, color="#111827", y=1.02)
    fig.tight_layout()
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_relationship_cluster_count_scatter(
    prompt_name: str,
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    plt = _load_pyplot()
    if plt is None or not rows:
        return

    x_vals = [float(_safe_float(row.get("loki_n_clusters")) or 0.0) for row in rows]
    y_vals = [float(_safe_float(row.get("n_clusters")) or 0.0) for row in rows]
    deltas = [float(_safe_float(row.get("delta_raw_pair_oracle_f1_vs_loki")) or 0.0) for row in rows]
    max_axis = max(max(x_vals, default=0.0), max(y_vals, default=0.0), 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    fig.patch.set_facecolor("white")
    scatter = ax.scatter(x_vals, y_vals, c=deltas, cmap="coolwarm", s=42, alpha=0.85, edgecolors="white", linewidths=0.4)
    ax.plot([0, max_axis], [0, max_axis], linestyle="--", color="#6b7280", linewidth=1.0)
    ax.set_xlim(0, max_axis * 1.05)
    ax.set_ylim(0, max_axis * 1.05)
    ax.set_xlabel("LOKI clusters per admission", fontsize=10, color="#1f2937")
    ax.set_ylabel(f"{prompt_name} clusters per admission", fontsize=10, color="#1f2937")
    ax.set_title(f"Relationship Clustering Granularity: {prompt_name} vs LOKI", fontsize=12, color="#111827")
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Raw oracle F1 delta vs LOKI", fontsize=9, color="#1f2937")
    fig.tight_layout()
    _save_plot_outputs(fig, out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-file", type=Path, default=DEFAULT_GT_FILE)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--loki-gpt-resume", type=Path, default=DEFAULT_LOKI_GPT_RESUME)
    parser.add_argument("--loki-gpt-results-csv", type=Path, default=DEFAULT_LOKI_GPT_RESULTS_CSV)
    parser.add_argument("--loki-qwen-resume", type=Path, default=DEFAULT_LOKI_QWEN_RESUME)
    parser.add_argument("--loki-qwen-results-csv", type=Path, default=DEFAULT_LOKI_QWEN_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gt_file.exists():
        raise FileNotFoundError(f"GT file not found: {args.gt_file}")
    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {args.pred_dir}")
    if not args.loki_gpt_resume.exists():
        raise FileNotFoundError(f"LOKI GPT resume state not found: {args.loki_gpt_resume}")
    if not args.loki_gpt_results_csv.exists():
        raise FileNotFoundError(f"LOKI GPT batch results CSV not found: {args.loki_gpt_results_csv}")
    if not args.loki_qwen_resume.exists():
        raise FileNotFoundError(f"LOKI Qwen resume state not found: {args.loki_qwen_resume}")
    if not args.loki_qwen_results_csv.exists():
        raise FileNotFoundError(f"LOKI Qwen batch results CSV not found: {args.loki_qwen_results_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)

    _resolve_rel_types_from_gt(args.gt_file)
    annotation_entries = load_annotation_entries(args.gt_file)

    # Load LOKI + GPT-OSS 20B
    loki_gpt_dashboard_rows = load_dashboard_rows(args.loki_gpt_results_csv)
    loki_gpt_dashboard_summary = summarize_dashboard_rows(
        system_name="LOKI+GPT-OSS 20B",
        source_file=args.loki_gpt_results_csv.name,
        scope="full",
        rows=loki_gpt_dashboard_rows,
    )
    loki_gpt_cluster_full = load_loki_cluster_evaluations(args.loki_gpt_resume)
    for evaluation in loki_gpt_cluster_full:
        evaluation.system_name = "LOKI+GPT-OSS 20B"
    loki_gpt_cluster_by_admission = {evaluation.admission_id: evaluation for evaluation in loki_gpt_cluster_full}

    # Load LOKI + Qwen-3.6
    loki_qwen_dashboard_rows = load_dashboard_rows(args.loki_qwen_results_csv)
    loki_qwen_dashboard_summary = summarize_dashboard_rows(
        system_name="LOKI+Qwen-3.6",
        source_file=args.loki_qwen_results_csv.name,
        scope="full",
        rows=loki_qwen_dashboard_rows,
    )
    loki_qwen_cluster_full = load_loki_cluster_evaluations(args.loki_qwen_resume)
    for evaluation in loki_qwen_cluster_full:
        evaluation.system_name = "LOKI+Qwen-3.6"
    loki_qwen_cluster_by_admission = {evaluation.admission_id: evaluation for evaluation in loki_qwen_cluster_full}

    # For compatibility/baseline comparisons where a single dictionary is needed,
    # we default loki_cluster_by_admission to the primary one (GPT-OSS).
    loki_cluster_by_admission = loki_gpt_cluster_by_admission
    loki_variants_to_compare = [
        ("LOKI+GPT-OSS 20B", loki_gpt_cluster_by_admission, args.loki_gpt_resume),
        ("LOKI+Qwen-3.6", loki_qwen_cluster_by_admission, args.loki_qwen_resume),
    ]

    dashboard_summary_rows: List[Dict[str, Any]] = [loki_gpt_dashboard_summary, loki_qwen_dashboard_summary]
    dashboard_per_admission_rows: List[Dict[str, Any]] = [
        {
            "system_name": "LOKI+GPT-OSS 20B",
            "source_file": args.loki_gpt_results_csv.name,
            **row,
        }
        for row in loki_gpt_dashboard_rows
    ] + [
        {
            "system_name": "LOKI+Qwen-3.6",
            "source_file": args.loki_qwen_results_csv.name,
            **row,
        }
        for row in loki_qwen_dashboard_rows
    ]
    cluster_summary_rows: List[Dict[str, Any]] = []
    cluster_per_admission_rows: List[Dict[str, Any]] = []
    cluster_fairness_rows: List[Dict[str, Any]] = []
    prompt_cluster_by_name: Dict[str, Dict[str, AdmissionClusterEvaluation]] = {}
    prompt_source_files: Dict[str, str] = {}
    shared_admission_ids_by_prompt: Dict[str, Set[str]] = {}
    hard_admission_ids_by_prompt: Dict[str, Set[str]] = {}

    visualize_relationship_clustering_dashboard(
        "LOKI+GPT-OSS 20B",
        loki_gpt_dashboard_rows,
        loki_gpt_dashboard_summary,
        args.viz_dir / f"{_slug('LOKI+GPT-OSS 20B')}_relationship_clustering_dashboard.png",
        color_metric_key="pair_average_precision",
        color_metric_label="Pair average precision",
    )
    visualize_relationship_clustering_dashboard(
        "LOKI+Qwen-3.6",
        loki_qwen_dashboard_rows,
        loki_qwen_dashboard_summary,
        args.viz_dir / f"{_slug('LOKI+Qwen-3.6')}_relationship_clustering_dashboard.png",
        color_metric_key="pair_average_precision",
        color_metric_label="Pair average precision",
    )
    visualize_loki_per_admission_semantic_integration(
        [
            ("LOKI+GPT-OSS 20B", loki_gpt_dashboard_rows, loki_gpt_dashboard_summary, "viridis_redcap"),
            ("LOKI+Qwen-3.6", loki_qwen_dashboard_rows, loki_qwen_dashboard_summary, "viridis_redcap"),
        ],
        args.viz_dir / "loki_per_admission_relationship_clustering_quality.png",
    )

    cluster_summary_rows.append(
        summarize_cluster_system(
            system_name="LOKI+GPT-OSS 20B",
            source_file=args.loki_gpt_resume.name,
            scope="full",
            evaluations=loki_gpt_cluster_full,
        )
    )
    cluster_summary_rows.append(
        summarize_cluster_system(
            system_name="LOKI+Qwen-3.6",
            source_file=args.loki_qwen_resume.name,
            scope="full",
            evaluations=loki_qwen_cluster_full,
        )
    )

    pred_files = sorted(path for path in args.pred_dir.glob("*.json") if path.is_file())
    if not pred_files:
        raise FileNotFoundError(f"No prediction JSON files found under {args.pred_dir}")

    for pred_file in pred_files:
        stem_slug = _slug(pred_file.stem)
        _remove_plot_outputs(args.viz_dir / f"{stem_slug}_relationship_clustering_metrics.png")
        _remove_plot_outputs(args.viz_dir / f"{stem_slug}_relationship_clustering_slices.png")

    for pred_file in pred_files:
        prompt_dashboard_rows = build_prompt_dashboard_rows(pred_file, annotation_entries)
        prompt_dashboard_summary = summarize_dashboard_rows(
            system_name=pred_file.stem,
            source_file=pred_file.name,
            scope="full",
            rows=prompt_dashboard_rows,
        )
        dashboard_summary_rows.append(prompt_dashboard_summary)
        dashboard_per_admission_rows.extend(
            {
                "system_name": pred_file.stem,
                "source_file": pred_file.name,
                **row,
            }
            for row in prompt_dashboard_rows
        )

        prompt_cluster_full = build_prompt_cluster_evaluations(pred_file, annotation_entries)
        prompt_cluster_by_admission = {evaluation.admission_id: evaluation for evaluation in prompt_cluster_full}
        prompt_cluster_by_name[pred_file.stem] = prompt_cluster_by_admission
        prompt_source_files[pred_file.stem] = pred_file.name

        # Summarize prompt system itself
        cluster_summary_rows.append(
            summarize_cluster_system(
                system_name=pred_file.stem,
                source_file=pred_file.name,
                scope="full",
                evaluations=prompt_cluster_full,
            )
        )

        # loki_variants_to_compare is already defined at the outer main scope

        primary_loki_name, primary_loki_by_admission, primary_loki_resume = loki_variants_to_compare[0]
        primary_shared_ids = sorted(set(prompt_cluster_by_admission) & set(primary_loki_by_admission), key=int)
        shared_admission_ids_by_prompt[pred_file.stem] = set(primary_shared_ids)

        primary_hard_ids = [
            admission_id
            for admission_id in primary_shared_ids
            if prompt_cluster_by_admission[admission_id].gt_label_cardinality > 1
            and primary_loki_by_admission[admission_id].gt_label_cardinality > 1
        ]
        hard_admission_ids_by_prompt[pred_file.stem] = set(primary_hard_ids)

        for loki_name, loki_by_admission, loki_resume_file in loki_variants_to_compare:
            cluster_shared_ids = sorted(set(prompt_cluster_by_admission) & set(loki_by_admission), key=int)
            prompt_cluster_matched = [prompt_cluster_by_admission[admission_id] for admission_id in cluster_shared_ids]
            loki_cluster_matched = [loki_by_admission[admission_id] for admission_id in cluster_shared_ids]
            cluster_hard_ids = [
                admission_id
                for admission_id in cluster_shared_ids
                if prompt_cluster_by_admission[admission_id].gt_label_cardinality > 1
                and loki_by_admission[admission_id].gt_label_cardinality > 1
            ]

            cluster_summary_rows.append(
                summarize_cluster_system(
                    system_name=loki_name,
                    source_file=loki_resume_file.name,
                    scope=f"matched_to_{pred_file.stem}",
                    evaluations=loki_cluster_matched,
                )
            )
            cluster_summary_rows.append(
                summarize_cluster_system(
                    system_name=pred_file.stem,
                    source_file=pred_file.name,
                    scope=f"matched_to_{loki_name}",
                    evaluations=prompt_cluster_matched,
                )
            )
            for evaluation in prompt_cluster_matched:
                loki_eval = loki_by_admission[evaluation.admission_id]
                cluster_per_admission_rows.append({
                    "system_name": pred_file.stem,
                    "source_file": pred_file.name,
                    "admission_id": evaluation.admission_id,
                    "patient_id": evaluation.patient_id,
                    "n_gt_matched_pairs": evaluation.n_gt_matched_pairs,
                    "n_clusters": evaluation.n_clusters,
                    "raw_pair_cluster_purity": evaluation.raw_pair_cluster_purity,
                    "raw_pair_oracle_precision": evaluation.raw_pair_oracle_precision,
                    "raw_pair_oracle_recall": evaluation.raw_pair_oracle_recall,
                    "raw_pair_oracle_f1": evaluation.raw_pair_oracle_f1,
                    "cluster_label_macro_precision": evaluation.cluster_label_macro_precision,
                    "cluster_label_macro_recall": evaluation.cluster_label_macro_recall,
                    "cluster_label_macro_f1": evaluation.cluster_label_macro_f1,
                    "cluster_label_precision": evaluation.cluster_label_precision,
                    "cluster_label_recall": evaluation.cluster_label_recall,
                    "cluster_label_f1": evaluation.cluster_label_f1,
                    "cluster_label_accuracy": evaluation.cluster_label_accuracy,
                    "cluster_label_n_evaluated": evaluation.cluster_label_n_evaluated,
                    "cluster_label_n_correct": evaluation.cluster_label_n_correct,
                    "loki_system_name": loki_name,
                    "loki_n_gt_matched_pairs": loki_eval.n_gt_matched_pairs,
                    "loki_n_clusters": loki_eval.n_clusters,
                    "loki_raw_pair_cluster_purity": loki_eval.raw_pair_cluster_purity,
                    "loki_raw_pair_oracle_f1": loki_eval.raw_pair_oracle_f1,
                    "loki_cluster_label_macro_f1": loki_eval.cluster_label_macro_f1,
                    "loki_cluster_label_accuracy": loki_eval.cluster_label_accuracy,
                    "delta_raw_pair_oracle_f1_vs_loki": None if evaluation.raw_pair_oracle_f1 is None or loki_eval.raw_pair_oracle_f1 is None else round(evaluation.raw_pair_oracle_f1 - loki_eval.raw_pair_oracle_f1, 4),
                    "delta_cluster_label_macro_f1_vs_loki": None if evaluation.cluster_label_macro_f1 is None or loki_eval.cluster_label_macro_f1 is None else round(evaluation.cluster_label_macro_f1 - loki_eval.cluster_label_macro_f1, 4),
                    "delta_cluster_label_accuracy_vs_loki": None if evaluation.cluster_label_accuracy is None or loki_eval.cluster_label_accuracy is None else round(evaluation.cluster_label_accuracy - loki_eval.cluster_label_accuracy, 4),
                })

            cluster_fairness_rows.append(
                build_cluster_fairness_row(
                    prompt_name=pred_file.stem,
                    prompt_source_file=pred_file.name,
                    scope="matched_all",
                    admission_ids=cluster_shared_ids,
                    prompt_by_admission=prompt_cluster_by_admission,
                    loki_by_admission=loki_by_admission,
                    loki_system_name=loki_name,
                    loki_resume_name=loki_resume_file.name,
                )
            )
            if cluster_hard_ids:
                cluster_fairness_rows.append(
                    build_cluster_fairness_row(
                        prompt_name=pred_file.stem,
                        prompt_source_file=pred_file.name,
                        scope="matched_multitype_overlap",
                        admission_ids=cluster_hard_ids,
                        prompt_by_admission=prompt_cluster_by_admission,
                        loki_by_admission=loki_by_admission,
                        loki_system_name=loki_name,
                        loki_resume_name=loki_resume_file.name,
                    )
                )

        matched_cluster_per_admission_rows = [
            row for row in cluster_per_admission_rows if row.get("system_name") == pred_file.stem
        ]
        stem_slug = _slug(pred_file.stem)
        visualize_relationship_clustering_dashboard(
            pred_file.stem,
            prompt_dashboard_rows,
            prompt_dashboard_summary,
            args.viz_dir / f"{stem_slug}_relationship_clustering_dashboard.png",
            color_metric_key="raw_pair_oracle_f1",
            color_metric_label="Raw oracle pair F1",
        )
        visualize_relationship_cluster_count_scatter(
            pred_file.stem,
            matched_cluster_per_admission_rows,
            args.viz_dir / f"{stem_slug}_relationship_clustering_cluster_counts.png",
        )
        visualize_ranked_delta_plot(
            pred_file.stem,
            matched_cluster_per_admission_rows,
            "delta_raw_pair_oracle_f1_vs_loki",
            "Relationship Clustering Raw Oracle F1 Delta",
            args.viz_dir / f"{stem_slug}_relationship_clustering_raw_oracle_f1_delta.png",
        )

    prompt_dashboard_summaries = [
        row
        for row in dashboard_summary_rows
        if row.get("scope") == "full" and row.get("system_name") not in (None, "", "LOKI")
    ]
    combined_slice_rows: List[Dict[str, Any]] = []
    combined_prompt_names = sorted(prompt_cluster_by_name, key=str.lower)
    if combined_prompt_names:
        # Intersect admission IDs across all prompt and LOKI systems
        combined_shared_ids_set = set.intersection(
            *(set(prompt_cluster_by_name[p]) for p in combined_prompt_names)
        )
        for _, loki_by_admission, _ in loki_variants_to_compare:
            combined_shared_ids_set = combined_shared_ids_set & set(loki_by_admission)
        combined_shared_ids = sorted(combined_shared_ids_set, key=int)

        if combined_shared_ids:
            for loki_name, loki_by_admission, loki_resume_file in loki_variants_to_compare:
                for prompt_name in combined_prompt_names:
                    combined_slice_rows.append(
                        build_cluster_fairness_row(
                            prompt_name=prompt_name,
                            prompt_source_file=prompt_source_files[prompt_name],
                            scope="combined_matched_all",
                            admission_ids=combined_shared_ids,
                            prompt_by_admission=prompt_cluster_by_name[prompt_name],
                            loki_by_admission=loki_by_admission,
                            loki_system_name=loki_name,
                            loki_resume_name=loki_resume_file.name,
                        )
                    )

        combined_hard_ids = []
        if combined_shared_ids:
            for admission_id in combined_shared_ids:
                is_hard = True
                for prompt_name in combined_prompt_names:
                    if prompt_cluster_by_name[prompt_name][admission_id].gt_label_cardinality <= 1:
                        is_hard = False
                        break
                if is_hard:
                    for _, loki_by_admission, _ in loki_variants_to_compare:
                        if loki_by_admission[admission_id].gt_label_cardinality <= 1:
                            is_hard = False
                            break
                if is_hard:
                    combined_hard_ids.append(admission_id)
            combined_hard_ids = sorted(combined_hard_ids, key=int)

        if combined_hard_ids:
            for loki_name, loki_by_admission, loki_resume_file in loki_variants_to_compare:
                for prompt_name in combined_prompt_names:
                    combined_slice_rows.append(
                        build_cluster_fairness_row(
                            prompt_name=prompt_name,
                            prompt_source_file=prompt_source_files[prompt_name],
                            scope="combined_matched_multitype_overlap",
                            admission_ids=combined_hard_ids,
                            prompt_by_admission=prompt_cluster_by_name[prompt_name],
                            loki_by_admission=loki_by_admission,
                            loki_system_name=loki_name,
                            loki_resume_name=loki_resume_file.name,
                        )
                    )

    visualize_combined_main_metric_bars(
        dashboard_summary_rows,
        args.viz_dir / "all_models_main_comparison_metrics.png",
    )
    visualize_semantic_integration_metric_bars(
        dashboard_summary_rows,
        args.viz_dir / "all_models_semantic_integration_metrics.png",
    )
    visualize_semantic_integration_fairness_slices(
        combined_slice_rows,
        args.viz_dir / "all_models_semantic_integration_slices.png",
    )
    visualize_relationship_clustering_metric_bars(
        dashboard_summary_rows,
        args.viz_dir / "all_models_relationship_clustering_metrics.png",
    )
    visualize_relationship_clustering_fairness_slices(
        combined_slice_rows,
        args.viz_dir / "all_models_relationship_clustering_slices.png",
    )
    visualize_all_models_compute_cost(
        loki_gpt_dashboard_rows,
        loki_qwen_dashboard_rows,
        args.output_dir,
        args.viz_dir / "all_models_compute_cost.png",
    )
    visualize_all_models_compute_cost_half_circle(
        loki_gpt_dashboard_rows,
        loki_qwen_dashboard_rows,
        args.output_dir,
        args.viz_dir / "all_models_compute_cost_half_circle.png",
    )
    visualize_all_models_compute_cost_flat(
        loki_gpt_dashboard_rows,
        loki_qwen_dashboard_rows,
        args.output_dir,
        args.viz_dir / "all_models_compute_cost_flat.png",
    )
    visualize_all_models_compute_cost_broken_axis(
        loki_gpt_dashboard_rows,
        loki_qwen_dashboard_rows,
        args.output_dir,
        args.viz_dir / "all_models_compute_cost_broken_axis.png",
    )
    visualize_all_models_compute_cost_side_by_side(
        loki_gpt_dashboard_rows,
        loki_qwen_dashboard_rows,
        args.output_dir,
        args.viz_dir / "all_models_compute_cost_side_by_side.png",
    )
    visualize_all_models_data_quality(
        args.output_dir,
        args.viz_dir / "all_models_data_quality.png",
    )

    cluster_summary_fieldnames = [
        "system_name",
        "source_file",
        "scope",
        "n_admissions",
        "n_gt_matched_pairs",
        "n_clusters",
        "raw_pair_cluster_purity",
        "raw_pair_oracle_precision",
        "raw_pair_oracle_recall",
        "raw_pair_oracle_f1",
        "cluster_label_macro_precision",
        "cluster_label_macro_recall",
        "cluster_label_macro_f1",
        "cluster_label_precision",
        "cluster_label_recall",
        "cluster_label_f1",
        "cluster_label_accuracy",
        "n_evaluated_clusters",
        "n_correct_clusters",
    ]
    cluster_per_admission_fieldnames = [
        "system_name",
        "source_file",
        "admission_id",
        "patient_id",
        "n_gt_matched_pairs",
        "n_clusters",
        "raw_pair_cluster_purity",
        "raw_pair_oracle_precision",
        "raw_pair_oracle_recall",
        "raw_pair_oracle_f1",
        "cluster_label_macro_precision",
        "cluster_label_macro_recall",
        "cluster_label_macro_f1",
        "cluster_label_precision",
        "cluster_label_recall",
        "cluster_label_f1",
        "cluster_label_accuracy",
        "cluster_label_n_evaluated",
        "cluster_label_n_correct",
        "loki_system_name",
        "loki_n_gt_matched_pairs",
        "loki_n_clusters",
        "loki_raw_pair_cluster_purity",
        "loki_raw_pair_oracle_f1",
        "loki_cluster_label_macro_f1",
        "loki_cluster_label_accuracy",
        "delta_raw_pair_oracle_f1_vs_loki",
        "delta_cluster_label_macro_f1_vs_loki",
        "delta_cluster_label_accuracy_vs_loki",
    ]
    cluster_fairness_fieldnames = [
        "prompt_system_name",
        "prompt_source_file",
        "loki_system_name",
        "scope",
        "n_admissions",
        "prompt_n_gt_matched_pairs",
        "loki_n_gt_matched_pairs",
        "prompt_n_clusters",
        "loki_n_clusters",
        "prompt_raw_pair_cluster_purity",
        "loki_raw_pair_cluster_purity",
        "prompt_raw_pair_oracle_precision",
        "prompt_raw_pair_oracle_recall",
        "prompt_raw_pair_oracle_f1",
        "loki_raw_pair_oracle_f1",
        "prompt_cluster_label_macro_precision",
        "loki_cluster_label_macro_precision",
        "prompt_cluster_label_macro_recall",
        "loki_cluster_label_macro_recall",
        "prompt_cluster_label_macro_f1",
        "loki_cluster_label_macro_f1",
        "prompt_cluster_label_accuracy",
        "loki_cluster_label_accuracy",
        "prompt_cluster_ari",
        "loki_cluster_ari",
        "delta_raw_pair_oracle_f1",
        "delta_cluster_label_macro_f1",
        "delta_cluster_label_accuracy",
        "prompt_single_type_admissions",
        "loki_single_type_admissions",
        "prompt_more_pairs_admissions",
        "equal_pair_count_admissions",
        "prompt_fewer_pairs_admissions",
        "prompt_more_clusters_admissions",
        "equal_cluster_count_admissions",
        "prompt_fewer_clusters_admissions",
    ]

    relationship_summary_csv = args.output_dir / "relationship_clustering_summary.csv"
    relationship_per_admission_csv = args.output_dir / "relationship_clustering_per_admission.csv"
    relationship_report_md = args.output_dir / "relationship_clustering_report.md"
    relationship_fairness_csv = args.output_dir / "relationship_clustering_fairness_summary.csv"
    relationship_fairness_report_md = args.output_dir / "relationship_clustering_fairness_report.md"
    relationship_dashboard_summary_csv = args.output_dir / "relationship_clustering_dashboard_summary.csv"
    relationship_dashboard_per_admission_csv = args.output_dir / "relationship_clustering_dashboard_per_admission.csv"
    relationship_dashboard_report_md = args.output_dir / "relationship_clustering_dashboard_report.md"
    relationship_visualizations_md = args.output_dir / "relationship_clustering_visualizations.md"

    _write_csv(relationship_summary_csv, cluster_summary_rows, cluster_summary_fieldnames)
    _write_csv(relationship_per_admission_csv, cluster_per_admission_rows, cluster_per_admission_fieldnames)
    _write_csv(relationship_fairness_csv, cluster_fairness_rows, cluster_fairness_fieldnames)
    dashboard_summary_fieldnames = [
        "system_name",
        "source_file",
        "scope",
        "n_admissions",
        "n_pred_pairs",
        "n_gt_pairs",
        "n_final_clusters",
        "cluster_label_n_evaluated",
        "cluster_label_n_correct",
        "pair_average_precision",
        "cluster_label_macro_precision",
        "cluster_label_macro_recall",
        "cluster_label_macro_f1",
        "cluster_label_precision",
        "cluster_label_recall",
        "cluster_label_f1",
        "cluster_label_accuracy",
        "raw_pair_cluster_purity",
        "raw_pair_oracle_precision",
        "raw_pair_oracle_recall",
        "raw_pair_oracle_f1",
        "cluster_ari",
        "cluster_silhouette",
    ]
    dashboard_per_admission_fieldnames = [
        "system_name",
        "source_file",
        "dataset",
        "evaluation_profile",
        "admission_id",
        "patient_id",
        "runtime_sec",
        "n_diag_rows",
        "n_med_rows",
        "n_sentences",
        "n_paths",
        "n_pred_pairs",
        "n_gt_pairs",
        "n_final_clusters",
        "cluster_label_backend",
        "gliner2_label_input_mode",
        "pair_average_precision",
        "cluster_label_macro_precision",
        "cluster_label_macro_recall",
        "cluster_label_macro_f1",
        "cluster_label_precision",
        "cluster_label_recall",
        "cluster_label_f1",
        "cluster_label_accuracy",
        "cluster_label_n_evaluated",
        "cluster_label_n_correct",
        "raw_pair_cluster_purity",
        "raw_pair_oracle_precision",
        "raw_pair_oracle_recall",
        "raw_pair_oracle_f1",
        "cluster_purity",
        "cluster_ari",
        "cluster_silhouette",
    ]
    _write_csv(relationship_dashboard_summary_csv, dashboard_summary_rows, dashboard_summary_fieldnames)
    _write_csv(relationship_dashboard_per_admission_csv, dashboard_per_admission_rows, dashboard_per_admission_fieldnames)
    relationship_report_md.write_text(build_cluster_report(cluster_summary_rows), encoding="utf-8")
    relationship_fairness_report_md.write_text(build_cluster_fairness_report(cluster_fairness_rows), encoding="utf-8")
    relationship_dashboard_report_md.write_text(build_relationship_dashboard_report(dashboard_summary_rows), encoding="utf-8")
    relationship_visualizations_md.write_text(
        build_relationship_visualization_gallery(dashboard_summary_rows, cluster_fairness_rows, args.output_dir, args.viz_dir),
        encoding="utf-8",
    )

    print(f"Saved Relationship Clustering summary CSV: {relationship_summary_csv}")
    print(f"Saved Relationship Clustering per-admission CSV: {relationship_per_admission_csv}")
    print(f"Saved Relationship Clustering report MD: {relationship_report_md}")
    print(f"Saved Relationship Clustering fairness CSV: {relationship_fairness_csv}")
    print(f"Saved Relationship Clustering fairness report MD: {relationship_fairness_report_md}")
    print(f"Saved Relationship Clustering dashboard summary CSV: {relationship_dashboard_summary_csv}")
    print(f"Saved Relationship Clustering dashboard per-admission CSV: {relationship_dashboard_per_admission_csv}")
    print(f"Saved Relationship Clustering dashboard report MD: {relationship_dashboard_report_md}")
    print(f"Saved Relationship Clustering visualization page: {relationship_visualizations_md}")
    print(f"Saved Relationship Clustering visualizations under: {args.viz_dir}")
    for row in dashboard_summary_rows:
        print(
            f"Relationship Clustering dashboard | {row['system_name']}: admissions={row['n_admissions']} "
            f"pred_pairs={row['n_pred_pairs']} gt_pairs={row['n_gt_pairs']} "
            f"final_clusters={row['n_final_clusters']} "
            f"macro_f1={_fmt(_safe_float(row.get('cluster_label_macro_f1')))} "
            f"accuracy={_fmt(_safe_float(row.get('cluster_label_accuracy')))} "
            f"raw_oracle_f1={_fmt(_safe_float(row.get('raw_pair_oracle_f1')))}"
        )
    for row in cluster_fairness_rows:
        if row.get("scope") != "matched_all":
            continue
        print(
            f"Relationship Clustering | {row['prompt_system_name']} vs LOKI: admissions={row['n_admissions']} "
            f"prompt_pairs={row['prompt_n_gt_matched_pairs']} loki_pairs={row['loki_n_gt_matched_pairs']} "
            f"prompt_clusters={row['prompt_n_clusters']} loki_clusters={row['loki_n_clusters']} "
            f"prompt_raw_oracle_f1={_fmt(_safe_float(row.get('prompt_raw_pair_oracle_f1')))} "
            f"loki_raw_oracle_f1={_fmt(_safe_float(row.get('loki_raw_pair_oracle_f1')))}"
        )
    for row in cluster_fairness_rows:
        if row.get("scope") != "matched_multitype_overlap":
            continue
        print(
            f"Relationship Clustering hard slice | {row['prompt_system_name']} vs LOKI: admissions={row['n_admissions']} "
            f"prompt_raw_oracle_f1={_fmt(_safe_float(row.get('prompt_raw_pair_oracle_f1')))} "
            f"loki_raw_oracle_f1={_fmt(_safe_float(row.get('loki_raw_pair_oracle_f1')))}"
        )


if __name__ == "__main__":
    main()