import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

UNI_R_TO_S = "Uni (R⟶S)"
UNI_S_TO_R = "Uni (S⟶R)"

MODEL_NAME_ALIASES = {
    "Uni-cross": UNI_R_TO_S,
    "Uni (R-S)": UNI_R_TO_S,
    "Uni (R→S)": UNI_R_TO_S,
    "Uni (S-R)": UNI_S_TO_R,
    "Uni (S→R)": UNI_S_TO_R,
}

MODEL_ORDER = ["Baseline", "FT-Encoder", UNI_R_TO_S, UNI_S_TO_R, "LOKI"]


def _normalize_model_name(name: str) -> str:
    return MODEL_NAME_ALIASES.get(name, name)


def _normalize_model_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in mapping.items():
        normalized[_normalize_model_name(key)] = value
    return normalized


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _collect_stage_rows(results_json_paths: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in results_json_paths:
        payload = _load_json(p)
        model_name = _normalize_model_name(
            payload.get("evaluation_info", {}).get("model_display_name", p.parent.name)
        )
        evaluations = payload.get("evaluations", {})
        for stage_key, stage_data in evaluations.items():
            if not isinstance(stage_data, dict):
                continue
            ranking_raw = stage_data.get("ranking_raw_counts", {})
            pred = stage_data.get("prediction_breakdown", {})
            diag = stage_data.get("diagnosis_prediction_breakdown", {})
            med = stage_data.get("medication_prediction_breakdown", {})
            rows.append(
                {
                    "model": model_name,
                    "stage_key": stage_key,
                    "average_precision": stage_data.get("average_precision"),
                    "dynamic_f1": stage_data.get("dynamic_f1", stage_data.get("overall_accuracy")),
                    "roc_auc": stage_data.get("roc_auc"),
                    "queries_evaluated": ranking_raw.get("queries_evaluated"),
                    "total_rows": ranking_raw.get("total_rows"),
                    "total_documents": ranking_raw.get("total_documents"),
                    "total_candidate_pairs": ranking_raw.get("total_candidate_pairs"),
                    "total_ground_truth_pairs": ranking_raw.get("total_ground_truth_pairs"),
                    "hits@1": _safe_get(ranking_raw, ["hits_at_k", "1"], 0),
                    "hits@3": _safe_get(ranking_raw, ["hits_at_k", "3"], 0),
                    "hits@5": _safe_get(ranking_raw, ["hits_at_k", "5"], 0),
                    "hits@10": _safe_get(ranking_raw, ["hits_at_k", "10"], 0),
                    "max_hits@1": _safe_get(ranking_raw, ["max_possible_hits_at_k", "1"], 0),
                    "max_hits@3": _safe_get(ranking_raw, ["max_possible_hits_at_k", "3"], 0),
                    "max_hits@5": _safe_get(ranking_raw, ["max_possible_hits_at_k", "5"], 0),
                    "max_hits@10": _safe_get(ranking_raw, ["max_possible_hits_at_k", "10"], 0),
                    "tp": pred.get("tp"),
                    "fp": pred.get("fp"),
                    "fn": pred.get("fn"),
                    "diag_tp": diag.get("tp"),
                    "diag_fp": diag.get("fp"),
                    "diag_fn": diag.get("fn"),
                    "med_tp": med.get("tp"),
                    "med_fp": med.get("fp"),
                    "med_fn": med.get("fn"),
                }
            )
    return pd.DataFrame(rows)


def _build_comparison_tables(
    comparison_raw_counts_path: Path,
    dataset_statistics_path: Path,
    comparison_metrics_path: Path,
) -> Dict[str, pd.DataFrame]:
    comparison_raw = _normalize_model_mapping(_load_json(comparison_raw_counts_path))
    ds_stats = _load_json(dataset_statistics_path)
    comparison_metrics = _load_json(comparison_metrics_path) if comparison_metrics_path.exists() else {}
    metrics_by_model = (
        _normalize_model_mapping(comparison_metrics.get("metrics", {}))
        if isinstance(comparison_metrics, dict)
        else {}
    )

    ranking_rows: List[Dict[str, Any]] = []
    hits_long_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    table_conf_rows: List[Dict[str, Any]] = []

    for model_name, payload in comparison_raw.items():
        ranking_raw = payload.get("ranking_raw_counts", {})
        hits = ranking_raw.get("hits_at_k", {})
        max_hits = ranking_raw.get("max_possible_hits_at_k", {})
        row = {
            "model": model_name,
            "queries_evaluated": ranking_raw.get("queries_evaluated", 0),
            "total_rows": ranking_raw.get("total_rows", 0),
            "total_documents": ranking_raw.get("total_documents", 0),
            "total_candidate_pairs": ranking_raw.get("total_candidate_pairs", 0),
            "total_ground_truth_pairs": ranking_raw.get("total_ground_truth_pairs", 0),
        }
        model_metrics = metrics_by_model.get(model_name, {})
        plot_precision_at_k = model_metrics.get("precision_at_k", {})
        plot_recall_at_k = model_metrics.get("recall_at_k", {})

        queries_eval = ranking_raw.get("queries_evaluated", 0) or 0
        gt_total = ranking_raw.get("total_ground_truth_pairs", 0) or 0
        for k in ["1", "3", "5", "10"]:
            h = hits.get(k, 0)
            m = max_hits.get(k, 0)
            row[f"hits@{k}"] = h
            row[f"max_hits@{k}"] = m
            # This is not the plotted precision/recall; it is a ceiling-normalized hit coverage.
            row[f"hit_ceiling_ratio@{k}"] = (h / m) if m else 0.0
            # Micro-style precision/recall from global counts.
            row[f"micro_precision@{k}_from_hits"] = (
                h / (queries_eval * int(k))
            ) if queries_eval else 0.0
            row[f"micro_recall@{k}_from_hits"] = (h / gt_total) if gt_total else 0.0
            # Exact plotted values from comparison_metrics.json (macro over examples).
            row[f"plot_precision@{k}"] = plot_precision_at_k.get(k, plot_precision_at_k.get(int(k), None))
            row[f"plot_recall@{k}"] = plot_recall_at_k.get(k, plot_recall_at_k.get(int(k), None))
            hits_long_rows.append(
                {
                    "model": model_name,
                    "k": int(k),
                    "hits": h,
                    "max_possible_hits": m,
                    "hit_ceiling_ratio": (h / m) if m else 0.0,
                    "micro_precision_from_hits": (
                        h / (queries_eval * int(k))
                    ) if queries_eval else 0.0,
                    "micro_recall_from_hits": (h / gt_total) if gt_total else 0.0,
                    "plot_precision_macro": plot_precision_at_k.get(k, plot_precision_at_k.get(int(k), None)),
                    "plot_recall_macro": plot_recall_at_k.get(k, plot_recall_at_k.get(int(k), None)),
                }
            )
        ranking_rows.append(row)

        pred = payload.get("prediction_breakdown", {})
        tp = pred.get("tp", 0)
        fp = pred.get("fp", 0)
        fn = pred.get("fn", 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        confusion_rows.append(
            {
                "model": model_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision_from_counts": precision,
                "recall_from_counts": recall,
                "f1_from_counts": f1,
            }
        )

        for table_key, table_name in [
            ("diagnosis_prediction_breakdown", "diagnosis"),
            ("medication_prediction_breakdown", "medication"),
        ]:
            t = payload.get(table_key, {})
            ttp = t.get("tp", 0)
            tfp = t.get("fp", 0)
            tfn = t.get("fn", 0)
            p = ttp / (ttp + tfp) if (ttp + tfp) else 0.0
            r = ttp / (ttp + tfn) if (ttp + tfn) else 0.0
            f = (2 * p * r / (p + r)) if (p + r) else 0.0
            table_conf_rows.append(
                {
                    "model": model_name,
                    "table_type": table_name,
                    "tp": ttp,
                    "fp": tfp,
                    "fn": tfn,
                    "precision_from_counts": p,
                    "recall_from_counts": r,
                    "f1_from_counts": f,
                }
            )

    split_rows: List[Dict[str, Any]] = []
    sentlen_rows: List[Dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        s = ds_stats.get(split_name, {})
        if not s:
            continue
        split_rows.append(
            {
                "split": split_name,
                "num_examples": s.get("num_examples"),
                "total_rows": s.get("total_rows"),
                "avg_rows_per_example": s.get("avg_rows_per_example"),
                "total_primary_sentences": s.get("total_primary_sentences"),
                "avg_primary_sentences_per_example": s.get("avg_primary_sentences_per_example"),
                "total_additional_positive_sentences": s.get("total_additional_positive_sentences"),
                "total_negative_sentences": s.get("total_negative_sentences"),
                "source_file": s.get("source_file"),
                "matched_to_annotations": s.get("matched_to_annotations"),
                "annotation_coverage_ratio": s.get("annotation_coverage_ratio"),
            }
        )
        for granularity_key, granularity_name in [
            ("sentence_length_tokens", "tokens"),
            ("sentence_length_chars", "chars"),
        ]:
            g = s.get(granularity_key, {})
            sentlen_rows.append(
                {
                    "split": split_name,
                    "granularity": granularity_name,
                    "mean": g.get("mean"),
                    "median": g.get("median"),
                    "std": g.get("std"),
                    "p95": g.get("p95"),
                }
            )

    ann = ds_stats.get("annotation_summary", {})
    annotation_df = pd.DataFrame(
        [
            {
                "num_admissions": ann.get("num_admissions"),
                "num_anchor_mappings": ann.get("num_anchor_mappings"),
            }
        ]
    )

    return {
        "ranking_raw_counts": pd.DataFrame(ranking_rows),
        "ranking_hits_long": pd.DataFrame(hits_long_rows),
        "grounding_confusion_overall": pd.DataFrame(confusion_rows),
        "grounding_confusion_by_table": pd.DataFrame(table_conf_rows),
        "dataset_split_statistics": pd.DataFrame(split_rows),
        "dataset_sentence_length": pd.DataFrame(sentlen_rows),
        "annotation_summary": annotation_df,
        "metric_notes_ranking": _build_metric_notes_table(),
        "metric_notes_grounding": _build_grounding_metric_notes_table(),
        "metric_notes_dataset": _build_dataset_metric_notes_table(),
    }


def _write_workbook(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            # Excel sheet name limit is 31 chars.
            safe_sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)


def _build_paper_condensed_tables(
    comparison_raw_counts_path: Path,
    dataset_statistics_path: Path,
    comparison_metrics_path: Path,
) -> Dict[str, pd.DataFrame]:
    comparison_raw = _load_json(comparison_raw_counts_path)
    ds_stats = _load_json(dataset_statistics_path)
    comparison_metrics = _load_json(comparison_metrics_path) if comparison_metrics_path.exists() else {}
    metrics_by_model = comparison_metrics.get("metrics", {}) if isinstance(comparison_metrics, dict) else {}

    result_rows: List[Dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        if model_name not in comparison_raw:
            continue
        raw = comparison_raw.get(model_name, {})
        ranking_raw = raw.get("ranking_raw_counts", {})
        metrics = metrics_by_model.get(model_name, {})
        p_at_k = metrics.get("precision_at_k", {})
        r_at_k = metrics.get("recall_at_k", {})
        ndcg_at_k = metrics.get("ndcg_at_k", {})

        hits = ranking_raw.get("hits_at_k", {})
        max_hits = ranking_raw.get("max_possible_hits_at_k", {})
        queries = ranking_raw.get("queries_evaluated", 0)
        total_candidates = ranking_raw.get("total_candidate_pairs", 0)
        mean_rank = metrics.get("mean_rank")
        avg_candidates_per_query = (total_candidates / queries) if queries else 0.0
        mean_rank_percent_of_candidates = (
            (mean_rank / avg_candidates_per_query) * 100.0
            if (avg_candidates_per_query and mean_rank is not None)
            else None
        )

        result_rows.append(
            {
                "Model": model_name,
                "P@1": p_at_k.get("1", p_at_k.get(1)),
                "P@5": p_at_k.get("5", p_at_k.get(5)),
                "P@10": p_at_k.get("10", p_at_k.get(10)),
                "R@1": r_at_k.get("1", r_at_k.get(1)),
                "R@5": r_at_k.get("5", r_at_k.get(5)),
                "R@10": r_at_k.get("10", r_at_k.get(10)),
                "F1@1": metrics.get("f1_at_k", {}).get("1", metrics.get("f1_at_k", {}).get(1)),
                "F1@5": metrics.get("f1_at_k", {}).get("5", metrics.get("f1_at_k", {}).get(5)),
                "F1@10": metrics.get("f1_at_k", {}).get("10", metrics.get("f1_at_k", {}).get(10)),
                "NDCG@1": ndcg_at_k.get("1", ndcg_at_k.get(1)),
                "NDCG@5": ndcg_at_k.get("5", ndcg_at_k.get(5)),
                "NDCG@10": ndcg_at_k.get("10", ndcg_at_k.get(10)),
                "Mean Rank (raw, lower better)": mean_rank,
                "Avg candidates/query": avg_candidates_per_query,
                "Mean Rank (% of avg candidates, lower better)": mean_rank_percent_of_candidates,
                "Hits@1": hits.get("1", 0),
                "Max Hits@1": max_hits.get("1", 0),
                "Hits@5": hits.get("5", 0),
                "Max Hits@5": max_hits.get("5", 0),
                "Hits@10": hits.get("10", 0),
                "Max Hits@10": max_hits.get("10", 0),
                "Queries (examples)": queries,
                "Ground-truth pairs": ranking_raw.get("total_ground_truth_pairs", 0),
                "Candidate pairs": total_candidates,
            }
        )

    dataset_rows: List[Dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        s = ds_stats.get(split_name, {})
        if not s:
            continue
        dataset_rows.append(
            {
                "Split": split_name,
                "# Examples": s.get("num_examples"),
                "# Rows (total)": s.get("total_rows"),
                "Avg rows/example": s.get("avg_rows_per_example"),
                "# Primary sentences (total)": s.get("total_primary_sentences"),
                "Avg primary sentences/example": s.get("avg_primary_sentences_per_example"),
                "Sentence length tokens (mean)": _safe_get(s, ["sentence_length_tokens", "mean"]),
                "Sentence length tokens (p95)": _safe_get(s, ["sentence_length_tokens", "p95"]),
                "Sentence length chars (mean)": _safe_get(s, ["sentence_length_chars", "mean"]),
                "Sentence length chars (p95)": _safe_get(s, ["sentence_length_chars", "p95"]),
                "Annotation coverage ratio (test)": s.get("annotation_coverage_ratio"),
            }
        )

    ann = ds_stats.get("annotation_summary", {})
    annotation_note = pd.DataFrame(
        [
            {
                "Note": "Annotation summary",
                "num_admissions": ann.get("num_admissions"),
                "num_anchor_mappings": ann.get("num_anchor_mappings"),
            }
        ]
    )

    return {
        "result_section_table": pd.DataFrame(result_rows),
        "dataset_section_table": pd.DataFrame(dataset_rows),
        "annotation_note": annotation_note,
    }


def _build_metric_notes_table() -> pd.DataFrame:
    notes = [
        {
            "Metric": "Mean Rank (raw, lower better)",
            "Definition": "Average rank position of all ground-truth row-sentence pairs in the flattened ranked list.",
            "Interpretation": "Lower is better. Value is in rank units, not bounded to [0,1].",
        },
        {
            "Metric": "Avg candidates/query",
            "Definition": "Total candidate pairs divided by number of evaluated queries/examples.",
            "Interpretation": "Gives scale context for raw Mean Rank.",
        },
        {
            "Metric": "Mean Rank (% of avg candidates, lower better)",
            "Definition": "(Mean Rank / Avg candidates per query) * 100.",
            "Interpretation": "Normalized Mean Rank for easier cross-reading; still lower is better.",
        },
        {
            "Metric": "Hits@K / Max Hits@K",
            "Definition": "Hits@K = recovered GT pairs in top-K across queries; Max Hits@K = Σ min(|GT_q|, K).",
            "Interpretation": "Shows top-K GT coverage relative to ceiling imposed by GT density per query.",
        },
    ]
    return pd.DataFrame(notes)


def _build_grounding_metric_notes_table() -> pd.DataFrame:
    notes = [
        {
            "Metric": "Dynamic F1",
            "Definition": "F1 from per-example dynamic-threshold pair classification.",
            "Interpretation": "Secondary analysis metric; threshold adapts per example.",
        },
        {
            "Metric": "Average Precision",
            "Definition": "Area under precision-recall curve using pair-level scores.",
            "Interpretation": "Threshold-free ranking quality over all candidate pairs.",
        },
        {
            "Metric": "TP / FP / FN",
            "Definition": "Counts from dynamic-threshold pair classification.",
            "Interpretation": "Use for error decomposition, not directly comparable to @K ranking metrics.",
        },
    ]
    return pd.DataFrame(notes)


def _build_dataset_metric_notes_table() -> pd.DataFrame:
    notes = [
        {
            "Metric": "# Rows (total), Avg rows/example",
            "Definition": "Rows extracted from table content in each split.",
            "Interpretation": "Indicates structured candidate-side complexity.",
        },
        {
            "Metric": "# Primary sentences, Avg primary sentences/example",
            "Definition": "Sentences from primary positive notes per split.",
            "Interpretation": "Indicates document-side candidate volume.",
        },
        {
            "Metric": "Sentence length tokens/chars (mean, p95)",
            "Definition": "Distribution statistics over primary sentences.",
            "Interpretation": "Shows textual complexity and long-tail behavior.",
        },
        {
            "Metric": "Annotation coverage ratio (test)",
            "Definition": "Matched test examples / total test examples.",
            "Interpretation": "Higher is better; reflects usable annotation coverage.",
        },
    ]
    return pd.DataFrame(notes)


def _build_stage_metric_notes_table() -> pd.DataFrame:
    notes = [
        {
            "Metric": "stage_key",
            "Definition": "Evaluation stage identifier (stage_0 / stage_2 / stage_3 variants).",
            "Interpretation": "Used to compare frozen, pretrain, and trained checkpoints.",
        },
        {
            "Metric": "precision_at_k / recall_at_k / f1_at_k",
            "Definition": "Per-example @K metrics macro-averaged across queries.",
            "Interpretation": "Primary ranking metrics for retrieval-style performance.",
        },
        {
            "Metric": "ranking_raw_counts",
            "Definition": "Query/doc/pair totals and top-K hit counts.",
            "Interpretation": "Contextualizes ranking metrics with raw denominators.",
        },
    ]
    return pd.DataFrame(notes)


def _build_paper_grounding_table(
    comparison_raw_counts_path: Path,
    comparison_metrics_path: Path,
) -> pd.DataFrame:
    comparison_raw = _normalize_model_mapping(_load_json(comparison_raw_counts_path))
    comparison_metrics = _load_json(comparison_metrics_path) if comparison_metrics_path.exists() else {}
    metrics_by_model = (
        _normalize_model_mapping(comparison_metrics.get("metrics", {}))
        if isinstance(comparison_metrics, dict)
        else {}
    )

    rows: List[Dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        if model_name not in comparison_raw:
            continue
        raw = comparison_raw.get(model_name, {})
        metrics = metrics_by_model.get(model_name, {})
        pred = raw.get("prediction_breakdown", {})
        diag = raw.get("diagnosis_prediction_breakdown", {})
        med = raw.get("medication_prediction_breakdown", {})
        rows.append(
            {
                "Model": model_name,
                "Dynamic F1": metrics.get("dynamic_f1", metrics.get("overall_accuracy")),
                "Average Precision": metrics.get("average_precision"),
                "ROC-AUC": metrics.get("roc_auc"),
                "TP": pred.get("tp", 0),
                "FP": pred.get("fp", 0),
                "FN": pred.get("fn", 0),
                "Diagnosis TP": diag.get("tp", 0),
                "Diagnosis FP": diag.get("fp", 0),
                "Diagnosis FN": diag.get("fn", 0),
                "Medication TP": med.get("tp", 0),
                "Medication FP": med.get("fp", 0),
                "Medication FN": med.get("fn", 0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export VLDB-style evaluation tables to Excel workbooks."
    )
    parser.add_argument(
        "--comparison_raw_counts",
        type=str,
        default="Post_Training_Comparison_Plots/comparison_raw_counts.json",
    )
    parser.add_argument(
        "--dataset_statistics",
        type=str,
        default="Post_Training_Comparison_Plots/dataset_statistics.json",
    )
    parser.add_argument(
        "--comparison_metrics",
        type=str,
        default="Post_Training_Comparison_Plots/comparison_metrics.json",
    )
    parser.add_argument(
        "--results_glob_root",
        type=str,
        default="Post_Training_Results",
    )
    parser.add_argument(
        "--comparison_xlsx_out",
        type=str,
        default="Post_Training_Comparison_Plots/Tables_Comparison.xlsx",
    )
    parser.add_argument(
        "--stage_xlsx_out",
        type=str,
        default="Post_Training_Comparison_Plots/Tables_Stage_Details.xlsx",
    )
    parser.add_argument(
        "--paper_xlsx_out",
        type=str,
        default="Post_Training_Comparison_Plots/Tables_Paper_Condensed.xlsx",
    )
    parser.add_argument(
        "--paper_ranking_xlsx_out",
        type=str,
        default="Post_Training_Comparison_Plots/Result_Ranking.xlsx",
    )
    parser.add_argument(
        "--paper_grounding_xlsx_out",
        type=str,
        default="Post_Training_Comparison_Plots/Result_Grounding.xlsx",
    )
    parser.add_argument(
        "--paper_dataset_xlsx_out",
        type=str,
        default="Post_Training_Comparison_Plots/Dataset.xlsx",
    )
    args = parser.parse_args()

    comparison_raw_counts_path = Path(args.comparison_raw_counts)
    dataset_statistics_path = Path(args.dataset_statistics)
    comparison_metrics_path = Path(args.comparison_metrics)
    results_json_paths = sorted(Path(args.results_glob_root).glob("*/results_post_training_eval.json"))

    if not comparison_raw_counts_path.exists():
        raise FileNotFoundError(f"Missing file: {comparison_raw_counts_path}")
    if not dataset_statistics_path.exists():
        raise FileNotFoundError(f"Missing file: {dataset_statistics_path}")
    if not results_json_paths:
        raise FileNotFoundError(
            f"No results_post_training_eval.json found under: {args.results_glob_root}"
        )

    comparison_sheets = _build_comparison_tables(
        comparison_raw_counts_path=comparison_raw_counts_path,
        dataset_statistics_path=dataset_statistics_path,
        comparison_metrics_path=comparison_metrics_path,
    )
    _write_workbook(Path(args.comparison_xlsx_out), comparison_sheets)

    stage_df = _collect_stage_rows(results_json_paths)
    stage_sheets = {
        "stage_metrics_and_raw_counts": stage_df,
        "metric_notes": _build_stage_metric_notes_table(),
    }
    _write_workbook(Path(args.stage_xlsx_out), stage_sheets)

    paper_sheets = _build_paper_condensed_tables(
        comparison_raw_counts_path=comparison_raw_counts_path,
        dataset_statistics_path=dataset_statistics_path,
        comparison_metrics_path=comparison_metrics_path,
    )
    paper_sheets["metric_notes_ranking"] = _build_metric_notes_table()
    paper_sheets["metric_notes_dataset"] = _build_dataset_metric_notes_table()
    _write_workbook(Path(args.paper_xlsx_out), paper_sheets)

    # Isolated paper tables: ranking, grounding, and dataset in separate files.
    ranking_only = {
        "result_section_table": paper_sheets["result_section_table"],
        "metric_notes": _build_metric_notes_table(),
    }
    dataset_only = {
        "dataset_section_table": paper_sheets["dataset_section_table"],
        "annotation_note": paper_sheets["annotation_note"],
        "metric_notes": _build_dataset_metric_notes_table(),
    }
    grounding_only = {
        "grounding_result_table": _build_paper_grounding_table(
            comparison_raw_counts_path=comparison_raw_counts_path,
            comparison_metrics_path=comparison_metrics_path,
        ),
        "metric_notes": _build_grounding_metric_notes_table(),
    }
    _write_workbook(Path(args.paper_ranking_xlsx_out), ranking_only)
    _write_workbook(Path(args.paper_grounding_xlsx_out), grounding_only)
    _write_workbook(Path(args.paper_dataset_xlsx_out), dataset_only)

    print(f"[OK] Wrote comparison workbook: {args.comparison_xlsx_out}")
    print(f"[OK] Wrote stage-details workbook: {args.stage_xlsx_out}")
    print(f"[OK] Wrote paper-condensed workbook: {args.paper_xlsx_out}")
    print(f"[OK] Wrote paper ranking table: {args.paper_ranking_xlsx_out}")
    print(f"[OK] Wrote paper grounding table: {args.paper_grounding_xlsx_out}")
    print(f"[OK] Wrote paper dataset table: {args.paper_dataset_xlsx_out}")


if __name__ == "__main__":
    main()

