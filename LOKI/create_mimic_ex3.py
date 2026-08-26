#!/usr/bin/env python3
"""Build Datasets/mimic_ex3 from strong admissions across multiple runs.

Default behavior:
- Use mimic_small and mimic run groups.
- For each admission, compare baseline to all configured candidate runs.
- Keep the best candidate per admission by weighted delta score.
- Select admissions that satisfy "shines" thresholds.
- Export a merged dataset package under Datasets/mimic_ex3.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CandidateDelta:
    selection_reason: str
    dataset: str
    admission_id: str
    patient_id: str
    baseline_run: str
    best_run: str
    base_relaxed_pair_f1: float
    best_relaxed_pair_f1: float
    delta_relaxed_pair_f1: float
    base_gt_recovery: float
    best_gt_recovery: float
    delta_gt_recovery: float
    base_oracle_pair_f1: float
    best_oracle_pair_f1: float
    delta_oracle_pair_f1: float
    base_exact_triple_f1: float
    best_exact_triple_f1: float
    delta_exact_triple_f1: float
    base_pred_pairs: int
    best_pred_pairs: int
    delta_pred_pairs: int
    pred_pair_growth_ratio: float
    score: float


def _resolve(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def _as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return 0.0
    return float(value)


def _as_int(row: Dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "" or value is None:
        return 0
    return int(float(value))


def _load_results(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["admission_id"]: r for r in rows}


@dataclass
class RunGroup:
    dataset: str
    baseline_name: str
    baseline_results: Path
    candidate_runs: List[tuple[str, Path]]
    source_test: Path
    source_annotations: Path


def _default_run_groups() -> List[RunGroup]:
    return [
        RunGroup(
            dataset="mimic_small",
            baseline_name="loki_batch_mimic_small_diagrow025_stopcuediag16_20260526_123332",
            baseline_results=_resolve(
                "Batch_Materialization/"
                "loki_batch_mimic_small_diagrow025_stopcuediag16_20260526_123332/"
                "materialized_batch_results_mimic_small.csv"
            ),
            candidate_runs=[
                (
                    "loki_batch_mimic_small",
                    _resolve("Batch_Materialization/loki_batch_mimic_small/materialized_batch_results_mimic_small.csv"),
                ),
                (
                    "loki_batch_mimic_small_diagrow025_nostopcuediag16_20260526_130455",
                    _resolve(
                        "Batch_Materialization/"
                        "loki_batch_mimic_small_diagrow025_nostopcuediag16_20260526_130455/"
                        "materialized_batch_results_mimic_small.csv"
                    ),
                ),
                (
                    "loki_batch_mimic_small_diagrow025_nostopcuediag16_adcap0295_20260526_145615",
                    _resolve(
                        "Batch_Materialization/"
                        "loki_batch_mimic_small_diagrow025_nostopcuediag16_adcap0295_20260526_145615/"
                        "materialized_batch_results_mimic_small.csv"
                    ),
                ),
                (
                    "loki_batch_mimic_small_diagrow025_nostopcuediag16_adcap0295_gap005_20260526_164953",
                    _resolve(
                        "Batch_Materialization/"
                        "loki_batch_mimic_small_diagrow025_nostopcuediag16_adcap0295_gap005_20260526_164953/"
                        "materialized_batch_results_mimic_small.csv"
                    ),
                ),
            ],
            source_test=_resolve("Datasets/mimic_small/test_row_level.json"),
            source_annotations=_resolve("Datasets/mimic_small/Annotated_Test.json"),
        ),
        RunGroup(
            dataset="mimic",
            baseline_name="loki_batch_mimic_max20_diagrow025_stopcuediag16_20260526_100549",
            baseline_results=_resolve(
                "Batch_Materialization/"
                "loki_batch_mimic_max20_diagrow025_stopcuediag16_20260526_100549/"
                "materialized_batch_results_mimic.csv"
            ),
            candidate_runs=[
                (
                    "loki_batch_mimic",
                    _resolve("Batch_Materialization/loki_batch_mimic/materialized_batch_results_mimic.csv"),
                ),
                (
                    "loki_batch_mimic_max20_diagrow025_nostopcuediag16_20260526_110245",
                    _resolve(
                        "Batch_Materialization/"
                        "loki_batch_mimic_max20_diagrow025_nostopcuediag16_20260526_110245/"
                        "materialized_batch_results_mimic.csv"
                    ),
                ),
            ],
            source_test=_resolve("Datasets/mimic/test_row_level.json"),
            source_annotations=_resolve("Datasets/mimic/Annotated_Test.json"),
        ),
    ]


def _best_delta_for_group(
    group: RunGroup,
    weight_relaxed: float,
    weight_gt: float,
    weight_oracle: float,
    weight_exact: float,
    weight_pred_growth: float,
) -> List[CandidateDelta]:
    base_rows = _load_results(group.baseline_results)
    cand_maps = []
    for run_name, run_path in group.candidate_runs:
        cand_maps.append((run_name, _load_results(run_path)))

    shared_ids = sorted(base_rows.keys(), key=int)
    deltas: List[CandidateDelta] = []
    for admission_id in shared_ids:
        b = base_rows[admission_id]

        best_payload = None
        for run_name, cand_rows in cand_maps:
            c = cand_rows.get(admission_id)
            if c is None:
                continue

            base_rf1 = _as_float(b, "relaxed_pair_f1")
            cand_rf1 = _as_float(c, "relaxed_pair_f1")
            d_rf1 = cand_rf1 - base_rf1

            base_gt = _as_float(b, "gt_pair_recovery_ratio")
            cand_gt = _as_float(c, "gt_pair_recovery_ratio")
            d_gt = cand_gt - base_gt

            base_oracle = _as_float(b, "raw_pair_oracle_f1")
            cand_oracle = _as_float(c, "raw_pair_oracle_f1")
            d_oracle = cand_oracle - base_oracle

            base_exact = _as_float(b, "exact_triple_f1")
            cand_exact = _as_float(c, "exact_triple_f1")
            d_exact = cand_exact - base_exact

            base_pred = _as_int(b, "n_pred_pairs")
            cand_pred = _as_int(c, "n_pred_pairs")
            d_pred = cand_pred - base_pred
            growth = max(0.0, d_pred / max(1, base_pred))

            score = (
                weight_relaxed * d_rf1
                + weight_gt * d_gt
                + weight_oracle * d_oracle
                + weight_exact * d_exact
                - weight_pred_growth * growth
            )

            payload = {
                "run_name": run_name,
                "base_rf1": base_rf1,
                "cand_rf1": cand_rf1,
                "d_rf1": d_rf1,
                "base_gt": base_gt,
                "cand_gt": cand_gt,
                "d_gt": d_gt,
                "base_oracle": base_oracle,
                "cand_oracle": cand_oracle,
                "d_oracle": d_oracle,
                "base_exact": base_exact,
                "cand_exact": cand_exact,
                "d_exact": d_exact,
                "base_pred": base_pred,
                "cand_pred": cand_pred,
                "d_pred": d_pred,
                "growth": growth,
                "score": score,
                "patient_id": str(c.get("patient_id", "")),
            }
            if best_payload is None or payload["score"] > best_payload["score"]:
                best_payload = payload

        if best_payload is None:
            continue

        deltas.append(
            CandidateDelta(
                selection_reason="delta_vs_baseline",
                dataset=group.dataset,
                admission_id=admission_id,
                patient_id=best_payload["patient_id"],
                baseline_run=group.baseline_name,
                best_run=best_payload["run_name"],
                base_relaxed_pair_f1=best_payload["base_rf1"],
                best_relaxed_pair_f1=best_payload["cand_rf1"],
                delta_relaxed_pair_f1=best_payload["d_rf1"],
                base_gt_recovery=best_payload["base_gt"],
                best_gt_recovery=best_payload["cand_gt"],
                delta_gt_recovery=best_payload["d_gt"],
                base_oracle_pair_f1=best_payload["base_oracle"],
                best_oracle_pair_f1=best_payload["cand_oracle"],
                delta_oracle_pair_f1=best_payload["d_oracle"],
                base_exact_triple_f1=best_payload["base_exact"],
                best_exact_triple_f1=best_payload["cand_exact"],
                delta_exact_triple_f1=best_payload["d_exact"],
                base_pred_pairs=best_payload["base_pred"],
                best_pred_pairs=best_payload["cand_pred"],
                delta_pred_pairs=best_payload["d_pred"],
                pred_pair_growth_ratio=best_payload["growth"],
                score=best_payload["score"],
            )
        )
    return deltas


def _contextual_absolute_candidates(
    contextual_results: Path,
    min_relaxed_f1: float,
    min_oracle_f1: float,
    min_pred_pairs: int,
    top_k: int,
    score_relaxed_weight: float,
    score_oracle_weight: float,
) -> List[CandidateDelta]:
    rows = list(csv.DictReader(contextual_results.open("r", encoding="utf-8", newline="")))
    picked: List[CandidateDelta] = []
    for r in rows:
        relaxed_f1 = _as_float(r, "relaxed_pair_f1")
        oracle_f1 = _as_float(r, "raw_pair_oracle_f1")
        pred_pairs = _as_int(r, "n_pred_pairs")
        if relaxed_f1 < min_relaxed_f1:
            continue
        if oracle_f1 < min_oracle_f1:
            continue
        if pred_pairs < min_pred_pairs:
            continue

        absolute_score = score_relaxed_weight * relaxed_f1 + score_oracle_weight * oracle_f1
        picked.append(
            CandidateDelta(
                selection_reason="contextual_absolute",
                dataset="mimic",
                admission_id=str(r.get("admission_id", "")),
                patient_id=str(r.get("patient_id", "")),
                baseline_run="contextual_absolute_baseline0",
                best_run="loki_batch_mimic_contextual_best",
                base_relaxed_pair_f1=0.0,
                best_relaxed_pair_f1=relaxed_f1,
                delta_relaxed_pair_f1=relaxed_f1,
                base_gt_recovery=0.0,
                best_gt_recovery=_as_float(r, "gt_pair_recovery_ratio"),
                delta_gt_recovery=_as_float(r, "gt_pair_recovery_ratio"),
                base_oracle_pair_f1=0.0,
                best_oracle_pair_f1=oracle_f1,
                delta_oracle_pair_f1=oracle_f1,
                base_exact_triple_f1=0.0,
                best_exact_triple_f1=_as_float(r, "exact_triple_f1"),
                delta_exact_triple_f1=_as_float(r, "exact_triple_f1"),
                base_pred_pairs=0,
                best_pred_pairs=pred_pairs,
                delta_pred_pairs=pred_pairs,
                pred_pair_growth_ratio=0.0,
                score=absolute_score,
            )
        )

    picked.sort(
        key=lambda x: (
            x.score,
            x.delta_relaxed_pair_f1,
            x.delta_oracle_pair_f1,
            x.best_pred_pairs,
        ),
        reverse=True,
    )
    if top_k > 0:
        picked = picked[:top_k]
    return picked


def _merge_unique_by_admission(rows: Sequence[CandidateDelta]) -> List[CandidateDelta]:
    by_id: Dict[str, CandidateDelta] = {}
    for row in rows:
        existing = by_id.get(row.admission_id)
        if existing is None:
            by_id[row.admission_id] = row
            continue
        if row.score > existing.score:
            by_id[row.admission_id] = row
    merged = list(by_id.values())
    merged.sort(key=lambda x: (x.score, x.delta_relaxed_pair_f1, x.delta_oracle_pair_f1), reverse=True)
    return merged


def _select_candidates(
    deltas: Iterable[CandidateDelta],
    min_relaxed_for_relaxed_rule: float,
    min_gt_for_gt_rule: float,
    min_oracle_for_oracle_rule: float,
    min_relaxed_floor_for_gt_or_oracle_rule: float,
    min_score: float,
    max_exact_f1_drop: float,
    top_k: int,
) -> List[CandidateDelta]:
    picked = []
    for d in deltas:
        pass_relaxed_rule = d.delta_relaxed_pair_f1 >= min_relaxed_for_relaxed_rule
        pass_gt_rule = (
            d.delta_gt_recovery >= min_gt_for_gt_rule
            and d.delta_relaxed_pair_f1 >= min_relaxed_floor_for_gt_or_oracle_rule
        )
        pass_oracle_rule = (
            d.delta_oracle_pair_f1 >= min_oracle_for_oracle_rule
            and d.delta_relaxed_pair_f1 >= min_relaxed_floor_for_gt_or_oracle_rule
        )
        if not (pass_relaxed_rule or pass_gt_rule or pass_oracle_rule):
            continue
        if d.score < min_score:
            continue
        if d.delta_exact_triple_f1 < -abs(max_exact_f1_drop):
            continue
        picked.append(d)

    picked.sort(
        key=lambda x: (
            x.score,
            x.delta_relaxed_pair_f1,
            x.delta_gt_recovery,
            x.delta_oracle_pair_f1,
        ),
        reverse=True,
    )
    if top_k > 0:
        picked = picked[:top_k]
    return picked


def _apply_top_k_per_dataset(selected: Sequence[CandidateDelta], top_k_per_dataset: int) -> List[CandidateDelta]:
    if top_k_per_dataset <= 0:
        return list(selected)
    grouped: Dict[str, List[CandidateDelta]] = {}
    for row in selected:
        grouped.setdefault(row.dataset, []).append(row)
    trimmed: List[CandidateDelta] = []
    for dataset, rows in grouped.items():
        rows.sort(key=lambda x: (x.score, x.delta_relaxed_pair_f1, x.delta_gt_recovery), reverse=True)
        trimmed.extend(rows[:top_k_per_dataset])
    trimmed.sort(key=lambda x: (x.score, x.delta_relaxed_pair_f1, x.delta_gt_recovery), reverse=True)
    return trimmed


def _write_selection(out_dir: Path, selected: List[CandidateDelta]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "selected_admissions.csv"
    json_path = out_dir / "selected_admissions.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(selected[0]).keys()) if selected else ["admission_id"])
        writer.writeheader()
        for row in selected:
            writer.writerow(asdict(row))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in selected], f, indent=2)


def _filter_examples(source_test: Path, admission_ids: set[str]) -> List[dict]:
    with source_test.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return [r for r in records if str(r.get("admission_id", "")) in admission_ids]


def _filter_annotations(source_annot: Path, admission_ids: set[str]) -> Dict[str, dict]:
    with source_annot.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    return {k: v for k, v in annotations.items() if str(k) in admission_ids}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Datasets/mimic_ex3 from best-performing admissions.")
    parser.add_argument("--out-dir", default="Datasets/mimic_ex3")
    parser.add_argument(
        "--contextual-results",
        default=(
            "Batch_Materialization/"
            "loki_batch_mimic_contextual_best/"
            "materialized_batch_results_mimic.csv"
        ),
    )

    parser.add_argument("--min-relaxed-for-relaxed-rule", type=float, default=0.01)
    parser.add_argument("--min-gt-for-gt-rule", type=float, default=0.10)
    parser.add_argument("--min-oracle-for-oracle-rule", type=float, default=0.10)
    parser.add_argument("--min-relaxed-floor-for-gt-or-oracle-rule", type=float, default=-0.02)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--max-exact-f1-drop",
        type=float,
        default=1.0,
        help="Drop candidates whose exact_triple_f1 delta is lower than -this value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Limit output to top-K candidates by score (0 means keep all that pass filters).",
    )
    parser.add_argument(
        "--top-k-per-dataset",
        type=int,
        default=0,
        help="Limit each dataset to top-K admissions after filtering (0 means keep all).",
    )

    parser.add_argument("--weight-relaxed", type=float, default=0.60)
    parser.add_argument("--weight-gt", type=float, default=0.25)
    parser.add_argument("--weight-oracle", type=float, default=0.15)
    parser.add_argument("--weight-exact", type=float, default=0.05)
    parser.add_argument("--weight-pred-growth", type=float, default=0.05)

    parser.add_argument(
        "--disable-contextual-augment",
        action="store_true",
        help="Disable adding high absolute-F1 mimic admissions from contextual_best.",
    )
    parser.add_argument(
        "--include-mimic-small-selection",
        action="store_true",
        help="Include admissions whose winning selection row is from mimic_small (default excludes them).",
    )
    parser.add_argument(
        "--include-mimic-small-source",
        action="store_true",
        help="Include examples/annotations from Datasets/mimic_small in output files (default excludes them).",
    )
    parser.add_argument("--contextual-min-relaxed-f1", type=float, default=0.45)
    parser.add_argument("--contextual-min-oracle-f1", type=float, default=0.65)
    parser.add_argument("--contextual-min-pred-pairs", type=int, default=8)
    parser.add_argument("--contextual-top-k", type=int, default=30)
    parser.add_argument("--contextual-score-relaxed-weight", type=float, default=0.70)
    parser.add_argument("--contextual-score-oracle-weight", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = _resolve(args.out_dir)
    contextual_results = _resolve(args.contextual_results)

    groups = _default_run_groups()
    deltas: List[CandidateDelta] = []
    for group in groups:
        deltas.extend(
            _best_delta_for_group(
                group,
                weight_relaxed=args.weight_relaxed,
                weight_gt=args.weight_gt,
                weight_oracle=args.weight_oracle,
                weight_exact=args.weight_exact,
                weight_pred_growth=args.weight_pred_growth,
            )
        )

    selected = _select_candidates(
        deltas,
        min_relaxed_for_relaxed_rule=args.min_relaxed_for_relaxed_rule,
        min_gt_for_gt_rule=args.min_gt_for_gt_rule,
        min_oracle_for_oracle_rule=args.min_oracle_for_oracle_rule,
        min_relaxed_floor_for_gt_or_oracle_rule=args.min_relaxed_floor_for_gt_or_oracle_rule,
        min_score=args.min_score,
        max_exact_f1_drop=args.max_exact_f1_drop,
        top_k=args.top_k,
    )
    selected = _apply_top_k_per_dataset(selected, args.top_k_per_dataset)

    contextual_selected: List[CandidateDelta] = []
    if not args.disable_contextual_augment:
        contextual_selected = _contextual_absolute_candidates(
            contextual_results=contextual_results,
            min_relaxed_f1=args.contextual_min_relaxed_f1,
            min_oracle_f1=args.contextual_min_oracle_f1,
            min_pred_pairs=args.contextual_min_pred_pairs,
            top_k=args.contextual_top_k,
            score_relaxed_weight=args.contextual_score_relaxed_weight,
            score_oracle_weight=args.contextual_score_oracle_weight,
        )

    selected = _merge_unique_by_admission([*selected, *contextual_selected])
    if not args.include_mimic_small_selection:
        selected = [row for row in selected if row.dataset != "mimic_small"]

    out_dir.mkdir(parents=True, exist_ok=True)
    if not selected:
        _write_selection(out_dir, selected)
        (out_dir / "test_row_level.json").write_text("[]\n", encoding="utf-8")
        (out_dir / "Annotated_Test.json").write_text("{}\n", encoding="utf-8")
        print("No admissions met selection criteria.")
        return

    selected_ids = {d.admission_id for d in selected}
    _write_selection(out_dir, selected)

    winner_counts = dict(Counter(d.dataset for d in selected))
    reason_counts = dict(Counter(d.selection_reason for d in selected))

    merged_examples: List[dict] = []
    merged_annotations: Dict[str, dict] = {}
    source_coverage_counts: Dict[str, int] = {}
    for group in groups:
        if (not args.include_mimic_small_source) and group.dataset == "mimic_small":
            continue
        examples = _filter_examples(group.source_test, selected_ids)
        annotations = _filter_annotations(group.source_annotations, selected_ids)
        merged_examples.extend(examples)
        merged_annotations.update(annotations)
        source_coverage_counts[group.dataset] = len({x["admission_id"] for x in examples})

    with (out_dir / "test_row_level.json").open("w", encoding="utf-8") as f:
        json.dump(merged_examples, f, indent=2)
    with (out_dir / "Annotated_Test.json").open("w", encoding="utf-8") as f:
        json.dump(merged_annotations, f, indent=2)

    with (out_dir / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_admissions": len(selected_ids),
                "selected_examples": len(merged_examples),
                "selected_annotations": len(merged_annotations),
                "winning_dataset_counts": winner_counts,
                "selection_reason_counts": reason_counts,
                "source_coverage_admissions": source_coverage_counts,
            },
            f,
            indent=2,
        )

    print(f"Selected admissions: {len(selected_ids)}")
    print("IDs:", ", ".join(sorted(selected_ids, key=int)))
    print(f"Winning dataset counts: {winner_counts}")
    print(f"Selection reason counts: {reason_counts}")
    print(f"Source coverage counts: {source_coverage_counts}")
    print(f"Saved examples   : {len(merged_examples)} -> {out_dir / 'test_row_level.json'}")
    print(f"Saved annotations: {len(merged_annotations)} -> {out_dir / 'Annotated_Test.json'}")
    print(f"Saved manifest   : {out_dir / 'selected_admissions.csv'}")


if __name__ == "__main__":
    main()
