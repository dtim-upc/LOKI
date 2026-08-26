#!/usr/bin/env python3
"""
Repair Out-of-Bounds Sentence Indices in Annotation Files

Problem: LLM annotators produced sentence indices that exceed the valid range for
their respective documents. For example, a document with 128 sentences (valid
indices 0-127) may have grounding entries referencing index 680.

Root cause: Annotators were likely given the full raw discharge note, but the
pipeline source data (test_row_level_v2.json) contains only a windowed subset.
The OOB indices reference sentences outside that window — those sentences are no
longer available, so the only safe repair is to strip them.

Repair strategy:
  - row_grounding: remove OOB sentence indices; trim mention_types to match;
    if ALL sentences removed, mark row with _oob_all_removed=True (row kept for
    auditing but treated as ungrounded).
  - relationships.evidence_sentences: remove OOB indices;
    if ALL evidence removed, downgrade evidence_scope to "document" and add
    _oob_evidence_stripped=True to the relationship.

Output: repaired files written to output directory (default: Annotations/Individual_Repaired).
Original files are never modified. A JSON repair report is written alongside.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# Source data helpers
# ---------------------------------------------------------------------------

def build_admission_sentence_bounds(data_paths: List[str]) -> Dict[str, int]:
    """
    Build a map of admission_id -> max_valid_sentence_index from one or more
    row-level v2 data files (test / val / train).
    """
    bounds: Dict[str, int] = {}
    for path_str in data_paths:
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] Data file not found: {path_str}")
            continue
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for ex in data:
            adm = str(ex.get('admission_id', ''))
            sents = ex.get('primary_positive', {}).get('sentences', {})
            if not adm or not sents:
                continue
            max_key = max(int(k) for k in sents.keys())
            # Keep the highest max if an admission appears in multiple splits
            if adm not in bounds or max_key > bounds[adm]:
                bounds[adm] = max_key
    return bounds


# ---------------------------------------------------------------------------
# Repair logic
# ---------------------------------------------------------------------------

def repair_grounding_row(
    row_data: Dict[str, Any],
    max_valid: int,
    row_key: str,
    entity_type: str,
) -> Tuple[Dict[str, Any], int, int, bool]:
    """
    Repair a single grounding row by stripping OOB and None sentence indices.

    Returns (repaired_row, n_valid_kept, n_invalid_stripped, was_originally_empty).
    - n_invalid_stripped counts both OOB indices and None values.
    - was_originally_empty is True when sentences was [] before any stripping.
    When n_valid_kept == 0, the caller should drop this row entirely.
    """
    raw: List = row_data.get('sentences', [])
    was_originally_empty = len(raw) == 0

    # Filter out None values (treat as invalid references)
    n_null = sum(1 for s in raw if s is None)
    sentences: List[int] = [int(s) for s in raw if s is not None]
    mention_types: List[str] = list(row_data.get('mention_types', []))

    valid_pairs = [
        (s, mention_types[i] if i < len(mention_types) else 'explicit')
        for i, s in enumerate(sentences)
        if s <= max_valid
    ]
    n_invalid = (len(sentences) - len(valid_pairs)) + n_null

    repaired = dict(row_data)  # shallow copy, preserves provenance fields etc.
    repaired['sentences'] = [p[0] for p in valid_pairs]
    repaired['mention_types'] = [p[1] for p in valid_pairs]

    return repaired, len(valid_pairs), n_invalid, was_originally_empty


def repair_relationship(
    rel: Dict[str, Any],
    max_valid: int,
) -> Tuple[Dict[str, Any], int, int]:
    """
    Repair a relationship by stripping OOB and None evidence_sentences.

    Returns (repaired_rel, n_valid_kept, n_invalid_stripped).
    When n_valid_kept == 0, the caller should drop this relationship entirely.
    """
    raw: List = rel.get('evidence_sentences', [])
    ev: List[int] = [int(s) for s in raw if s is not None]
    n_null = len(raw) - len(ev)
    valid_ev = [s for s in ev if s <= max_valid]
    n_invalid = (len(ev) - len(valid_ev)) + n_null

    repaired = dict(rel)
    repaired['evidence_sentences'] = valid_ev

    return repaired, len(valid_ev), n_invalid


def repair_annotation(
    annotation: Dict[str, Any],
    max_valid: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Repair all sentence indices in a single annotation dict.

    Returns (repaired_annotation, per_field_stats).
    """
    stats: Dict[str, int] = {
        'diag_rows_total': 0,
        'diag_rows_fully_stripped': 0,   # had sentences, all became invalid (OOB/None)
        'diag_rows_empty_dropped': 0,    # sentences was [] to begin with
        'diag_oob_refs': 0,
        'med_rows_total': 0,
        'med_rows_fully_stripped': 0,
        'med_rows_empty_dropped': 0,
        'med_oob_refs': 0,
        'rel_total': 0,
        'rel_evidence_stripped': 0,      # had evidence, all became invalid
        'rel_empty_evidence_dropped': 0, # evidence_sentences was [] to begin with
        'rel_oob_refs': 0,
    }

    repaired = dict(annotation)

    # --- row_grounding ---
    rg = annotation.get('row_grounding', {})
    repaired_rg: Dict[str, Any] = {}

    for entity_type, stat_prefix in [('diagnosis', 'diag'), ('medication', 'med')]:
        repaired_entity: Dict[str, Any] = {}
        for row_key, row_data in rg.get(entity_type, {}).items():
            stats[f'{stat_prefix}_rows_total'] += 1
            fixed, n_kept, n_invalid, was_empty = repair_grounding_row(
                row_data, max_valid, row_key, entity_type
            )
            stats[f'{stat_prefix}_oob_refs'] += n_invalid
            if n_kept == 0:
                # Drop the row — no valid sentences remain (empty or all invalid)
                if was_empty:
                    stats[f'{stat_prefix}_rows_empty_dropped'] += 1
                else:
                    stats[f'{stat_prefix}_rows_fully_stripped'] += 1
            else:
                repaired_entity[row_key] = fixed
        repaired_rg[entity_type] = repaired_entity

    repaired['row_grounding'] = repaired_rg

    # --- relationships ---
    repaired_rels = []
    for rel in annotation.get('relationships', []):
        stats['rel_total'] += 1
        raw_ev = rel.get('evidence_sentences', [])
        was_empty_ev = len(raw_ev) == 0
        fixed_rel, n_kept, n_invalid = repair_relationship(rel, max_valid)
        stats['rel_oob_refs'] += n_invalid
        if n_kept == 0:
            # Drop the relationship — no valid evidence sentences remain
            if was_empty_ev:
                stats['rel_empty_evidence_dropped'] += 1
            else:
                stats['rel_evidence_stripped'] += 1
        else:
            repaired_rels.append(fixed_rel)

    repaired['relationships'] = repaired_rels

    return repaired, stats


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main repair pipeline
# ---------------------------------------------------------------------------

def repair_annotator_directory(
    annotator_dir: Path,
    output_dir: Path,
    bounds: Dict[str, int],
    quiet: bool = False,
) -> Dict[str, Any]:
    """Repair all annotation files in one annotator directory."""

    overall_stats: Dict[str, int] = defaultdict(int)
    per_file_report: List[Dict] = []
    unknown_admissions: List[str] = []

    json_files = sorted(annotator_dir.rglob('*.json'))
    overall_stats['total_files'] = len(json_files)

    for jf in json_files:
        try:
            ann = load_json(jf)
        except json.JSONDecodeError as e:
            if not quiet:
                print(f"  [SKIP] Malformed JSON: {jf.name} — {e}")
            overall_stats['malformed'] += 1
            continue

        adm = str(ann.get('admission_id', ''))
        if not adm:
            overall_stats['no_admission_id'] += 1
            continue

        max_valid = bounds.get(adm)
        if max_valid is None:
            unknown_admissions.append(adm)
            overall_stats['unknown_admission'] += 1
            # Still copy the file unchanged
            rel_path = jf.relative_to(annotator_dir)
            save_json(ann, output_dir / rel_path)
            continue

        repaired, stats = repair_annotation(ann, max_valid)

        total_oob = stats['diag_oob_refs'] + stats['med_oob_refs'] + stats['rel_oob_refs']
        if total_oob > 0:
            overall_stats['files_with_oob'] += 1
            overall_stats['total_oob_refs'] += total_oob
        else:
            overall_stats['files_clean'] += 1

        for k, v in stats.items():
            overall_stats[k] += v

        # Add repair metadata to the annotation
        if total_oob > 0:
            repaired['_repair_metadata'] = {
                'max_valid_sentence_idx': max_valid,
                'repair_timestamp': datetime.now().isoformat(),
            }

        # Save to output directory preserving subdirectory structure
        rel_path = jf.relative_to(annotator_dir)
        save_json(repaired, output_dir / rel_path)

        if total_oob > 0 and not quiet:
            print(f"  [{adm}] Stripped {total_oob} OOB refs "
                  f"(diag:{stats['diag_oob_refs']} "
                  f"med:{stats['med_oob_refs']} "
                  f"rel:{stats['rel_oob_refs']})")

        per_file_report.append({
            'admission_id': adm,
            'file': str(rel_path),
            'max_valid_idx': max_valid,
            'stats': dict(stats),
        })

    return {
        'annotator': annotator_dir.name,
        'overall_stats': dict(overall_stats),
        'unknown_admissions': list(set(unknown_admissions)),
        'per_file': per_file_report,
    }


def print_annotator_summary(report: Dict[str, Any]) -> None:
    s = report['overall_stats']
    name = report['annotator']
    total = s.get('total_files', 0)
    clean = s.get('files_clean', 0)
    oob_files = s.get('files_with_oob', 0)
    oob_refs = s.get('total_oob_refs', 0)

    print(f"\n  {name}")
    print(f"    Files total:                    {total}")
    print(f"    Files clean (no OOB):           {clean}")
    print(f"    Files repaired (had OOB):       {oob_files}")
    print(f"    Total invalid refs stripped:    {oob_refs}")
    print(f"    Diag rows dropped (OOB/None):   {s.get('diag_rows_fully_stripped', 0)}")
    print(f"    Diag rows dropped (was empty):  {s.get('diag_rows_empty_dropped', 0)}")
    print(f"    Med rows dropped (OOB/None):    {s.get('med_rows_fully_stripped', 0)}")
    print(f"    Med rows dropped (was empty):   {s.get('med_rows_empty_dropped', 0)}")
    print(f"    Rels dropped (ev OOB/None):     {s.get('rel_evidence_stripped', 0)}")
    print(f"    Rels dropped (ev was empty):    {s.get('rel_empty_evidence_dropped', 0)}")
    if report['unknown_admissions']:
        print(f"    Unknown admissions:             {len(report['unknown_admissions'])} "
              f"(copied unchanged)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strip out-of-bounds sentence indices from all individual annotation files."
    )
    parser.add_argument(
        '--input_dir', '-i', type=str,
        default='Annotations/Individual',
        help='Directory containing per-annotator subdirectories (default: Annotations/Individual)',
    )
    parser.add_argument(
        '--output_dir', '-o', type=str,
        default='Annotations/Individual_Repaired',
        help='Output directory for repaired files (default: Annotations/Individual_Repaired)',
    )
    parser.add_argument(
        '--data_files', '-d', type=str, nargs='+',
        default=[
            'mimic_data/test_row_level_v2.json',
            'mimic_data/val_row_level_v2.json',
            'mimic_data/train_row_level_v2.json',
        ],
        help='One or more row-level v2 JSON files used to determine valid sentence bounds',
    )
    parser.add_argument(
        '--report', '-r', type=str,
        default='Annotations/oob_repair_report.json',
        help='Path to write the JSON repair report',
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress per-file output',
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_dir  = script_dir / args.input_dir
    output_dir = script_dir / args.output_dir
    report_path = script_dir / args.report

    print("=" * 72)
    print("  OOB SENTENCE INDEX REPAIR")
    print("=" * 72)

    # Build sentence bounds from source data
    data_paths = [str(script_dir / p) for p in args.data_files]
    print("\n[STEP 1] Building sentence bounds from source data...")
    bounds = build_admission_sentence_bounds(data_paths)
    print(f"  Loaded bounds for {len(bounds)} admissions")

    if not bounds:
        print("[ERROR] No sentence bounds could be loaded. Check --data_files paths.")
        return 1

    # Find annotator subdirectories
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1

    annotator_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not annotator_dirs:
        print(f"[ERROR] No annotator subdirectories found in {input_dir}")
        return 1

    print(f"\n[STEP 2] Repairing annotations from {len(annotator_dirs)} annotators...")
    print(f"  Output directory: {output_dir}")

    all_reports: List[Dict] = []

    for ann_dir in annotator_dirs:
        ann_output = output_dir / ann_dir.name
        if not args.quiet:
            print(f"\n  Processing: {ann_dir.name}")

        report = repair_annotator_directory(ann_dir, ann_output, bounds, quiet=args.quiet)
        all_reports.append(report)

    # Print summary
    print("\n" + "=" * 72)
    print("  REPAIR SUMMARY")
    print("=" * 72)

    grand_total_files = 0
    grand_oob_refs = 0
    for rep in all_reports:
        print_annotator_summary(rep)
        grand_total_files += rep['overall_stats'].get('total_files', 0)
        grand_oob_refs += rep['overall_stats'].get('total_oob_refs', 0)

    print(f"\n  Grand total files:        {grand_total_files}")
    print(f"  Grand total OOB refs:     {grand_oob_refs}")
    print(f"  Original files: UNCHANGED (in {input_dir})")
    print(f"  Repaired files: written to {output_dir}")
    print("=" * 72)

    # Save report
    full_report = {
        'repair_timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'data_files': data_paths,
        'admissions_with_bounds': len(bounds),
        'annotators': all_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2)
    print(f"\n  Detailed report saved to: {report_path}")

    return 0


if __name__ == '__main__':
    exit(main())
