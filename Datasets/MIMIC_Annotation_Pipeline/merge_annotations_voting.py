#!/usr/bin/env python3
"""
Majority Voting Annotation Merge Tool

This script merges annotations from multiple annotators using majority voting.
For each relationship, it requires at least half+1 annotators to agree for inclusion.

Features:
1. Majority voting (>=50%+1 of annotators must agree)
2. Provenance tracking (which annotators agreed)
3. Confidence scoring based on agreement level
4. Grounding sentence merging (union of agreed sentences)
5. Detailed merge report

Output:
- Merged annotation files
- Merge summary report
- Provenance tracking JSON
"""

import json
import os
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def load_annotations(path_str: str) -> Dict[str, Any]:
    """Load annotations from either a folder of JSON files or a unified master JSON file."""
    annotations = {}
    path = Path(path_str)
    
    if not path.exists():
        return {}
    
    if path.is_file() and path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return {}
                
            # Master template format
            if isinstance(data, dict) and "annotations" in data:
                for entry in data["annotations"]:
                    if entry.get("status") == "completed":
                        adm_id = entry.get("admission_id")
                        if not adm_id and "anchor_metadata" in entry:
                            parts = entry["anchor_metadata"].split("-")
                            if len(parts) >= 3:
                                adm_id = parts[1]
                        if adm_id:
                            annotations[adm_id] = entry
            # Dictionary format
            elif isinstance(data, dict):
                for k, v in data.items():
                    annotations[k] = v
                    
    elif path.is_dir():
        adm_id_sources: Dict[str, str] = {}  # adm_id -> filename that first claimed it
        for json_file in sorted(path.rglob("*.json")):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
                
                # Standalone annotation file
                if isinstance(data, list):
                    for entry in data:
                        adm_id = entry.get('admission_id', json_file.stem)
                        if adm_id.startswith('annotation_'): adm_id = adm_id[11:]
                        if adm_id in adm_id_sources:
                            print(f"[LOAD-WARN] {json_file.name}  —  admission_id '{adm_id}' "
                                  f"collides with '{adm_id_sources[adm_id]}', overwriting")
                        else:
                            adm_id_sources[adm_id] = json_file.name
                        annotations[adm_id] = entry
                elif isinstance(data, dict):
                    raw_adm_id = data.get('admission_id')
                    if raw_adm_id is None:
                        adm_id = json_file.stem
                        print(f"[LOAD-WARN] {json_file.name}  —  no 'admission_id' field, "
                              f"falling back to filename stem '{adm_id}'")
                    else:
                        adm_id = raw_adm_id
                    if adm_id.startswith('annotation_'):
                        adm_id = adm_id[11:]
                    if adm_id in adm_id_sources:
                        print(f"[LOAD-WARN] {json_file.name}  —  admission_id '{adm_id}' "
                              f"collides with '{adm_id_sources[adm_id]}', overwriting")
                    else:
                        adm_id_sources[adm_id] = json_file.name
                    annotations[adm_id] = data

    return annotations


def get_relationship_key(rel: Dict) -> Tuple:
    """Create a hashable key for a relationship (drug, diagnosis, type)."""
    return (rel.get('drug_row'), rel.get('diagnosis_row'), rel.get('relationship_type'))


def get_relationship_pair_key(rel: Dict) -> Tuple:
    """Create a key for drug-diagnosis pair (ignoring relationship type)."""
    return (rel.get('drug_row'), rel.get('diagnosis_row'))


def merge_grounding(grounding_list: List[Dict], entity_type: str,
                    n_annotators: int, voting_threshold: float = 0.5,
                    annotator_names: Optional[List[str]] = None) -> Dict:
    """
    Merge grounding from multiple annotators using majority voting.
    Only includes groundings where majority of annotators agree.
    
    Args:
        grounding_list: List of grounding dicts from each annotator
        entity_type: 'diagnosis' or 'medication'
        n_annotators: Total number of annotators
        voting_threshold: Minimum fraction of annotators required (default 0.5)
        annotator_names: Ordered list of annotator names matching grounding_list;
                         used for _sources provenance. Falls back to positional
                         labels if not provided.
    """
    if not grounding_list:
        return {}
    
    merged = {}
    # ceil(n * threshold) gives the correct minimum, max(2,...) enforces at least 2 must agree.
    # Using ceil avoids int()+1 over-counting (e.g. threshold=1.0 with int()+1 would require n+1 votes).
    min_votes = max(2, math.ceil(n_annotators * voting_threshold))
    
    def _name(i: int) -> str:
        if annotator_names and i < len(annotator_names):
            return annotator_names[i]
        return f"annotator_{i + 1}"

    # Collect all row keys
    all_rows = set()
    for grounding in grounding_list:
        if entity_type in grounding:
            all_rows.update(grounding[entity_type].keys())
    
    for row_key in all_rows:
        # Track which annotators have this row and their sentences
        annotator_indices = []
        sentence_votes = Counter()  # sentence -> number of annotators who found it
        mention_type_counts = defaultdict(Counter)  # sentence -> mention_type -> count
        
        for i, grounding in enumerate(grounding_list):
            if entity_type not in grounding:
                continue
            if row_key not in grounding[entity_type]:
                continue
            
            annotator_indices.append(i)
            row_data = grounding[entity_type][row_key]
            sentences = row_data.get('sentences', [])
            mention_types = row_data.get('mention_types', [])

            if sentences and mention_types and len(sentences) != len(mention_types):
                print(f"[WARN] merge_grounding: {entity_type} row {row_key} from {_name(i)}: "
                      f"{len(sentences)} sentences but {len(mention_types)} mention_types — "
                      f"excess sentences will fall back to 'explicit' mention type")

            for j, sent in enumerate(sentences):
                sentence_votes[sent] += 1
                if j < len(mention_types):
                    mention_type_counts[sent][mention_types[j]] += 1
        
        # Only include if majority of annotators have this row
        n_annotators_with_row = len(annotator_indices)
        if n_annotators_with_row < min_votes:
            continue  # Skip - not enough annotators agree on this row
        
        # Only include sentences that majority of those annotators agree on
        agreed_sentences = []
        for sent, vote_count in sentence_votes.items():
            # Sentence needs majority agreement among annotators who have this row
            if vote_count >= min_votes:
                agreed_sentences.append(sent)
        
        if not agreed_sentences:
            continue  # No sentences with majority agreement
        
        sorted_sentences = sorted(agreed_sentences)
        
        # Get majority mention type for each sentence
        mention_types = []
        for sent in sorted_sentences:
            if mention_type_counts[sent]:
                most_common = mention_type_counts[sent].most_common(1)[0][0]
                mention_types.append(most_common)
            else:
                mention_types.append('explicit')  # default
        
        # Build source list using real annotator names
        sources = [_name(i) for i in annotator_indices]
        
        merged[row_key] = {
            'sentences': sorted_sentences,
            'mention_types': mention_types,
            '_sources': sources,
            '_vote_count': n_annotators_with_row,
        }
    
    return merged


def merge_relationships(all_rels: Dict[Tuple, List[Dict]], 
                        n_annotators: int,
                        voting_threshold: float = 0.5,
                        include_reasoning: bool = True) -> List[Dict]:
    """
    Merge relationships using majority voting.
    
    Args:
        all_rels: Dict mapping relationship key to list of (annotator_name, relationship) tuples
        n_annotators: Total number of annotators
        voting_threshold: Minimum fraction of annotators required (default 0.5 = majority)
    
    Returns:
        List of merged relationships
    """
    merged = []
    min_votes = max(2, math.ceil(n_annotators * voting_threshold))

    rel_id = 1
    
    for rel_key, annotator_rels in all_rels.items():
        n_votes = len(annotator_rels)
        
        if n_votes < min_votes:
            continue  # Skip if not enough votes
        
        # Collect data from all annotators who agreed
        evidence_sentences = set()
        evidence_scopes = []
        reasonings = []
        confidences = []
        annotator_names = []
        
        for ann_name, rel in annotator_rels:
            annotator_names.append(ann_name)
            evidence_sentences.update(rel.get('evidence_sentences', []))
            evidence_scopes.append(rel.get('evidence_scope', 'document'))
            if rel.get('reasoning'):
                reasonings.append(rel.get('reasoning'))
            confidences.append(rel.get('confidence', 'medium'))
        
        # Determine merged evidence scope (most common)
        scope_counts = Counter(evidence_scopes)
        merged_scope = scope_counts.most_common(1)[0][0]
        
        # Determine merged confidence based on voting
        if n_votes == n_annotators:
            merged_confidence = 'high'  # Unanimous
            agreement_level = 'unanimous'
        elif n_votes >= n_annotators * 0.67:
            merged_confidence = 'high'
            agreement_level = 'strong_majority'
        else:
            merged_confidence = 'medium'
            agreement_level = 'majority'
        
        # Combine reasonings
        if not include_reasoning:
            merged_reasoning = "anonymized"
        elif reasonings:
            unique_reasonings = list(dict.fromkeys(reasonings))  # Remove duplicates, preserve order
            if len(unique_reasonings) == 1:
                merged_reasoning = unique_reasonings[0]
            else:
                merged_reasoning = " | ".join(unique_reasonings[:2])  # Combine first 2
        else:
            merged_reasoning = ""
        
        merged_rel = {
            'id': f'rel_{rel_id:03d}',
            'drug_row': rel_key[0],
            'diagnosis_row': rel_key[1],
            'relationship_type': rel_key[2],
            'evidence_sentences': sorted(evidence_sentences),
            'evidence_scope': merged_scope,
            'reasoning': merged_reasoning,
            'confidence': merged_confidence,
            '_provenance': {
                'vote_count': n_votes,
                'total_annotators': n_annotators,
                'agreement_level': agreement_level,
                'agreeing_annotators': annotator_names,
            }
        }
        
        merged.append(merged_rel)
        rel_id += 1
    
    return merged


def merge_multi_relationship_flags(all_flags: List[List[Dict]], include_reasoning: bool = True) -> List[Dict]:
    """Merge multi-relationship flags from all annotators."""
    # Collect unique flags
    seen = set()
    merged = []
    
    for flags in all_flags:
        for flag in flags:
            key = (flag.get('drug_row'), flag.get('diagnosis_row'), 
                   tuple(sorted(flag.get('relationship_types', []))))
            if key not in seen:
                seen.add(key)
                if not include_reasoning and 'note' in flag:
                    flag_copy = dict(flag)
                    flag_copy['note'] = 'anonymized'
                    merged.append(flag_copy)
                else:
                    merged.append(flag)
    
    return merged


def merge_negative_relationships(all_negatives: List[List[Dict]], include_reasoning: bool = True) -> List[Dict]:
    """Merge negative relationships from all annotators."""
    # Collect unique negatives (majority vote could be applied here too)
    seen = set()
    merged = []
    
    for negatives in all_negatives:
        for neg in negatives:
            key = (neg.get('drug_row'), neg.get('diagnosis_row'))
            if key not in seen:
                seen.add(key)
                if not include_reasoning and 'reason' in neg:
                    neg_copy = dict(neg)
                    neg_copy['reason'] = 'anonymized'
                    merged.append(neg_copy)
                else:
                    merged.append(neg)
    
    return merged


def _reconcile_one_row(
    row_key: str,
    entity_type: str,
    agreeing: List[str],
    all_annotations: Dict[str, Dict[str, Any]],
    file_id: str,
) -> Optional[Dict]:
    """
    Build a reconciled grounding row for *row_key* from all agreeing annotators
    that have valid (non-empty) sentences for it.

    Sentences are unioned across contributors; the first annotator's mention
    type wins for any given sentence index (stable, deterministic).
    All contributing annotator names are recorded in _sources.

    Returns the merged row dict, or None if no annotator has valid grounding.
    """
    contrib_names: List[str] = []
    # sentence_index -> mention_type  (first writer wins)
    sentence_to_mtype: Dict = {}
    base_row: Optional[Dict] = None

    for ann_name in agreeing:
        ann_data = all_annotations.get(ann_name, {}).get(file_id, {})
        grounding = ann_data.get('row_grounding', {}).get(entity_type, {})
        row = grounding.get(row_key)
        if not row or not row.get('sentences'):
            continue

        contrib_names.append(ann_name)
        if base_row is None:
            base_row = row  # retain other metadata from the first contributor

        sentences    = row.get('sentences', [])
        mention_types = row.get('mention_types', [])
        for j, sent in enumerate(sentences):
            if sent not in sentence_to_mtype:
                mtype = mention_types[j] if j < len(mention_types) else 'explicit'
                sentence_to_mtype[sent] = mtype

    if not contrib_names:
        return None

    merged_sentences = sorted(sentence_to_mtype.keys())
    entry = {k: v for k, v in base_row.items() if not k.startswith('_')}
    entry['sentences']     = merged_sentences
    entry['mention_types'] = [sentence_to_mtype[s] for s in merged_sentences]
    entry['_sources']      = contrib_names
    entry['_reconciled']   = True   # fell below voting threshold; union of available sources
    return entry


def reconcile_grounding_for_relationships(
    merged_relationships: List[Dict],
    merged_diag_grounding: Dict,
    merged_med_grounding: Dict,
    all_annotations: Dict[str, Dict[str, Any]],
    file_id: str,
) -> Tuple[List[Dict], Dict, Dict, int, int]:
    """
    Ensure every merged relationship has its drug_row in medication grounding
    and its diagnosis_row in diagnosis grounding.

    The independent voting thresholds for relationships and grounding rows can
    diverge: a relationship may pass with 2/3 votes while its referenced row
    only has 1/3 annotators grounding it (e.g. one annotator left sentences
    empty and it was stripped during repair).  This pass fixes that gap.

    Strategy:
      - If a referenced row is already in merged grounding → nothing to do.
      - If missing → collect the row from ALL agreeing annotators that have
        valid sentences for it, union their sentences, and record every
        contributor in _sources (marked _reconciled=True).
      - If no agreeing annotator has valid grounding for the row → drop the
        relationship entirely (it has no textual support) and log it.

    Returns:
        (reconciled_relationships, diag_grounding, med_grounding,
         n_rows_reconciled, n_rels_dropped)
    """
    reconciled_rels: List[Dict] = []
    n_rows_reconciled = 0
    n_rels_dropped = 0

    for rel in merged_relationships:
        agreeing: List[str] = rel['_provenance']['agreeing_annotators']
        drug_key  = str(rel['drug_row'])
        diag_key  = str(rel['diagnosis_row'])
        missing: List[str] = []

        # ── medication row ──────────────────────────────────────────────────
        if drug_key not in merged_med_grounding:
            entry = _reconcile_one_row(drug_key, 'medication', agreeing,
                                       all_annotations, file_id)
            if entry:
                merged_med_grounding[drug_key] = entry
                n_rows_reconciled += 1
            else:
                missing.append(f"drug_row {drug_key} (no valid grounding in any agreeing annotator)")

        # ── diagnosis row ────────────────────────────────────────────────────
        if diag_key not in merged_diag_grounding:
            entry = _reconcile_one_row(diag_key, 'diagnosis', agreeing,
                                       all_annotations, file_id)
            if entry:
                merged_diag_grounding[diag_key] = entry
                n_rows_reconciled += 1
            else:
                missing.append(f"diagnosis_row {diag_key} (no valid grounding in any agreeing annotator)")

        if missing:
            n_rels_dropped += 1
            print(f"  [RECONCILE-DROP] {file_id} {rel['id']}: dropped — "
                  + "; ".join(missing))
        else:
            reconciled_rels.append(rel)

    return reconciled_rels, merged_diag_grounding, merged_med_grounding, n_rows_reconciled, n_rels_dropped


def merge_annotations_for_file(file_id: str,
                                all_annotations: Dict[str, Dict[str, Any]],
                                voting_threshold: float = 0.5,
                                include_reasoning: bool = True) -> Tuple[Dict, Dict]:
    """
    Merge annotations for a single file from multiple annotators.
    
    Returns:
        Tuple of (merged_annotation, merge_stats)
    """
    annotator_names = list(all_annotations.keys())
    n_annotators = len(annotator_names)
    
    # Collect all data
    patient_ids = []
    diagnosis_anchors = []
    medication_anchors = []
    all_grounding = []
    all_grounding_names: List[str] = []  # parallel list: annotator name for each grounding entry
    all_rels = defaultdict(list)  # rel_key -> [(annotator, rel_data)]
    all_multi_flags = []
    all_negatives = []
    quality_notes = []
    
    for ann_name in annotator_names:
        annotations = all_annotations[ann_name]
        if file_id not in annotations:
            continue
        
        ann_data = annotations[file_id]
        
        patient_ids.append(ann_data.get('patient_id', ''))
        diagnosis_anchors.append(ann_data.get('diagnosis_anchor_id', ''))
        medication_anchors.append(ann_data.get('medication_anchor_id', ''))
        
        all_grounding.append(ann_data.get('row_grounding', {}))
        all_grounding_names.append(ann_name)
        
        seen_rel_keys: Set[Tuple] = set()
        for rel in ann_data.get('relationships', []):
            rel_key = get_relationship_key(rel)
            if rel_key in seen_rel_keys:
                print(f"[DEDUP] {ann_name}/{file_id}: duplicate relationship "
                      f"{rel_key} — keeping first occurrence, skipping duplicate")
                continue
            seen_rel_keys.add(rel_key)
            all_rels[rel_key].append((ann_name, rel))
        
        all_multi_flags.append(ann_data.get('multi_relationship_flags', []))
        all_negatives.append(ann_data.get('negative_relationships', []))
        
        if ann_data.get('quality_notes'):
            quality_notes.append(ann_data.get('quality_notes'))
    
    # Use most common values for metadata
    patient_id = Counter(patient_ids).most_common(1)[0][0] if patient_ids else ''
    diagnosis_anchor = Counter(diagnosis_anchors).most_common(1)[0][0] if diagnosis_anchors else ''
    medication_anchor = Counter(medication_anchors).most_common(1)[0][0] if medication_anchors else ''
    
    # Merge components (independent voting passes)
    merged_diagnosis_grounding = merge_grounding(
        all_grounding, 'diagnosis', n_annotators, voting_threshold,
        annotator_names=all_grounding_names,
    )
    merged_medication_grounding = merge_grounding(
        all_grounding, 'medication', n_annotators, voting_threshold,
        annotator_names=all_grounding_names,
    )
    merged_relationships = merge_relationships(all_rels, n_annotators, voting_threshold, include_reasoning=include_reasoning)

    # Reconciliation: fill grounding rows that are referenced by a voted-in
    # relationship but fell below the grounding voting threshold, then drop any
    # relationship whose rows cannot be grounded by any agreeing annotator.
    (merged_relationships,
     merged_diagnosis_grounding,
     merged_medication_grounding,
     n_rows_reconciled,
     n_rels_dropped_reconcile) = reconcile_grounding_for_relationships(
        merged_relationships,
        merged_diagnosis_grounding,
        merged_medication_grounding,
        all_annotations,
        file_id,
    )

    merged_multi_flags = merge_multi_relationship_flags(all_multi_flags, include_reasoning=include_reasoning)
    merged_negatives = merge_negative_relationships(all_negatives, include_reasoning=include_reasoning)
    
    # Build quality notes safely
    if not include_reasoning:
        q_notes = f"Merged from {n_annotators} annotators using majority voting. [anonymized extra notes]" if quality_notes else f"Merged from {n_annotators} annotators using majority voting."
    else:
        q_notes = f"Merged from {n_annotators} annotators using majority voting. " + " | ".join(quality_notes[:2]) if quality_notes else f"Merged from {n_annotators} annotators using majority voting."
    
    # Build merged annotation
    merged = {
        'patient_id': patient_id,
        'admission_id': file_id,
        'diagnosis_anchor_id': diagnosis_anchor,
        'medication_anchor_id': medication_anchor,
        'row_grounding': {
            'diagnosis': merged_diagnosis_grounding,
            'medication': merged_medication_grounding,
        },
        'relationships': merged_relationships,
        'multi_relationship_flags': merged_multi_flags,
        'negative_relationships': merged_negatives,
        'quality_notes': q_notes,
        '_merge_metadata': {
            'merge_timestamp': datetime.now().isoformat(),
            'n_annotators': n_annotators,
            'annotator_names': annotator_names,
            'voting_threshold': voting_threshold,
        }
    }
    
    # Compute stats
    total_candidate_rels = len(all_rels)
    included_rels = len(merged_relationships)
    # excluded = failed voting + dropped during reconciliation
    excluded_rels = total_candidate_rels - included_rels - n_rels_dropped_reconcile

    unanimous_rels = sum(1 for r in merged_relationships 
                         if r['_provenance']['agreement_level'] == 'unanimous')
    majority_rels = included_rels - unanimous_rels

    # Count grounding candidates vs included for audit purposes
    all_diag_rows: set = set()
    all_med_rows: set = set()
    for g in all_grounding:
        all_diag_rows.update(g.get('diagnosis', {}).keys())
        all_med_rows.update(g.get('medication', {}).keys())
    total_diag_candidates = len(all_diag_rows)
    total_med_candidates = len(all_med_rows)
    included_diag_rows = len(merged_diagnosis_grounding)
    included_med_rows = len(merged_medication_grounding)

    stats = {
        'file_id': file_id,
        'n_annotators': n_annotators,
        'total_candidate_relationships': total_candidate_rels,
        'included_relationships': included_rels,
        'excluded_relationships': excluded_rels,
        'reconcile_rows_filled': n_rows_reconciled,
        'reconcile_rels_dropped': n_rels_dropped_reconcile,
        'unanimous_relationships': unanimous_rels,
        'majority_relationships': majority_rels,
        'inclusion_rate': included_rels / total_candidate_rels * 100 if total_candidate_rels > 0 else 100,
        'total_candidate_diagnosis_rows': total_diag_candidates,
        'included_diagnosis_rows': included_diag_rows,
        'excluded_diagnosis_rows': total_diag_candidates - included_diag_rows,
        'total_candidate_medication_rows': total_med_candidates,
        'included_medication_rows': included_med_rows,
        'excluded_medication_rows': total_med_candidates - included_med_rows,
    }
    
    return merged, stats


def merge_all_annotations(all_annotations: Dict[str, Dict[str, Any]],
                          voting_threshold: float = 0.5,
                          include_reasoning: bool = True,
                          quiet: bool = False) -> Tuple[Dict[str, Dict], Dict]:
    """
    Merge annotations for all files.
    
    Returns:
        Tuple of (merged_annotations_dict, overall_stats)
    """
    annotator_names = list(all_annotations.keys())
    
    # Get all unique file IDs
    all_file_ids = set()
    for annotations in all_annotations.values():
        all_file_ids.update(annotations.keys())
    
    merged_annotations = {}
    all_stats = []
    
    for file_id in sorted(all_file_ids):
        # Check how many annotators have this file
        annotators_with_file = sum(1 for ann in all_annotations.values() if file_id in ann)
        
        if annotators_with_file < 2:
            if not quiet:
                print(f"  Skipping {file_id}: Only {annotators_with_file} annotator(s)")
            continue
        
        merged, stats = merge_annotations_for_file(file_id, all_annotations, voting_threshold, include_reasoning)
        merged_annotations[file_id] = merged
        all_stats.append(stats)
        
        if not quiet:
            reconcile_note = ""
            if stats.get('reconcile_rows_filled') or stats.get('reconcile_rels_dropped'):
                reconcile_note = (f" [reconcile: +{stats['reconcile_rows_filled']} rows filled"
                                  f", {stats['reconcile_rels_dropped']} rels dropped]")
            print(f"  Merged {file_id}: {stats['included_relationships']}/{stats['total_candidate_relationships']} relationships "
                  f"({stats['unanimous_relationships']} unanimous, {stats['majority_relationships']} majority)"
                  f"{reconcile_note}")
    
    # Compute overall statistics
    total_cand_rels = sum(s['total_candidate_relationships'] for s in all_stats)
    overall = {
        'total_files': len(merged_annotations),
        'total_relationships': sum(s['included_relationships'] for s in all_stats),
        'total_candidates': total_cand_rels,
        'total_excluded': sum(s['excluded_relationships'] for s in all_stats),
        'total_unanimous': sum(s['unanimous_relationships'] for s in all_stats),
        'total_majority': sum(s['majority_relationships'] for s in all_stats),
        'total_reconcile_rows_filled': sum(s.get('reconcile_rows_filled', 0) for s in all_stats),
        'total_reconcile_rels_dropped': sum(s.get('reconcile_rels_dropped', 0) for s in all_stats),
        'overall_inclusion_rate': (sum(s['included_relationships'] for s in all_stats) /
                                   total_cand_rels * 100) if total_cand_rels > 0 else 100,
        'total_candidate_diagnosis_rows': sum(s['total_candidate_diagnosis_rows'] for s in all_stats),
        'total_included_diagnosis_rows': sum(s['included_diagnosis_rows'] for s in all_stats),
        'total_excluded_diagnosis_rows': sum(s['excluded_diagnosis_rows'] for s in all_stats),
        'total_candidate_medication_rows': sum(s['total_candidate_medication_rows'] for s in all_stats),
        'total_included_medication_rows': sum(s['included_medication_rows'] for s in all_stats),
        'total_excluded_medication_rows': sum(s['excluded_medication_rows'] for s in all_stats),
        'per_file_stats': all_stats,
        'merge_config': {
            'voting_threshold': voting_threshold,
            'n_annotators': len(annotator_names),
            'annotator_names': annotator_names,
            'merge_timestamp': datetime.now().isoformat(),
        }
    }
    
    return merged_annotations, overall


def save_merged_annotations(merged_annotations: Dict[str, Dict],
                            output_dir: str,
                            save_individual: bool = True,
                            save_combined: bool = True):
    """Save merged annotations to files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if save_individual:
        # Save individual annotation files
        for file_id, annotation in merged_annotations.items():
            file_path = output_path / f"{file_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, indent=2)
        print(f"Saved {len(merged_annotations)} individual annotation files to: {output_path}")
    
    if save_combined:
        # Save combined file
        combined_path = output_path / "merged_annotations_all.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(merged_annotations, f, indent=2)
        print(f"Saved combined annotations to: {combined_path}")


def generate_provenance_report(merged_annotations: Dict[str, Dict],
                               overall_stats: Dict,
                               output_path: str):
    """Generate detailed provenance report."""
    
    report = {
        'summary': overall_stats,
        'per_file_details': {},
    }
    
    for file_id, annotation in merged_annotations.items():
        file_details = {
            'included_relationships': len(annotation['relationships']),
            'relationship_details': []
        }
        
        for rel in annotation['relationships']:
            file_details['relationship_details'].append({
                'id': rel['id'],
                'drug_row': rel['drug_row'],
                'diagnosis_row': rel['diagnosis_row'],
                'relationship_type': rel['relationship_type'],
                'vote_count': rel['_provenance']['vote_count'],
                'agreement_level': rel['_provenance']['agreement_level'],
                'agreeing_annotators': rel['_provenance']['agreeing_annotators'],
            })
        
        report['per_file_details'][file_id] = file_details
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Provenance report saved to: {output_path}")


def print_merge_report(overall_stats: Dict):
    """Print merge summary to console."""
    
    print("\n" + "="*80)
    print("  MAJORITY VOTING MERGE REPORT")
    print("="*80 + "\n")
    
    config = overall_stats['merge_config']
    
    print("[CONFIGURATION]")
    print("-" * 40)
    print(f"  Annotators:           {', '.join(config['annotator_names'])}")
    print(f"  Voting Threshold:     {config['voting_threshold']*100:.0f}% (>={int(config['n_annotators']*config['voting_threshold'])+1}/{config['n_annotators']})")
    print(f"  Merge Timestamp:      {config['merge_timestamp']}")
    print()
    
    print("[SUMMARY]")
    print("-" * 40)
    print(f"  Files Merged:              {overall_stats['total_files']}")
    print(f"  Total Candidate Rels:      {overall_stats['total_candidates']}")
    print(f"  Included Relationships:    {overall_stats['total_relationships']}")
    print(f"  Excluded (no majority):    {overall_stats['total_excluded']}")
    print(f"  Inclusion Rate:            {overall_stats['overall_inclusion_rate']:.1f}%")
    print()
    print("[RECONCILIATION]")
    print("-" * 40)
    print(f"  Grounding rows filled in:  {overall_stats.get('total_reconcile_rows_filled', 0)}")
    print(f"  Rels dropped (no support): {overall_stats.get('total_reconcile_rels_dropped', 0)}")
    print()

    print("[GROUNDING EXCLUSIONS]")
    print("-" * 40)
    print(f"  Diagnosis Rows (candidates):  {overall_stats['total_candidate_diagnosis_rows']}")
    print(f"  Diagnosis Rows (included):    {overall_stats['total_included_diagnosis_rows']}")
    print(f"  Diagnosis Rows (excluded):    {overall_stats['total_excluded_diagnosis_rows']}")
    print(f"  Medication Rows (candidates): {overall_stats['total_candidate_medication_rows']}")
    print(f"  Medication Rows (included):   {overall_stats['total_included_medication_rows']}")
    print(f"  Medication Rows (excluded):   {overall_stats['total_excluded_medication_rows']}")
    print()
    
    print("[AGREEMENT BREAKDOWN]")
    print("-" * 40)
    total = overall_stats['total_relationships']
    unanimous = overall_stats['total_unanimous']
    majority = overall_stats['total_majority']
    
    unanimous_pct = unanimous / total * 100 if total > 0 else 0
    majority_pct = majority / total * 100 if total > 0 else 0
    
    print(f"  Unanimous ({config['n_annotators']}/{config['n_annotators']}):      {unanimous:>5} ({unanimous_pct:.1f}%)")
    print(f"  Majority Only:             {majority:>5} ({majority_pct:.1f}%)")
    print()
    
    print("[PER-FILE BREAKDOWN]")
    print("-" * 40)
    print(f"  {'File ID':<15} {'Included':>10} {'Excluded':>10} {'Unanimous':>10} {'Rate':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for stats in overall_stats['per_file_stats']:
        print(f"  {stats['file_id']:<15} {stats['included_relationships']:>10} "
              f"{stats['excluded_relationships']:>10} {stats['unanimous_relationships']:>10} "
              f"{stats['inclusion_rate']:>9.1f}%")
    
    print()
    print("="*80)
    print("  END OF REPORT")


def main():
    parser = argparse.ArgumentParser(
        description="Merge annotations from multiple LLMs using majority voting."
    )
    
    parser.add_argument('--input_dir', '-i', type=str,
                       default='Annotations/Individual',
                       help='Directory containing annotator folders or master JSON files (default: Annotations/Individual)')
    parser.add_argument('--output', '-o',
                       default='Annotations/Voting',
                       help='Output directory for merged annotations (default: Annotations/Voting)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Voting threshold (fraction of annotators required, default: 0.5)')
    parser.add_argument('--no-individual', action='store_true',
                       help='Do not save individual annotation files')
    parser.add_argument('--no-combined', action='store_true',
                       help='Do not save combined annotation file')
    parser.add_argument('--include-reasoning', action='store_true',
                       help='Include reasoning text in the merged annotations (default: False)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_dir = script_dir / args.output
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
        
    # Find annotators - either directories or standalone JSON files
    annotator_paths = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not annotator_paths:
        annotator_paths = sorted([f for f in input_dir.glob("*.json")])
        
    if not annotator_paths:
        print(f"Error: No annotator subdirectories or JSON master files found in {input_dir}")
        return 1
    
    # Load all annotators
    all_annotations = {}
    for path in annotator_paths:
        ann_name = path.stem
        annotations = load_annotations(str(path))
        if annotations:
            all_annotations[ann_name] = annotations
            if not args.quiet:
                print(f"Loaded {len(annotations)} files from {ann_name}")
    
    if len(all_annotations) < 2:
        print("Error: Need at least 2 valid annotators for majority voting merge")
        return 1
    
    # Perform merge
    if not args.quiet:
        print(f"\nMerging {len(all_annotations)} annotators with {args.threshold*100:.0f}% voting threshold...")
        print("-" * 60)
    
    merged_annotations, overall_stats = merge_all_annotations(
        all_annotations, 
        voting_threshold=args.threshold,
        include_reasoning=args.include_reasoning,
        quiet=args.quiet
    )
    
    if not merged_annotations:
        print("Error: No annotations were merged")
        return 1
    
    # Save results
    if not args.quiet:
        print("\nSaving merged annotations...")
    
    save_merged_annotations(
        merged_annotations, 
        str(output_dir),
        save_individual=not args.no_individual,
        save_combined=not args.no_combined
    )
    
    # Save provenance report
    provenance_path = output_dir / "merge_provenance.json"
    generate_provenance_report(merged_annotations, overall_stats, str(provenance_path))
    
    # Print summary
    if not args.quiet:
        print_merge_report(overall_stats)
    
    print(f"[SUCCESS] Merge complete! Results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
