#!/usr/bin/env python3
"""
Union Merge Tool for Two Annotators

This script merges annotations from two annotators using UNION strategy:
- All relationships from both annotators are included
- Provenance tracking shows which annotator(s) found each relationship
- Grounding sentences are merged (union of both)

This preserves rare but important relationship types (e.g., CONTRAINDICATED)
that may only be identified by one annotator.
"""

import json
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from datetime import datetime


def load_annotations(folder_path: str) -> Dict[str, Any]:
    """Load all annotation JSON files from a folder."""
    annotations = {}
    folder = Path(folder_path)
    
    if not folder.exists():
        return {}
    
    for json_file in sorted(folder.glob("*.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            admission_id = json_file.stem
            annotations[admission_id] = data
    
    return annotations


def get_relationship_key(rel: Dict) -> Tuple:
    """Create a hashable key for a relationship (drug, diagnosis, type)."""
    return (rel.get('drug_row'), rel.get('diagnosis_row'), rel.get('relationship_type'))


def merge_grounding_union(grounding1: Dict, grounding2: Dict, entity_type: str) -> Dict:
    """Merge grounding from two annotators using union."""
    merged = {}
    
    # Get all row keys from both
    all_rows = set()
    if entity_type in grounding1:
        all_rows.update(grounding1[entity_type].keys())
    if entity_type in grounding2:
        all_rows.update(grounding2[entity_type].keys())
    
    for row_key in all_rows:
        sentences = set()
        mention_types = {}
        sources = []
        
        # From annotator 1
        if entity_type in grounding1 and row_key in grounding1[entity_type]:
            row_data = grounding1[entity_type][row_key]
            for i, sent in enumerate(row_data.get('sentences', [])):
                sentences.add(sent)
                if i < len(row_data.get('mention_types', [])):
                    mention_types[sent] = row_data['mention_types'][i]
            sources.append('annotator_1')
        
        # From annotator 2
        if entity_type in grounding2 and row_key in grounding2[entity_type]:
            row_data = grounding2[entity_type][row_key]
            for i, sent in enumerate(row_data.get('sentences', [])):
                sentences.add(sent)
                if sent not in mention_types and i < len(row_data.get('mention_types', [])):
                    mention_types[sent] = row_data['mention_types'][i]
            sources.append('annotator_2')
        
        if sentences:
            sorted_sentences = sorted(sentences)
            merged[row_key] = {
                'sentences': sorted_sentences,
                'mention_types': [mention_types.get(s, 'explicit') for s in sorted_sentences],
                '_sources': sources,
            }
    
    return merged


def merge_relationships_union(ann1_rels: List[Dict], ann2_rels: List[Dict]) -> List[Dict]:
    """Merge relationships using union strategy."""
    
    # Index relationships by key
    ann1_by_key = {get_relationship_key(r): r for r in ann1_rels}
    ann2_by_key = {get_relationship_key(r): r for r in ann2_rels}
    
    all_keys = set(ann1_by_key.keys()) | set(ann2_by_key.keys())
    
    merged = []
    rel_id = 1
    
    for key in sorted(all_keys, key=lambda x: (x[0], x[1], x[2] or '')):
        in_ann1 = key in ann1_by_key
        in_ann2 = key in ann2_by_key
        
        # Determine agreement level
        if in_ann1 and in_ann2:
            agreement = 'both'
            confidence = 'high'
        elif in_ann1:
            agreement = 'annotator_1_only'
            confidence = 'medium'
        else:
            agreement = 'annotator_2_only'
            confidence = 'medium'
        
        # Get relationship data (prefer the one with more data, or ann1 if tied)
        if in_ann1 and in_ann2:
            rel1 = ann1_by_key[key]
            rel2 = ann2_by_key[key]
            # Merge evidence sentences (union)
            evidence_sentences = sorted(set(rel1.get('evidence_sentences', [])) | 
                                         set(rel2.get('evidence_sentences', [])))
            # Use highest confidence
            conf_order = {'high': 3, 'medium': 2, 'low': 1}
            if conf_order.get(rel2.get('confidence', 'medium'), 2) > conf_order.get(rel1.get('confidence', 'medium'), 2):
                base_rel = rel2
            else:
                base_rel = rel1
            # Combine reasoning if different
            reasoning1 = rel1.get('reasoning', '')
            reasoning2 = rel2.get('reasoning', '')
            if reasoning1 and reasoning2 and reasoning1 != reasoning2:
                reasoning = f"{reasoning1} | {reasoning2}"
            else:
                reasoning = reasoning1 or reasoning2
        elif in_ann1:
            base_rel = ann1_by_key[key]
            evidence_sentences = base_rel.get('evidence_sentences', [])
            reasoning = base_rel.get('reasoning', '')
        else:
            base_rel = ann2_by_key[key]
            evidence_sentences = base_rel.get('evidence_sentences', [])
            reasoning = base_rel.get('reasoning', '')
        
        merged_rel = {
            'id': f'rel_{rel_id:03d}',
            'drug_row': key[0],
            'diagnosis_row': key[1],
            'relationship_type': key[2],
            'evidence_sentences': evidence_sentences,
            'evidence_scope': base_rel.get('evidence_scope', 'document'),
            'reasoning': reasoning,
            'confidence': confidence,
            '_agreement': agreement,
        }
        
        # Preserve temporal notes if present
        if base_rel.get('temporal_note'):
            merged_rel['temporal_note'] = base_rel['temporal_note']
        
        merged.append(merged_rel)
        rel_id += 1
    
    return merged


def merge_file_union(file_id: str, ann1_data: Dict, ann2_data: Dict) -> Tuple[Dict, Dict]:
    """Merge annotations for a single file using union."""
    
    # Use metadata from ann1, fallback to ann2
    patient_id = ann1_data.get('patient_id') or ann2_data.get('patient_id', '')
    diagnosis_anchor = ann1_data.get('diagnosis_anchor_id') or ann2_data.get('diagnosis_anchor_id', '')
    medication_anchor = ann1_data.get('medication_anchor_id') or ann2_data.get('medication_anchor_id', '')
    
    # Merge grounding
    grounding1 = ann1_data.get('row_grounding', {})
    grounding2 = ann2_data.get('row_grounding', {})
    
    merged_diagnosis_grounding = merge_grounding_union(grounding1, grounding2, 'diagnosis')
    merged_medication_grounding = merge_grounding_union(grounding1, grounding2, 'medication')
    
    # Merge relationships
    ann1_rels = ann1_data.get('relationships', [])
    ann2_rels = ann2_data.get('relationships', [])
    merged_relationships = merge_relationships_union(ann1_rels, ann2_rels)
    
    # Merge multi-relationship flags (union)
    multi_flags = []
    seen_flags = set()
    for flags_list in [ann1_data.get('multi_relationship_flags', []), 
                        ann2_data.get('multi_relationship_flags', [])]:
        for flag in flags_list:
            key = (flag.get('drug_row'), flag.get('diagnosis_row'))
            if key not in seen_flags:
                seen_flags.add(key)
                multi_flags.append(flag)
    
    # Merge negative relationships (union)
    neg_rels = []
    seen_negs = set()
    for neg_list in [ann1_data.get('negative_relationships', []),
                      ann2_data.get('negative_relationships', [])]:
        for neg in neg_list:
            key = (neg.get('drug_row'), neg.get('diagnosis_row'))
            if key not in seen_negs:
                seen_negs.add(key)
                neg_rels.append(neg)
    
    # Combine quality notes
    notes1 = ann1_data.get('quality_notes', '')
    notes2 = ann2_data.get('quality_notes', '')
    if notes1 and notes2:
        quality_notes = f"[Ann1]: {notes1} | [Ann2]: {notes2}"
    else:
        quality_notes = notes1 or notes2 or ''
    
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
        'multi_relationship_flags': multi_flags,
        'negative_relationships': neg_rels,
        'quality_notes': quality_notes,
        '_merge_metadata': {
            'merge_strategy': 'union',
            'merge_timestamp': datetime.now().isoformat(),
            'annotators': ['annotator_1', 'annotator_2'],
        }
    }
    
    # Compute stats
    both_count = sum(1 for r in merged_relationships if r['_agreement'] == 'both')
    ann1_only = sum(1 for r in merged_relationships if r['_agreement'] == 'annotator_1_only')
    ann2_only = sum(1 for r in merged_relationships if r['_agreement'] == 'annotator_2_only')
    
    stats = {
        'file_id': file_id,
        'total_relationships': len(merged_relationships),
        'both_agreed': both_count,
        'annotator_1_only': ann1_only,
        'annotator_2_only': ann2_only,
        'agreement_rate': both_count / len(merged_relationships) * 100 if merged_relationships else 100,
    }
    
    return merged, stats


def print_merge_summary(all_stats: List[Dict], merged_annotations: Dict):
    """Print summary of the merge."""
    
    print("\n" + "="*80)
    print("  UNION MERGE SUMMARY (Annotators 1 & 2)")
    print("="*80 + "\n")
    
    total_rels = sum(s['total_relationships'] for s in all_stats)
    total_both = sum(s['both_agreed'] for s in all_stats)
    total_ann1 = sum(s['annotator_1_only'] for s in all_stats)
    total_ann2 = sum(s['annotator_2_only'] for s in all_stats)
    
    print("[OVERALL STATISTICS]")
    print("-" * 40)
    print(f"  Files Merged:              {len(all_stats)}")
    print(f"  Total Relationships:       {total_rels}")
    print()
    print(f"  Both Agreed (high conf):   {total_both} ({total_both/total_rels*100:.1f}%)")
    print(f"  Annotator 1 Only:          {total_ann1} ({total_ann1/total_rels*100:.1f}%)")
    print(f"  Annotator 2 Only:          {total_ann2} ({total_ann2/total_rels*100:.1f}%)")
    print()
    
    # Count relationship types
    type_counts = defaultdict(lambda: {'both': 0, 'ann1': 0, 'ann2': 0})
    for ann in merged_annotations.values():
        for rel in ann.get('relationships', []):
            rel_type = rel.get('relationship_type', 'UNKNOWN')
            agreement = rel.get('_agreement', 'both')
            if agreement == 'both':
                type_counts[rel_type]['both'] += 1
            elif agreement == 'annotator_1_only':
                type_counts[rel_type]['ann1'] += 1
            else:
                type_counts[rel_type]['ann2'] += 1
    
    print("[RELATIONSHIP TYPES]")
    print("-" * 40)
    print(f"  {'Type':<20} {'Both':>8} {'Ann1':>8} {'Ann2':>8} {'Total':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for rel_type in sorted(type_counts.keys()):
        counts = type_counts[rel_type]
        total = counts['both'] + counts['ann1'] + counts['ann2']
        print(f"  {rel_type:<20} {counts['both']:>8} {counts['ann1']:>8} {counts['ann2']:>8} {total:>8}")
    
    print()
    print("[PER-FILE BREAKDOWN]")
    print("-" * 40)
    print(f"  {'File ID':<15} {'Total':>8} {'Both':>8} {'Ann1':>8} {'Ann2':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for stats in all_stats:
        print(f"  {stats['file_id']:<15} {stats['total_relationships']:>8} "
              f"{stats['both_agreed']:>8} {stats['annotator_1_only']:>8} "
              f"{stats['annotator_2_only']:>8}")
    
    print()
    print("="*80)
    print("  END OF REPORT")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Merge two annotators using UNION strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script merges annotations from Annotators 1 & 2 using UNION:
- All relationships from both annotators are included
- Provenance tracking shows agreement level
- Rare relationship types (e.g., CONTRAINDICATED) are preserved
        """
    )
    
    parser.add_argument('--annotator1', '-a1', default='Annotations/Individual/annotator_1',
                       help='Path to annotator 1 folder (default: Annotations/Individual/annotator_1)')
    parser.add_argument('--annotator2', '-a2', default='Annotations/Individual/annotator_2',
                       help='Path to annotator 2 folder (default: Annotations/Individual/annotator_2)')
    parser.add_argument('--output', '-o', default='Annotations/Union',
                       help='Output directory (default: Annotations/Union)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    ann1_path = script_dir / args.annotator1
    ann2_path = script_dir / args.annotator2
    output_dir = script_dir / args.output
    
    # Load annotations
    print(f"Loading annotations from {args.annotator1}...")
    ann1 = load_annotations(str(ann1_path))
    print(f"  Loaded {len(ann1)} files")
    
    print(f"Loading annotations from {args.annotator2}...")
    ann2 = load_annotations(str(ann2_path))
    print(f"  Loaded {len(ann2)} files")
    
    # Get all file IDs (union)
    all_files = set(ann1.keys()) | set(ann2.keys())
    
    if not all_files:
        print("Error: No files found to merge")
        return 1
    
    # Merge each file
    merged_annotations = {}
    all_stats = []
    
    print(f"\nMerging {len(all_files)} files using UNION strategy...")
    
    for file_id in sorted(all_files):
        ann1_data = ann1.get(file_id, {})
        ann2_data = ann2.get(file_id, {})
        
        merged, stats = merge_file_union(file_id, ann1_data, ann2_data)
        merged_annotations[file_id] = merged
        all_stats.append(stats)
        
        if not args.quiet:
            print(f"  {file_id}: {stats['total_relationships']} rels "
                  f"(both={stats['both_agreed']}, ann1={stats['annotator_1_only']}, "
                  f"ann2={stats['annotator_2_only']})")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Save individual files
    for file_id, annotation in merged_annotations.items():
        file_path = output_dir / f"{file_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2)
    
    print(f"\nSaved {len(merged_annotations)} merged annotation files to: {output_dir}")
    
    # Save combined file
    combined_path = output_dir / "merged_annotations_all.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(merged_annotations, f, indent=2)
    print(f"Saved combined file to: {combined_path}")
    
    # Save provenance report
    provenance = {
        'merge_strategy': 'union',
        'timestamp': datetime.now().isoformat(),
        'annotators': ['annotator_1', 'annotator_2'],
        'summary': {
            'total_files': len(all_stats),
            'total_relationships': sum(s['total_relationships'] for s in all_stats),
            'both_agreed': sum(s['both_agreed'] for s in all_stats),
            'annotator_1_only': sum(s['annotator_1_only'] for s in all_stats),
            'annotator_2_only': sum(s['annotator_2_only'] for s in all_stats),
        },
        'per_file': all_stats,
    }
    
    provenance_path = output_dir / "merge_provenance.json"
    with open(provenance_path, 'w', encoding='utf-8') as f:
        json.dump(provenance, f, indent=2)
    print(f"Saved provenance report to: {provenance_path}")
    
    # Print summary
    if not args.quiet:
        print_merge_summary(all_stats, merged_annotations)
    
    print("[SUCCESS] Union merge complete!")
    
    return 0


if __name__ == '__main__':
    exit(main())
