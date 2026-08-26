#!/usr/bin/env python3
"""
Multi-Annotator Agreement Analysis Tool

This script analyzes annotation quality and inter-annotator agreement across
3 or more annotators for clinical relationship annotations.

Metrics computed:
1. Fleiss' Kappa (multi-rater agreement)
2. Pairwise Cohen's Kappa
3. Agreement matrices
4. Per-annotator statistics
5. Voting distribution analysis

Visualizations:
- Agreement heatmap (pairwise)
- Voting distribution charts
- Per-file agreement breakdown
- Annotator correlation matrix
"""

import json
import os
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Import numpy (required for core functionality)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Error: numpy is required for this script. Please install it with: pip install numpy")
    exit(1)

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


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
                        annotations[adm_id] = entry
                elif isinstance(data, dict):
                    adm_id = data.get('admission_id', json_file.stem)
                    if adm_id.startswith('annotation_'): adm_id = adm_id[11:]
                    annotations[adm_id] = data
    
    return annotations


def get_relationship_key(rel: Dict) -> Tuple:
    """Create a hashable key for a relationship."""
    return (rel.get('drug_row'), rel.get('diagnosis_row'), rel.get('relationship_type'))


def get_relationship_pair_key(rel: Dict) -> Tuple:
    """Create a key for drug-diagnosis pair (ignoring relationship type)."""
    return (rel.get('drug_row'), rel.get('diagnosis_row'))


def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Compute Fleiss' Kappa for multi-rater agreement.
    
    Args:
        ratings_matrix: N x k matrix where N = number of items, k = number of categories
                       Each cell contains the number of raters who assigned that category
    
    Returns:
        Fleiss' Kappa value
    """
    N, k = ratings_matrix.shape
    n = ratings_matrix.sum(axis=1)[0]  # Number of raters (assumed constant)
    
    if n <= 1:
        return 1.0
    
    # Proportion of assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (N * n)
    
    # Extent of agreement for each item
    P_i = (ratings_matrix.sum(axis=1)**2 - n) / (n * (n - 1)) if n > 1 else np.ones(N)
    # Correct calculation
    P_i = np.zeros(N)
    for i in range(N):
        P_i[i] = (np.sum(ratings_matrix[i]**2) - n) / (n * (n - 1)) if n > 1 else 1.0
    
    # Mean extent of agreement
    P_bar = np.mean(P_i)
    
    # Expected agreement by chance
    P_e_bar = np.sum(p_j**2)
    
    if P_e_bar == 1:
        return 1.0
    
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return kappa


def compute_cohens_kappa_binary(set1: Set, set2: Set, all_items: Set) -> float:
    """
    Compute Cohen's Kappa for binary agreement (item present or not).
    """
    if not all_items:
        return 1.0
    
    # Contingency table
    both_yes = len(set1 & set2)
    set1_only = len(set1 - set2)
    set2_only = len(set2 - set1)
    both_no = len(all_items - set1 - set2)
    
    total = len(all_items)
    if total == 0:
        return 1.0
    
    # Observed agreement
    po = (both_yes + both_no) / total
    
    # Expected agreement
    p1_yes = (both_yes + set1_only) / total
    p2_yes = (both_yes + set2_only) / total
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)
    
    if pe == 1:
        return 1.0
    
    kappa = (po - pe) / (1 - pe)
    return kappa


def analyze_voting_distribution(all_annotations: Dict[str, Dict[str, Any]], 
                                file_ids: List[str]) -> Dict[str, Any]:
    """
    Analyze how relationships are distributed across annotators.
    """
    annotator_names = list(all_annotations.keys())
    n_annotators = len(annotator_names)
    
    # Collect all unique relationships across all annotators
    all_relationships = defaultdict(lambda: defaultdict(set))  # file_id -> rel_key -> set of annotators
    
    for ann_name, annotations in all_annotations.items():
        for file_id in file_ids:
            if file_id not in annotations:
                continue
            for rel in annotations[file_id].get('relationships', []):
                rel_key = get_relationship_key(rel)
                all_relationships[file_id][rel_key].add(ann_name)
    
    # Count voting distribution
    vote_counts = Counter()  # Number of annotators agreeing
    relationship_votes = []  # List of (file_id, rel_key, vote_count, annotators)
    
    for file_id, rels in all_relationships.items():
        for rel_key, annotators in rels.items():
            vote_count = len(annotators)
            vote_counts[vote_count] += 1
            relationship_votes.append({
                'file_id': file_id,
                'relationship': rel_key,
                'vote_count': vote_count,
                'annotators': list(annotators),
                'unanimous': vote_count == n_annotators,
                'majority': vote_count >= (n_annotators / 2 + 0.5),
            })
    
    # Calculate statistics
    total_unique_rels = sum(vote_counts.values())
    
    return {
        'n_annotators': n_annotators,
        'annotator_names': annotator_names,
        'vote_distribution': dict(vote_counts),
        'total_unique_relationships': total_unique_rels,
        'unanimous_count': vote_counts.get(n_annotators, 0),
        'unanimous_pct': vote_counts.get(n_annotators, 0) / total_unique_rels * 100 if total_unique_rels > 0 else 0,
        'majority_count': sum(v for k, v in vote_counts.items() if k >= (n_annotators / 2 + 0.5)),
        'majority_pct': sum(v for k, v in vote_counts.items() if k >= (n_annotators / 2 + 0.5)) / total_unique_rels * 100 if total_unique_rels > 0 else 0,
        'relationship_votes': relationship_votes,
    }


def compute_pairwise_agreement(all_annotations: Dict[str, Dict[str, Any]], 
                               file_ids: List[str]) -> Dict[str, Any]:
    """
    Compute pairwise agreement metrics between all annotator pairs.
    """
    annotator_names = list(all_annotations.keys())
    n_annotators = len(annotator_names)
    
    pairwise_results = {}
    
    for i, ann1_name in enumerate(annotator_names):
        for j, ann2_name in enumerate(annotator_names):
            if i >= j:
                continue
            
            ann1 = all_annotations[ann1_name]
            ann2 = all_annotations[ann2_name]
            
            # Collect relationships
            rels1 = set()
            rels2 = set()
            all_rels = set()
            
            for file_id in file_ids:
                if file_id in ann1:
                    for rel in ann1[file_id].get('relationships', []):
                        rel_key = (file_id,) + get_relationship_key(rel)
                        rels1.add(rel_key)
                        all_rels.add(rel_key)
                
                if file_id in ann2:
                    for rel in ann2[file_id].get('relationships', []):
                        rel_key = (file_id,) + get_relationship_key(rel)
                        rels2.add(rel_key)
                        all_rels.add(rel_key)
            
            # Compute metrics
            intersection = len(rels1 & rels2)
            union = len(rels1 | rels2)
            
            agreement_pct = intersection / union * 100 if union > 0 else 100
            jaccard = intersection / union if union > 0 else 1.0
            kappa = compute_cohens_kappa_binary(rels1, rels2, all_rels)
            
            pair_key = f"{ann1_name}_vs_{ann2_name}"
            pairwise_results[pair_key] = {
                'annotator_1': ann1_name,
                'annotator_2': ann2_name,
                'ann1_count': len(rels1),
                'ann2_count': len(rels2),
                'intersection': intersection,
                'union': union,
                'agreement_pct': agreement_pct,
                'jaccard': jaccard,
                'cohens_kappa': kappa,
            }
    
    return pairwise_results


def compute_fleiss_kappa_for_relationships(all_annotations: Dict[str, Dict[str, Any]], 
                                            file_ids: List[str]) -> float:
    """
    Compute Fleiss' Kappa for relationship annotations.
    """
    annotator_names = list(all_annotations.keys())
    n_annotators = len(annotator_names)
    
    if n_annotators < 2:
        return 1.0
    
    # Collect all unique relationships
    all_relationships = set()
    for ann_name, annotations in all_annotations.items():
        for file_id in file_ids:
            if file_id not in annotations:
                continue
            for rel in annotations[file_id].get('relationships', []):
                rel_key = (file_id,) + get_relationship_key(rel)
                all_relationships.add(rel_key)
    
    if not all_relationships:
        return 1.0
    
    # Build ratings matrix (N items x 2 categories: present/absent)
    rel_list = sorted(all_relationships)
    N = len(rel_list)
    ratings_matrix = np.zeros((N, 2))  # [present, absent]
    
    for i, rel_key in enumerate(rel_list):
        present_count = 0
        for ann_name, annotations in all_annotations.items():
            file_id = rel_key[0]
            if file_id in annotations:
                ann_rels = {(file_id,) + get_relationship_key(r) 
                           for r in annotations[file_id].get('relationships', [])}
                if rel_key in ann_rels:
                    present_count += 1
        
        ratings_matrix[i, 0] = present_count
        ratings_matrix[i, 1] = n_annotators - present_count
    
    return compute_fleiss_kappa(ratings_matrix)


def compute_per_annotator_stats(all_annotations: Dict[str, Dict[str, Any]], 
                                 file_ids: List[str]) -> Dict[str, Any]:
    """
    Compute statistics for each individual annotator.
    """
    stats = {}
    
    for ann_name, annotations in all_annotations.items():
        total_rels = 0
        rel_types = Counter()
        confidence_levels = Counter()
        evidence_scopes = Counter()
        files_annotated = 0
        
        for file_id in file_ids:
            if file_id not in annotations:
                continue
            files_annotated += 1
            
            for rel in annotations[file_id].get('relationships', []):
                total_rels += 1
                rel_types[rel.get('relationship_type', 'UNKNOWN')] += 1
                confidence_levels[rel.get('confidence', 'unspecified')] += 1
                evidence_scopes[rel.get('evidence_scope', 'unspecified')] += 1
        
        stats[ann_name] = {
            'total_relationships': total_rels,
            'files_annotated': files_annotated,
            'avg_relationships_per_file': total_rels / files_annotated if files_annotated > 0 else 0,
            'relationship_types': dict(rel_types),
            'confidence_levels': dict(confidence_levels),
            'evidence_scopes': dict(evidence_scopes),
        }
    
    return stats


def compute_mention_type_stats(all_annotations: Dict[str, Dict[str, Any]], 
                               file_ids: List[str]) -> Dict[str, Any]:
    """
    Compute mention type statistics per annotator and per table.
    
    Returns:
        Dictionary containing:
        - per_annotator_mention_types: Mention type counts per annotator for drugs and diagnoses
        - mention_type_per_table: Count of each mention type across all tables
        - mention_type_per_table_by_annotator: Count per table type per annotator
    """
    stats = {
        'per_annotator_mention_types': {},
        'mention_type_per_table': {
            'medication': Counter(),
            'diagnosis': Counter(),
        },
        'mention_type_per_table_by_annotator': {},
    }
    
    for ann_name, annotations in all_annotations.items():
        medication_mention_types = Counter()
        diagnosis_mention_types = Counter()
        
        for file_id in file_ids:
            if file_id not in annotations:
                continue
            
            data = annotations[file_id]
            
            # Count mention types from medication (row_grounding)
            if 'row_grounding' in data and 'medication' in data['row_grounding']:
                for row_id, row_data in data['row_grounding']['medication'].items():
                    if 'mention_types' in row_data:
                        for mention_type in row_data['mention_types']:
                            medication_mention_types[mention_type] += 1
                            stats['mention_type_per_table']['medication'][mention_type] += 1
            
            # Count mention types from diagnosis (row_grounding)
            if 'row_grounding' in data and 'diagnosis' in data['row_grounding']:
                for row_id, row_data in data['row_grounding']['diagnosis'].items():
                    if 'mention_types' in row_data:
                        for mention_type in row_data['mention_types']:
                            diagnosis_mention_types[mention_type] += 1
                            stats['mention_type_per_table']['diagnosis'][mention_type] += 1
        
        stats['per_annotator_mention_types'][ann_name] = {
            'medication': dict(medication_mention_types),
            'diagnosis': dict(diagnosis_mention_types),
            'total': dict((medication_mention_types + diagnosis_mention_types)),
        }
        
        stats['mention_type_per_table_by_annotator'][ann_name] = {
            'medication': dict(medication_mention_types),
            'diagnosis': dict(diagnosis_mention_types),
        }
    
    # Convert Counters to dicts for JSON serialization
    stats['mention_type_per_table']['medication'] = dict(stats['mention_type_per_table']['medication'])
    stats['mention_type_per_table']['diagnosis'] = dict(stats['mention_type_per_table']['diagnosis'])
    
    return stats


def compute_mention_type_agreement(all_annotations: Dict[str, Dict[str, Any]], 
                                     file_ids: List[str]) -> Dict[str, Any]:
    """
    Compute pairwise agreement on row grounding stratified by mention type.
    
    This shows how much annotators agree when using specific mention types
    (e.g., explicit vs synonym vs abbreviated) for diagnosis and medication grounding.
    
    Returns:
        Dictionary containing:
        - diagnosis_mention_agreement: Agreement rates by mention type for diagnoses
        - medication_mention_agreement: Agreement rates by mention type for medications
        - pairwise_by_type: Detailed pairwise comparisons by type
    """
    annotator_names = list(all_annotations.keys())
    n_annotators = len(annotator_names)
    
    # Collect all mention instances with their types
    # Structure: {file_id: {table_type: {row_id: {annotator: set of (sentence_idx, mention_type)}}}}
    mention_instances = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    
    for ann_name, annotations in all_annotations.items():
        for file_id in file_ids:
            if file_id not in annotations:
                continue
            
            data = annotations[file_id]
            
            # Process medication mentions
            if 'row_grounding' in data and 'medication' in data['row_grounding']:
                for row_id, row_data in data['row_grounding']['medication'].items():
                    sentences = row_data.get('sentences', [])
                    mention_types = row_data.get('mention_types', [])
                    for sent_idx, mention_type in zip(sentences, mention_types):
                        mention_instances[file_id]['medication'][row_id][ann_name].add((sent_idx, mention_type))
            
            # Process diagnosis mentions
            if 'row_grounding' in data and 'diagnosis' in data['row_grounding']:
                for row_id, row_data in data['row_grounding']['diagnosis'].items():
                    sentences = row_data.get('sentences', [])
                    mention_types = row_data.get('mention_types', [])
                    for sent_idx, mention_type in zip(sentences, mention_types):
                        mention_instances[file_id]['diagnosis'][row_id][ann_name].add((sent_idx, mention_type))
    
    # Compute agreement by mention type
    results = {
        'diagnosis_mention_agreement': defaultdict(lambda: {'total': 0, 'agreed': 0, 'agreement_rate': 0.0}),
        'medication_mention_agreement': defaultdict(lambda: {'total': 0, 'agreed': 0, 'agreement_rate': 0.0}),
        'pairwise_by_type': {},
    }
    
    # For each table type
    for table_type in ['diagnosis', 'medication']:
        mention_type_counts = defaultdict(lambda: {
            'total': 0, 'agreed': 0, 'unanimous': 0, 'majority': 0, 
            'type_disagreement': 0,  # Same row, different type
            'no_grounding_overlap': 0  # Different rows entirely
        })
        
        # Track all rows each annotator used for each mention type (for no_grounding_overlap count)
        all_annotator_rows = defaultdict(lambda: defaultdict(set))  # mention_type -> annotator -> set of (file_id, row_id)
        
        for ann_name, annotations in all_annotations.items():
            for file_id in file_ids:
                if file_id not in annotations:
                    continue
                data = annotations[file_id]
                
                if 'row_grounding' in data and table_type in data['row_grounding']:
                    for row_id, row_data in data['row_grounding'][table_type].items():
                        mention_types = row_data.get('mention_types', [])
                        for mention_type in mention_types:
                            all_annotator_rows[mention_type][ann_name].add((file_id, row_id))
        
        # Iterate through all mention instances (rows with overlap)
        for file_id, tables in mention_instances.items():
            if table_type not in tables:
                continue
            
            for row_id, annotator_mentions in tables[table_type].items():
                if len(annotator_mentions) < 2:
                    continue  # Need at least 2 annotators
                
                # Count how many annotators used each mention type
                type_usage = defaultdict(set)  # mention_type -> set of annotators who used it
                for ann_name, mentions in annotator_mentions.items():
                    for _, m_type in mentions:
                        type_usage[m_type].add(ann_name)
                
                # For each mention type, check agreement level
                for m_type, annotators_using_type in type_usage.items():
                    n_using = len(annotators_using_type)
                    n_total = len(annotator_mentions)  # Total annotators who grounded to this row
                    
                    mention_type_counts[m_type]['total'] += 1
                    
                    if n_using >= 2:
                        # At least majority agreement on using this type
                        mention_type_counts[m_type]['agreed'] += 1
                        
                        if n_using == n_annotators:
                            # Unanimous - all annotators used this type
                            mention_type_counts[m_type]['unanimous'] += 1
                        else:
                            # Majority only - not all annotators
                            mention_type_counts[m_type]['majority'] += 1
                    else:
                        # Type disagreement - grounded to same row but different type
                        mention_type_counts[m_type]['type_disagreement'] += 1
        
        # Now compute no_grounding_overlap: rows where annotators didn't overlap
        for m_type, annotator_rows in all_annotator_rows.items():
            # Get all unique (file_id, row_id) pairs across all annotators
            all_rows = set()
            for rows in annotator_rows.values():
                all_rows.update(rows)
            
            # For each unique row, check if it appears in mention_instances (has overlap)
            for file_row in all_rows:
                file_id, row_id = file_row
                # Check if this row has overlap (appears in mention_instances)
                has_overlap = (file_id in mention_instances and 
                              table_type in mention_instances[file_id] and 
                              row_id in mention_instances[file_id][table_type] and
                              len(mention_instances[file_id][table_type][row_id]) >= 2)
                
                if not has_overlap:
                    # This row was used by only 1 annotator for this mention type
                    mention_type_counts[m_type]['no_grounding_overlap'] += 1
        
        # Compute agreement rates
        key = f'{table_type}_mention_agreement'
        for m_type, counts in mention_type_counts.items():
            results[key][m_type] = {
                'total': counts['total'],
                'agreed': counts['agreed'],
                'unanimous': counts['unanimous'],
                'majority_only': counts['majority'],
                'type_disagreement': counts['type_disagreement'],
                'no_grounding_overlap': counts['no_grounding_overlap'],
                'agreement_rate': counts['agreed'] / counts['total'] * 100 if counts['total'] > 0 else 0.0
            }
        
        # Convert defaultdict to dict
        results[key] = dict(results[key])
    
    return results


def compute_evidence_scope_agreement(all_annotations: Dict[str, Dict[str, Any]], 
                                       file_ids: List[str]) -> Dict[str, Any]:
    """
    Compute agreement on relationship evidence scope classification.
    
    This shows how much a annotators agree on classifying relationships
    as having 'explicit', 'section', 'document', or 'context' level evidence.
    
    Returns:
        Dictionary containing:
        - scope_agreement_rates: Agreement rate for each scope type
        - scope_distribution: How often each scope is used in unanimous vs non-unanimous cases
        - pairwise_scope_agreement: Pairwise agreement on scope classification
    """
    annotator_names = list(all_annotations.keys())
    n_annotators = len(annotator_names)
    
    # Collect relationships with their scopes
    # Structure: {(file_id, drug_row, diag_row, rel_type): {annotator: scope}}
    relationship_scopes = defaultdict(dict)
    
    for ann_name, annotations in all_annotations.items():
        for file_id in file_ids:
            if file_id not in annotations:
                continue
            
            for rel in annotations[file_id].get('relationships', []):
                rel_key = (file_id, rel.get('drug_row'), rel.get('diagnosis_row'), rel.get('relationship_type'))
                scope = rel.get('evidence_scope', 'unspecified')
                relationship_scopes[rel_key][ann_name] = scope
    
    # Compute scope agreement statistics - count each RELATIONSHIP once
    scope_counts = defaultdict(lambda: {
        'total': 0, 'unanimous': 0, 'majority': 0, 
        'scope_disagreement': 0,      # Relationship agreed, scope disagreed
        'no_existence_overlap': 0     # Relationship not agreed (only 1 found it)
    })
    pairwise_scope_agreement = defaultdict(lambda: {'total': 0, 'agreed': 0})
    
    for rel_key, ann_scopes in relationship_scopes.items():
        # Get all scopes for this relationship
        scopes = list(ann_scopes.values())
        scope_counter = Counter(scopes)
        most_common_scope = scope_counter.most_common(1)[0][0]
        most_common_count = scope_counter.most_common(1)[0][1]
        n_annotators = len(ann_scopes)
        
        scope_counts[most_common_scope]['total'] += 1
        
        if n_annotators < 2:
            # Only 1 annotator found this relationship - No Existence Overlap
            scope_counts[most_common_scope]['no_existence_overlap'] += 1
        else:
            # Relationship agreed (2+ annotators). Check scope agreement.
            if most_common_count == n_annotators:
                # Unanimous on scope
                scope_counts[most_common_scope]['unanimous'] += 1
            elif most_common_count >= n_annotators / 2 + 0.5:
                # Majority on scope
                scope_counts[most_common_scope]['majority'] += 1
            else:
                # No majority on scope - Scope Disagreement
                scope_counts[most_common_scope]['scope_disagreement'] += 1
        
        # Pairwise agreement (only if >1 annotator)
        if n_annotators >= 2:
            ann_list = list(ann_scopes.keys())
            for i in range(len(ann_list)):
                for j in range(i + 1, len(ann_list)):
                    scope1 = ann_scopes[ann_list[i]]
                    scope2 = ann_scopes[ann_list[j]]
                    
                    pair_key = f"{ann_list[i]}_vs_{ann_list[j]}"
                    pairwise_scope_agreement[pair_key]['total'] += 1
                    if scope1 == scope2:
                        pairwise_scope_agreement[pair_key]['agreed'] += 1
    
    # Compute agreement rates
    results = {
        'scope_agreement_rates': {},
        'scope_distribution': {},
        'pairwise_scope_agreement': {},
    }
    
    for scope, counts in scope_counts.items():
        results['scope_agreement_rates'][scope] = {
            'total_instances': counts['total'],
            'unanimous_count': counts['unanimous'],
            'unanimous_rate': counts['unanimous'] / counts['total'] * 100 if counts['total'] > 0 else 0.0,
            'majority_count': counts['majority'],
            'majority_rate': counts['majority'] / counts['total'] * 100 if counts['total'] > 0 else 0.0,
            'scope_disagreement': counts['scope_disagreement'],
            'no_existence_overlap': counts['no_existence_overlap'],
        }
    
    for pair_key, counts in pairwise_scope_agreement.items():
        results['pairwise_scope_agreement'][pair_key] = {
            'total_comparisons': counts['total'],
            'agreed': counts['agreed'],
            'agreement_rate': counts['agreed'] / counts['total'] * 100 if counts['total'] > 0 else 0.0,
        }
    
    return results


def compute_relationship_type_by_voting(all_annotations: Dict[str, Dict[str, Any]], 
                                         file_ids: List[str]) -> Dict[str, Any]:
    """
    Compute relationship type distribution broken down by voting level.
    
    This shows: Of the unanimous/majority relationships, how many are TREATS vs ADVERSE_EFFECT etc?
    This makes the totals match with voting distribution.
    
    Returns:
        Dictionary containing relationship type counts for unanimous, majority, and minority
    """
    annotator_names = list(all_annotations.keys())
    n_annotators = len(annotator_names)
    
    # Collect relationships
    # Structure: {(file_id, drug_row, diag_row, rel_type): set of annotators who identified it}
    relationship_votes = defaultdict(set)
    
    for ann_name, annotations in all_annotations.items():
        for file_id in file_ids:
            if file_id not in annotations:
                continue
            
            for rel in annotations[file_id].get('relationships', []):
                rel_key = (file_id, rel.get('drug_row'), rel.get('diagnosis_row'), rel.get('relationship_type'))
                relationship_votes[rel_key].add(ann_name)
    
    # Count relationships by voting level and type
    type_by_voting = {
        'unanimous': defaultdict(int),  # 3/3 annotators agreed relationship exists
        'majority': defaultdict(int),    # 2/3 annotators agreed relationship exists
        'minority': defaultdict(int),    # 1/3 only one annotator
    }
    
    for rel_key, annotators in relationship_votes.items():
        n_annotators_for_rel = len(annotators)
        rel_type = rel_key[3]  # Extract relationship type from key
        
        # Determine voting level
        if n_annotators_for_rel == n_annotators:
            voting_level = 'unanimous'
        elif n_annotators_for_rel >= n_annotators / 2 + 0.5:
            voting_level = 'majority'
        else:
            voting_level = 'minority'
        
        # Count this relationship under its voting level and type
        type_by_voting[voting_level][rel_type] += 1
    
    # Convert to regular dicts
    results = {
        'unanimous': dict(type_by_voting['unanimous']),
        'majority': dict(type_by_voting['majority']),
        'minority': dict(type_by_voting['minority']),
    }
    
    return results


def get_friendly_annotator_name(annotator_path: str) -> str:
    """
    Convert annotator folder path to friendly display name.
    
    Examples:
        annotator_1 or Annotations/Individual/annotator_1 -> Gemini
        annotator_2 or Annotations/Individual/annotator_2 -> ChatGPT
        annotator_3 or Annotations/Individual/annotator_3 -> Qwen
    """
    # Extract the annotator name from the path
    path_parts = annotator_path.replace('\\', '/').split('/')
    annotator_name = path_parts[-1]
    
    # Map to friendly names
    name_mapping = {
        'annotator_1': 'Gemini',
        'annotator_2': 'ChatGPT',
        'annotator_3': 'Qwen',
    }
    
    return name_mapping.get(annotator_name, annotator_name)


def generate_multi_annotator_report(all_annotations: Dict[str, Dict[str, Any]], 
                                    output_dir: str) -> Dict:
    """Generate comprehensive multi-annotator analysis report."""
    
    annotator_names = list(all_annotations.keys())
    # Create display names for annotators
    annotator_display_names = {name: get_friendly_annotator_name(name) for name in annotator_names}
    n_annotators = len(annotator_names)
    
    # Get common file IDs
    all_file_sets = [set(ann.keys()) for ann in all_annotations.values()]
    common_files = set.intersection(*all_file_sets) if all_file_sets else set()
    all_files = set.union(*all_file_sets) if all_file_sets else set()
    
    results = {
        'summary': {
            'n_annotators': n_annotators,
            'annotator_names': annotator_names,
            'annotator_display_names': annotator_display_names,
            'total_files': len(all_files),
            'common_files': len(common_files),
        },
        'fleiss_kappa': compute_fleiss_kappa_for_relationships(all_annotations, list(all_files)),
        'pairwise_agreement': compute_pairwise_agreement(all_annotations, list(all_files)),
        'voting_distribution': analyze_voting_distribution(all_annotations, list(all_files)),
        'per_annotator_stats': compute_per_annotator_stats(all_annotations, list(all_files)),
        'mention_type_stats': compute_mention_type_stats(all_annotations, list(all_files)),
        'mention_type_agreement': compute_mention_type_agreement(all_annotations, list(all_files)),
        'evidence_scope_agreement': compute_evidence_scope_agreement(all_annotations, list(all_files)),
        'relationship_type_by_voting': compute_relationship_type_by_voting(all_annotations, list(all_files)),
    }
    
    return results


def create_multi_annotator_visualizations(results: Dict, output_dir: str):
    """Create visualization charts for multi-annotator analysis."""
    
    if not HAS_MATPLOTLIB:
        print("Skipping visualizations - matplotlib not available")
        return
    
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    
    # Import patches for rounded bars
    from matplotlib.patches import FancyBboxPatch, Patch, PathPatch
    from matplotlib.path import Path
    
    # Helper function to draw bars with rounded tops only (pill-like top)
    def draw_rounded_bars(ax, x_positions, heights, width=0.2, color='#3498db', 
                          alpha=0.85, label=None, bottom=None):
        """
        Draw bars with rounded TOP corners only (flat bottom, rounded top like pills).
        
        Args:
            ax: matplotlib axis
            x_positions: x coordinates for bars (can be array or single value)
            heights: bar heights (can be array or single value)
            width: bar width
            color: bar color
            alpha: transparency
            label: legend label
            bottom: bottom offset for stacked bars
        
        Returns:
            list of patch objects
        """
        if not hasattr(x_positions, '__iter__'):
            x_positions = [x_positions]
        if not hasattr(heights, '__iter__'):
            heights = [heights]
        if bottom is None:
            bottom = [0] * len(heights)
        elif not hasattr(bottom, '__iter__'):
            bottom = [bottom] * len(heights)
        
        patches = []
        for xi, hi, bi in zip(x_positions, heights, bottom):
            if hi > 0:
                # Radius for rounded top corners only (small radius for corner rounding)
                radius = min(width / 4, hi * 0.5, 0.8)
                
                # Define the bar path with rounded top only
                left = xi - width / 2
                right = xi + width / 2
                bot = bi
                top = bi + hi
                
                # Create path: start bottom-left, go up, curve top-left, 
                # straight across top, curve top-right, go down, close
                verts = [
                    (left, bot),                          # Bottom-left
                    (left, top - radius),                 # Left side up to curve start
                    (left, top),                          # Control point for curve
                    (left + radius, top),                 # Top-left after curve
                    (right - radius, top),                # Top-right before curve
                    (right, top),                         # Control point for curve
                    (right, top - radius),                # Right side after curve
                    (right, bot),                         # Bottom-right
                    (left, bot),                          # Close path
                ]
                
                codes = [
                    Path.MOVETO,
                    Path.LINETO,
                    Path.CURVE3,
                    Path.CURVE3,
                    Path.LINETO,
                    Path.CURVE3,
                    Path.CURVE3,
                    Path.LINETO,
                    Path.CLOSEPOLY,
                ]
                
                path = Path(verts, codes)
                patch = PathPatch(path, facecolor=color, alpha=alpha,
                                 edgecolor='white', linewidth=0.5)
                ax.add_patch(patch)
                patches.append(patch)
            else:
                patches.append(None)
        
        # Update axis limits
        all_x = list(x_positions)
        all_tops = [h + b for h, b in zip(heights, bottom)]
        ax.set_xlim(min(all_x) - width, max(all_x) + width)
        current_ylim = ax.get_ylim()
        ax.set_ylim(0, max(current_ylim[1], max(all_tops) * 1.15))
        
        return patches
    
    fig_count = 0
    annotator_names = results['summary']['annotator_names']
    n_annotators = results['summary']['n_annotators']
    
    # ===== 1. Pairwise Agreement Heatmap =====
    if n_annotators >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Build agreement matrix
        agreement_matrix = np.zeros((n_annotators, n_annotators))
        kappa_matrix = np.zeros((n_annotators, n_annotators))
        
        for i, ann1 in enumerate(annotator_names):
            for j, ann2 in enumerate(annotator_names):
                if i == j:
                    agreement_matrix[i, j] = 100
                    kappa_matrix[i, j] = 1.0
                elif i < j:
                    pair_key = f"{ann1}_vs_{ann2}"
                    if pair_key in results['pairwise_agreement']:
                        agreement_matrix[i, j] = results['pairwise_agreement'][pair_key]['agreement_pct']
                        agreement_matrix[j, i] = agreement_matrix[i, j]
                        kappa_matrix[i, j] = results['pairwise_agreement'][pair_key]['cohens_kappa']
                        kappa_matrix[j, i] = kappa_matrix[i, j]
        
        # Create display labels
        display_labels = [results['summary']['annotator_display_names'][name] for name in annotator_names]
        
        if HAS_SEABORN:
            sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                       xticklabels=display_labels, yticklabels=display_labels, ax=ax,
                       vmin=0, vmax=100, cbar_kws={'label': 'Agreement %'})
        else:
            im = ax.imshow(agreement_matrix, cmap='RdYlGn', vmin=0, vmax=100)
            ax.set_xticks(np.arange(n_annotators))
            ax.set_yticks(np.arange(n_annotators))
            ax.set_xticklabels(annotator_names)
            ax.set_yticklabels(annotator_names)
            plt.colorbar(im, ax=ax, label='Agreement %')
            for i in range(n_annotators):
                for j in range(n_annotators):
                    ax.text(j, i, f'{agreement_matrix[i, j]:.1f}', ha='center', va='center')
        
        ax.set_title('Pairwise Agreement Matrix (Relationship Annotations)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pairwise_agreement_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        fig_count += 1
        
        # Kappa heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if HAS_SEABORN:
            sns.heatmap(kappa_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                       xticklabels=display_labels, yticklabels=display_labels, ax=ax,
                       vmin=-1, vmax=1, cbar_kws={'label': "Cohen's Kappa"})
        else:
            im = ax.imshow(kappa_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(n_annotators))
            ax.set_yticks(np.arange(n_annotators))
            ax.set_xticklabels(annotator_names)
            ax.set_yticklabels(annotator_names)
            plt.colorbar(im, ax=ax, label="Cohen's Kappa")
            for i in range(n_annotators):
                for j in range(n_annotators):
                    ax.text(j, i, f'{kappa_matrix[i, j]:.3f}', ha='center', va='center')
        
        ax.set_title("Pairwise Cohen's Kappa Matrix (Relationship Annotations)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pairwise_kappa_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # ===== 2. Voting Distribution =====
    voting = results['voting_distribution']
    vote_dist = voting['vote_distribution']
    
    if vote_dist:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart of vote counts with rounded bars
        ax = axes[0]
        votes = sorted(vote_dist.keys())
        counts = [vote_dist[v] for v in votes]
        vote_colors = ['#95a5a6' if v < n_annotators/2 + 0.5 else '#3498db' if v < n_annotators else '#e67e22' 
                  for v in votes]
        
        x = np.arange(len(votes))
        for xi, count, color in zip(x, counts, vote_colors):
            draw_rounded_bars(ax, [xi], [count], width=0.5, color=color, alpha=0.85)
            ax.annotate(f'{count}', xy=(xi, count), xytext=(0, 3),
                       textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v}/{n_annotators}' for v in votes])
        ax.set_xlabel('Number of Annotators Agreeing', fontsize=12)
        ax.set_ylabel('Number of Relationships', fontsize=12)
        ax.set_title('Voting Distribution (Relationship Annotations)', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, len(votes) - 0.5)
        
        # Legend
        legend_elements = [
            Patch(facecolor='#e67e22', alpha=1.0, label='Unanimous'),
            Patch(facecolor='#3498db', alpha=1.0, label='Majority'),
            Patch(facecolor='#95a5a6', alpha=1.0, label='Disagreement'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Pie chart of agreement levels
        ax = axes[1]
        unanimous = voting['unanimous_count']
        majority_only = voting['majority_count'] - unanimous
        minority = voting['total_unique_relationships'] - voting['majority_count']
        
        pie_data = [unanimous, majority_only, minority]
        pie_labels = [f'Unanimous\n({unanimous})', f'Majority Only\n({majority_only})', f'Minority\n({minority})']
        pie_colors = ['#e67e22', '#3498db', '#95a5a6']
        
        ax.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', colors=pie_colors, startangle=90)
        ax.set_title('Agreement Level Distribution (Relationship Annotations)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'voting_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # ===== 3. Per-Annotator Statistics (Rounded Bars) =====
    per_ann = results['per_annotator_stats']
    
    if per_ann:
        # Create display names
        display_names = [results['summary']['annotator_display_names'][name] for name in per_ann.keys()]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Total relationships with rounded bars
        ax = axes[0]
        names = list(per_ann.keys())
        totals = [per_ann[n]['total_relationships'] for n in names]
        bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'][:len(names)]
        
        x = np.arange(len(names))
        for xi, total, color in zip(x, totals, bar_colors):
            draw_rounded_bars(ax, [xi], [total], width=0.5, color=color, alpha=0.85)
            ax.annotate(f'{total}', xy=(xi, total), xytext=(0, 3),
                       textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(display_names)
        ax.set_ylabel('Total Relationships', fontsize=12)
        ax.set_title('Relationships per Annotator', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, len(names) - 0.5)
        
        # Relationship type breakdown (stacked) - keep standard bars for stacked
        ax = axes[1]
        all_types = set()
        for stats in per_ann.values():
            all_types.update(stats['relationship_types'].keys())
        all_types = sorted(all_types)
        
        bottom = np.zeros(len(names))
        type_colors = {'TREATS': '#3498db', 'ADVERSE_EFFECT': '#e74c3c', 
                       'CONTRAINDICATED': '#9b59b6', 'DISCONTINUED': '#f39c12'}
        
        for rel_type in all_types:
            values = [per_ann[n]['relationship_types'].get(rel_type, 0) for n in names]
            color = type_colors.get(rel_type, '#95a5a6')
            ax.bar(display_names, values, width=0.5, bottom=bottom, label=rel_type, color=color, 
                   alpha=0.85, edgecolor='white', linewidth=0.5)
            bottom += values
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Relationship Types by Annotator', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_annotator_stats.png'), dpi=150, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # ===== 4. Relationship Type Distribution (Grouped Bars with Rounded Tops) =====
    if per_ann:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(per_ann.keys())
        all_types = set()
        for stats in per_ann.values():
            all_types.update(stats['relationship_types'].keys())
        all_types = sorted(all_types)
        
        # Create display names for grouping
        display_names_list = [results['summary']['annotator_display_names'][name] for name in names]
        
        x = np.arange(len(all_types))
        width = 0.22
        offset = np.linspace(-width*(n_annotators-1)/2, width*(n_annotators-1)/2, n_annotators)
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
        legend_handles = []
        
        for i, ann_name in enumerate(names):
            values = [per_ann[ann_name]['relationship_types'].get(t, 0) for t in all_types]
            # Draw rounded bars
            for j, (xi, val) in enumerate(zip(x + offset[i], values)):
                if val > 0:
                    draw_rounded_bars(ax, [xi], [val], width=width, 
                                     color=colors[i % len(colors)], alpha=0.85)
                    # Add value label
                    ax.annotate(f'{int(val)}', xy=(xi, val), xytext=(0, 3), 
                               textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            # Add legend handle
            legend_handles.append(Patch(facecolor=colors[i % len(colors)], alpha=0.85, label=display_names_list[i]))
        
        ax.set_xlabel('Relationship Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Relationship Type Distribution by Annotator', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_types, rotation=45, ha='right')
        ax.legend(handles=legend_handles, loc='upper left')
        ax.set_xlim(-0.5, len(all_types) - 0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'relationship_type_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # ===== 5. Confidence Level Distribution (Rounded Bars) =====
    if per_ann:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        names = list(per_ann.keys())
        all_conf = set()
        for stats in per_ann.values():
            all_conf.update(stats['confidence_levels'].keys())
        all_conf = sorted(all_conf, key=lambda x: ['high', 'medium', 'low'].index(x) if x in ['high', 'medium', 'low'] else 3)
        
        # Create display names for legend
        display_names_list = [results['summary']['annotator_display_names'][name] for name in names]
        
        x = np.arange(len(all_conf))
        width = 0.22
        offset = np.linspace(-width*(n_annotators-1)/2, width*(n_annotators-1)/2, n_annotators)
        
        colors = ['#9b59b6', '#f39c12', '#1abc9c']
        legend_handles = []
        
        for i, ann_name in enumerate(names):
            values = [per_ann[ann_name]['confidence_levels'].get(c, 0) for c in all_conf]
            for j, (xi, val) in enumerate(zip(x + offset[i], values)):
                if val > 0:
                    draw_rounded_bars(ax, [xi], [val], width=width, 
                                     color=colors[i % len(colors)], alpha=0.85)
                    ax.annotate(f'{int(val)}', xy=(xi, val), xytext=(0, 3),
                               textcoords="offset points", ha='center', va='bottom', fontsize=9)
            legend_handles.append(Patch(facecolor=colors[i % len(colors)], alpha=0.85, label=display_names_list[i]))
        
        ax.set_xlabel('Confidence Level', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Confidence Level Distribution by Annotator', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_conf)
        ax.legend(handles=legend_handles)
        ax.set_xlim(-0.5, len(all_conf) - 0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # ===== 6. Evidence Scope Distribution (Rounded Bars) =====
    if per_ann:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        names = list(per_ann.keys())
        all_scope = set()
        for stats in per_ann.values():
            all_scope.update(stats['evidence_scopes'].keys())
        all_scope = sorted(all_scope)
        
        # Create display names for legend
        display_names_list = [results['summary']['annotator_display_names'][name] for name in names]
        
        x = np.arange(len(all_scope))
        width = 0.22
        offset = np.linspace(-width*(n_annotators-1)/2, width*(n_annotators-1)/2, n_annotators)
        
        colors = ['#1abc9c', '#e67e22', '#3498db']
        legend_handles = []
        
        for i, ann_name in enumerate(names):
            values = [per_ann[ann_name]['evidence_scopes'].get(s, 0) for s in all_scope]
            for j, (xi, val) in enumerate(zip(x + offset[i], values)):
                if val > 0:
                    draw_rounded_bars(ax, [xi], [val], width=width, 
                                     color=colors[i % len(colors)], alpha=0.85)
                    ax.annotate(f'{int(val)}', xy=(xi, val), xytext=(0, 3),
                               textcoords="offset points", ha='center', va='bottom', fontsize=9)
            legend_handles.append(Patch(facecolor=colors[i % len(colors)], alpha=0.85, label=display_names_list[i]))
        
        ax.set_xlabel('Evidence Scope', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Evidence Scope Distribution by Annotator', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_scope)
        ax.legend(handles=legend_handles)
        ax.set_xlim(-0.5, len(all_scope) - 0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evidence_scope_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # ===== 7. Disagreement Analysis (Rounded Bars) =====
    voting = results['voting_distribution']
    if voting['total_unique_relationships'] > 0:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        
        # Agreement breakdown by annotator pair
        ax = axes[0]
        pairwise = results['pairwise_agreement']
        if pairwise:
            pair_labels = []
            agreed_counts = []
            only_first = []
            only_second = []
            
            for pair_key, metrics in pairwise.items():
                # Create friendly pair label
                ann1 = metrics['annotator_1']
                ann2 = metrics['annotator_2']
                friendly_pair = f"{results['summary']['annotator_display_names'][ann1]}\nvs\n{results['summary']['annotator_display_names'][ann2]}"
                pair_labels.append(friendly_pair)
                agreed_counts.append(metrics['intersection'])
                only_first.append(metrics['ann1_count'] - metrics['intersection'])
                only_second.append(metrics['ann2_count'] - metrics['intersection'])
            
            x = np.arange(len(pair_labels))
            width = 0.22
            
            # Draw rounded bars for each group
            legend_handles = []
            for xi in range(len(x)):
                draw_rounded_bars(ax, [x[xi] - width], [agreed_counts[xi]], width=width, 
                                 color='#2ecc71', alpha=0.85)
                draw_rounded_bars(ax, [x[xi]], [only_first[xi]], width=width, 
                                 color='#3498db', alpha=0.85)
                draw_rounded_bars(ax, [x[xi] + width], [only_second[xi]], width=width, 
                                 color='#e74c3c', alpha=0.85)
            
            legend_handles = [
                Patch(facecolor='#2ecc71', alpha=0.85, label='Both Agree'),
                Patch(facecolor='#3498db', alpha=0.85, label='Only First'),
                Patch(facecolor='#e74c3c', alpha=0.85, label='Only Second'),
            ]
            
            ax.set_xlabel('Annotator Pair', fontsize=12)
            ax.set_ylabel('Number of Relationships', fontsize=12)
            ax.set_title('Pairwise Disagreement Analysis', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(pair_labels, fontsize=9)
            ax.legend(handles=legend_handles)
            ax.set_xlim(-0.5, len(pair_labels) - 0.5)
        
        # Unique relationships by source with rounded bars
        ax = axes[1]
        
        # Count how many relationships each annotator uniquely has
        unique_counts = {}
        for ann_name in annotator_names:
            unique_counts[ann_name] = 0
        
        for rel_vote in voting['relationship_votes']:
            if rel_vote['vote_count'] == 1:
                ann = rel_vote['annotators'][0]
                unique_counts[ann] = unique_counts.get(ann, 0) + 1
        
        names = list(unique_counts.keys())
        display_names_list = [results['summary']['annotator_display_names'][n] for n in names]
        counts = [unique_counts[n] for n in names]
        bar_colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(names)]
        
        x = np.arange(len(names))
        for xi, count, color in zip(x, counts, bar_colors):
            if count > 0:
                draw_rounded_bars(ax, [xi], [count], width=0.5, color=color, alpha=0.85)
                ax.annotate(f'{count}', xy=(xi, count), xytext=(0, 3),
                           textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(display_names_list)
        ax.set_xlabel('Annotator', fontsize=12)
        ax.set_ylabel('Unique Relationships (no agreement)', fontsize=12)
        ax.set_title('Relationships Found by Only One Annotator', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, len(names) - 0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'disagreement_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # ===== 8. Mention Type Analysis =====
    if 'mention_type_stats' in results:
        mention_stats = results['mention_type_stats']
        per_ann_mentions = mention_stats['per_annotator_mention_types']
        
        if per_ann_mentions:
            # Figure 1: Mention Types per Annotator (Drug Table)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Medication Table Mention Types
            ax = axes[0]
            names = list(per_ann_mentions.keys())
            display_names_list = [results['summary']['annotator_display_names'][name] for name in names]
            all_mention_types = set()
            for stats in per_ann_mentions.values():
                all_mention_types.update(stats['medication'].keys())
            all_mention_types = sorted(all_mention_types)
            
            x = np.arange(len(all_mention_types))
            width = 0.22
            offset = np.linspace(-width*(n_annotators-1)/2, width*(n_annotators-1)/2, n_annotators)
            
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
            legend_handles = []
            
            for i, ann_name in enumerate(names):
                values = [per_ann_mentions[ann_name]['medication'].get(mt, 0) for mt in all_mention_types]
                for j, (xi, val) in enumerate(zip(x + offset[i], values)):
                    if val > 0:
                        draw_rounded_bars(ax, [xi], [val], width=width, 
                                         color=colors[i % len(colors)], alpha=0.85)
                        ax.annotate(f'{int(val)}', xy=(xi, val), xytext=(0, 3),
                                   textcoords="offset points", ha='center', va='bottom', fontsize=8)
                legend_handles.append(Patch(facecolor=colors[i % len(colors)], alpha=0.85, label=display_names_list[i]))
            
            ax.set_xlabel('Mention Type', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Medication Row Grounding - Mention Types by Annotator', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(all_mention_types, rotation=45, ha='right')
            ax.legend(handles=legend_handles, loc='upper left')
            ax.set_xlim(-0.5, len(all_mention_types) - 0.5)
            
            # Diagnosis Table Mention Types
            ax = axes[1]
            all_mention_types_diag = set()
            for stats in per_ann_mentions.values():
                all_mention_types_diag.update(stats['diagnosis'].keys())
            all_mention_types_diag = sorted(all_mention_types_diag)
            
            x = np.arange(len(all_mention_types_diag))
            legend_handles = []
            
            for i, ann_name in enumerate(names):
                values = [per_ann_mentions[ann_name]['diagnosis'].get(mt, 0) for mt in all_mention_types_diag]
                for j, (xi, val) in enumerate(zip(x + offset[i], values)):
                    if val > 0:
                        draw_rounded_bars(ax, [xi], [val], width=width, 
                                         color=colors[i % len(colors)], alpha=0.85)
                        ax.annotate(f'{int(val)}', xy=(xi, val), xytext=(0, 3),
                                   textcoords="offset points", ha='center', va='bottom', fontsize=8)
                legend_handles.append(Patch(facecolor=colors[i % len(colors)], alpha=0.85, label=display_names_list[i]))
            
            ax.set_xlabel('Mention Type', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Diagnosis Row Grounding - Mention Types by Annotator', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(all_mention_types_diag, rotation=45, ha='right')
            ax.legend(handles=legend_handles, loc='upper left')
            ax.set_xlim(-0.5, len(all_mention_types_diag) - 0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mention_types_by_annotator.png'), dpi=150, bbox_inches='tight')
            plt.close()
            fig_count += 1
            
            # Figure 2: Overall Mention Type Distribution per Table
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Medication table overall
            ax = axes[0]
            drug_mt = mention_stats['mention_type_per_table']['medication']
            if drug_mt:
                mt_types = sorted(drug_mt.keys())
                counts = [drug_mt[mt] for mt in mt_types]
                x = np.arange(len(mt_types))
                
                mention_type_colors = {
                    'explicit': '#2ecc71',
                    'abbreviated': '#e74c3c', 
                    'brand_name': '#3498db',
                    'synonym': '#f39c12',
                    'context': '#9b59b6'
                }
                bar_colors = [mention_type_colors.get(mt, '#95a5a6') for mt in mt_types]
                
                for xi, count, color in zip(x, counts, bar_colors):
                    draw_rounded_bars(ax, [xi], [count], width=0.6, color=color, alpha=0.85)
                    ax.annotate(f'{count}', xy=(xi, count), xytext=(0, 3),
                               textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels(mt_types, rotation=45, ha='right')
                ax.set_ylabel('Total Count', fontsize=12)
                ax.set_title('Medication Row Grounding - Overall Mention Type Distribution', fontsize=14, fontweight='bold')
                ax.set_xlim(-0.5, len(mt_types) - 0.5)
            
            # Diagnosis table overall
            ax = axes[1]
            diag_mt = mention_stats['mention_type_per_table']['diagnosis']
            if diag_mt:
                mt_types = sorted(diag_mt.keys())
                counts = [diag_mt[mt] for mt in mt_types]
                x = np.arange(len(mt_types))
                
                bar_colors = [mention_type_colors.get(mt, '#95a5a6') for mt in mt_types]
                
                for xi, count, color in zip(x, counts, bar_colors):
                    draw_rounded_bars(ax, [xi], [count], width=0.6, color=color, alpha=0.85)
                    ax.annotate(f'{count}', xy=(xi, count), xytext=(0, 3),
                               textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels(mt_types, rotation=45, ha='right')
                ax.set_ylabel('Total Count', fontsize=12)
                ax.set_title('Diagnosis Row Grounding - Overall Mention Type Distribution', fontsize=14, fontweight='bold')
                ax.set_xlim(-0.5, len(mt_types) - 0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mention_types_per_table.png'), dpi=150, bbox_inches='tight')
            plt.close()
            fig_count += 1
    
    # ===== 9. Mention Type Agreement Analysis =====
    if 'mention_type_agreement' in results:
        mt_agreement = results['mention_type_agreement']
        
        # Create figure with 2 subplots (diagnosis and medication)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Diagnosis Mention Type Agreement (Stacked Counts with Minority)
        ax = axes[0]
        diag_agreement = mt_agreement.get('diagnosis_mention_agreement', {})
        if diag_agreement:
            mt_types = sorted(diag_agreement.keys(), 
                            key=lambda x: ['explicit', 'synonym', 'abbreviated', 'context'].index(x) 
                            if x in ['explicit', 'synonym', 'abbreviated', 'context'] else 999)
            unanimous_counts = [diag_agreement[mt]['unanimous'] for mt in mt_types]
            majority_only_counts = [diag_agreement[mt]['majority_only'] for mt in mt_types]
            type_disagreement_counts = [diag_agreement[mt]['type_disagreement'] for mt in mt_types]
            no_overlap_counts = [diag_agreement[mt]['no_grounding_overlap'] for mt in mt_types]
            totals = [diag_agreement[mt]['total'] for mt in mt_types]
            
            x = np.arange(len(mt_types))
            width = 0.35
            
            # Draw grouped bars: Agreement (blue+orange stacked) and Disagreement (dark gray + light gray stacked)
            # Agreement bar (left): Majority-only (blue) + Unanimous (orange)
            draw_rounded_bars(ax, x - width/2, majority_only_counts, width=width,
                             color='#3498db', alpha=1.0)  # Blue
            for i, (maj, unan) in enumerate(zip(majority_only_counts, unanimous_counts)):
                if unan > 0:
                    draw_rounded_bars(ax, [x[i] - width/2], [unan], width=width,
                                     color='#e67e22', alpha=1.0, bottom=[maj])  # Orange
            
            # Disagreement bar (right): Type disagreement (dark gray) + No grounding overlap (light  gray)
            draw_rounded_bars(ax, x + width/2, type_disagreement_counts, width=width,
                             color='#95a5a6', alpha=1.0)  # Dark gray
            for i, (type_dis, no_overlap) in enumerate(zip(type_disagreement_counts, no_overlap_counts)):
                if no_overlap > 0:
                    draw_rounded_bars(ax, [x[i] + width/2], [no_overlap], width=width,
                                     color='#bdc3c7', alpha=1.0, bottom=[type_dis])  # Light gray
            
            # Add annotations
            for xi, maj, unan, type_dis, no_overlap in zip(x, majority_only_counts, unanimous_counts, 
                                                            type_disagreement_counts, no_overlap_counts):
                # Agreement bar annotations
                if maj > 3:
                    ax.annotate(f'{maj}', xy=(xi - width/2, maj/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                if unan > 3:
                    ax.annotate(f'{unan}', xy=(xi - width/2, maj + unan/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                # Total on top of agreement bar
                total_agree = maj + unan
                if total_agree > 0:
                    ax.annotate(f'{total_agree}', xy=(xi - width/2, total_agree), xytext=(0, 3),
                               textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
                
                # Disagreement bar annotations
                if type_dis > 3:
                    ax.annotate(f'{type_dis}', xy=(xi + width/2, type_dis/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                if no_overlap > 3:
                    ax.annotate(f'{no_overlap}', xy=(xi + width/2, type_dis + no_overlap/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                # Total on top of disagreement bar
                total_disagree = type_dis + no_overlap
                if total_disagree > 0:
                    ax.annotate(f'{total_disagree}', xy=(xi + width/2, total_disagree), xytext=(0, 3),
                               textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels(mt_types, rotation=45, ha='right')
            ax.set_ylabel('Number of Cases', fontsize=12)
            ax.set_title('Diagnosis Mention Type Agreement', fontsize=13, fontweight='bold')
            ax.set_xlim(-0.6, len(mt_types) - 0.4)
            
            # Two separate legend boxes stacked vertically at top right
            agreement_legend = ax.legend(handles=[Patch(facecolor='#e67e22', label='Unanimous (all 3)'),
                                                  Patch(facecolor='#3498db', label='Majority (2 of 3)')],
                                        title='Agreement', loc='upper right', fontsize=9,
                                        title_fontsize=10, frameon=True, bbox_to_anchor=(1.0, 1.0))
            ax.add_artist(agreement_legend)
            ax.legend(handles=[Patch(facecolor='#95a5a6', label='Type Disagreement'),
                             Patch(facecolor='#bdc3c7', label='No Grounding Overlap')],
                     title='Disagreement', loc='upper right', fontsize=9,
                     title_fontsize=10, frameon=True, bbox_to_anchor=(1.0, 0.82))
        
        # Medication Mention Type Agreement (Stacked Counts with Type Dis and No Overlap)
        ax = axes[1]
        med_agreement = mt_agreement.get('medication_mention_agreement', {})
        if med_agreement:
            mt_types = sorted(med_agreement.keys(),
                            key=lambda x: ['explicit', 'brand_name', 'context'].index(x)
                            if x in ['explicit', 'brand_name', 'context'] else 999)
            unanimous_counts = [med_agreement[mt]['unanimous'] for mt in mt_types]
            majority_only_counts = [med_agreement[mt]['majority_only'] for mt in mt_types]
            type_disagreement_counts = [med_agreement[mt]['type_disagreement'] for mt in mt_types]
            no_overlap_counts = [med_agreement[mt]['no_grounding_overlap'] for mt in mt_types]
            totals = [med_agreement[mt]['total'] for mt in mt_types]
            
            x = np.arange(len(mt_types))
            width = 0.35
            
            # Draw grouped bars: Agreement (blue+orange stacked) and Disagreement (dark gray + light gray stacked)
            # Agreement bar (left): Majority-only (blue) + Unanimous (orange)
            draw_rounded_bars(ax, x - width/2, majority_only_counts, width=width,
                             color='#3498db', alpha=1.0)  # Blue
            for i, (maj, unan) in enumerate(zip(majority_only_counts, unanimous_counts)):
                if unan > 0:
                    draw_rounded_bars(ax, [x[i] - width/2], [unan], width=width,
                                     color='#e67e22', alpha=1.0, bottom=[maj])  # Orange
            
            # Disagreement bar (right): Type disagreement (dark gray) + No grounding overlap (light gray)
            draw_rounded_bars(ax, x + width/2, type_disagreement_counts, width=width,
                             color='#95a5a6', alpha=1.0)  # Dark gray
            for i, (type_dis, no_overlap) in enumerate(zip(type_disagreement_counts, no_overlap_counts)):
                if no_overlap > 0:
                    draw_rounded_bars(ax, [x[i] + width/2], [no_overlap], width=width,
                                     color='#bdc3c7', alpha=1.0, bottom=[type_dis])  # Light gray
            
            # Add annotations
            for xi, maj, unan, type_dis, no_overlap in zip(x, majority_only_counts, unanimous_counts,
                                                            type_disagreement_counts, no_overlap_counts):
                # Agreement bar annotations
                if maj > 3:
                    ax.annotate(f'{maj}', xy=(xi - width/2, maj/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                if unan > 3:
                    ax.annotate(f'{unan}', xy=(xi - width/2, maj + unan/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                # Total on top of agreement bar
                total_agree = maj + unan
                if total_agree > 0:
                    ax.annotate(f'{total_agree}', xy=(xi - width/2, total_agree), xytext=(0, 3),
                               textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
                
                # Disagreement bar annotations
                if type_dis > 3:
                    ax.annotate(f'{type_dis}', xy=(xi + width/2, type_dis/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                if no_overlap > 3:
                    ax.annotate(f'{no_overlap}', xy=(xi + width/2, type_dis + no_overlap/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                # Total on top of disagreement bar
                total_disagree = type_dis + no_overlap
                if total_disagree > 0:
                    ax.annotate(f'{total_disagree}', xy=(xi + width/2, total_disagree), xytext=(0, 3),
                               textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels(mt_types, rotation=45, ha='right')
            ax.set_ylabel('Number of Cases', fontsize=12)
            ax.set_title('Medication Mention Type Agreement', fontsize=13, fontweight='bold')
            ax.set_xlim(-0.6, len(mt_types) - 0.4)
            
            # Two separate legend boxes stacked vertically at top right
            agreement_legend = ax.legend(handles=[Patch(facecolor='#e67e22', label='Unanimous (all 3)'),
                                                  Patch(facecolor='#3498db', label='Majority (2 of 3)')],
                                        title='Agreement', loc='upper right', fontsize=9,
                                        title_fontsize=10, frameon=True, bbox_to_anchor=(1.0, 1.0))
            ax.add_artist(agreement_legend)
            ax.legend(handles=[Patch(facecolor='#95a5a6', label='Type Disagreement'),
                             Patch(facecolor='#bdc3c7', label='No Grounding Overlap')],
                     title='Disagreement', loc='upper right', fontsize=9,
                     title_fontsize=10, frameon=True, bbox_to_anchor=(1.0, 0.82))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mention_type_agreement.png'), dpi=150, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # ===== 10. Evidence Scope Agreement Analysis =====
    if 'evidence_scope_agreement' in results:
        scope_agreement = results['evidence_scope_agreement']
        scope_rates = scope_agreement.get('scope_agreement_rates', {})
        
        if scope_rates:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Unanimous and Majority Agreement Rates by Scope (Stacked Counts with Disagreement Breakdown)
            ax = axes[0]
            scope_types = sorted(scope_rates.keys(),
                               key=lambda x: ['explicit', 'section', 'document', 'context'].index(x)
                               if x in ['explicit', 'section', 'document', 'context'] else 999)
            unanimous_counts = [scope_rates[s]['unanimous_count'] for s in scope_types]
            majority_counts = [scope_rates[s]['majority_count'] for s in scope_types]
            scope_disagreement_counts = [scope_rates[s]['scope_disagreement'] for s in scope_types]
            no_overlap_counts = [scope_rates[s]['no_existence_overlap'] for s in scope_types]
            
            # Majority_only = majority (since our new logic counts majority separately from unanimous)
            majority_only_counts = majority_counts
            totals = [scope_rates[s]['total_instances'] for s in scope_types]
            
            x = np.arange(len(scope_types))
            width = 0.35
            
            # Draw grouped bars: Agreement (blue+orange stacked) and Disagreement (dark gray + light gray stacked)
            # Agreement bar (left): Majority-only (blue) + Unanimous (orange) stacked
            draw_rounded_bars(ax, x - width/2, majority_only_counts, width=width,
                             color='#3498db', alpha=1.0)  # Blue
            for i, (maj, unan) in enumerate(zip(majority_only_counts, unanimous_counts)):
                if unan > 0:
                    draw_rounded_bars(ax, [x[i] - width/2], [unan], width=width,
                                     color='#e67e22', alpha=1.0, bottom=[maj])  # Orange
            
            # Disagreement bar (right): Scope Disagreement (Dark Gray) + No Existence Overlap (Light Gray)
            draw_rounded_bars(ax, x + width/2, scope_disagreement_counts, width=width,
                             color='#95a5a6', alpha=1.0)  # Dark Gray
            for i, (scope_dis, no_overlap) in enumerate(zip(scope_disagreement_counts, no_overlap_counts)):
                if no_overlap > 0:
                    draw_rounded_bars(ax, [x[i] + width/2], [no_overlap], width=width,
                                     color='#bdc3c7', alpha=1.0, bottom=[scope_dis])  # Light Gray
            
            # Add annotations
            for xi, maj_only, unan, scope_dis, no_overlap in zip(x, majority_only_counts, unanimous_counts, 
                                                                  scope_disagreement_counts, no_overlap_counts):
                # Agreement bar annotations
                if maj_only > 3:
                    ax.annotate(f'{maj_only}', xy=(xi - width/2, maj_only/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                if unan > 3:
                    ax.annotate(f'{unan}', xy=(xi - width/2, maj_only + unan/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                # Total on top of agreement bar
                total_agree = maj_only + unan
                if total_agree > 0:
                    ax.annotate(f'{total_agree}', xy=(xi - width/2, total_agree), xytext=(0, 3),
                               textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
                
                # Disagreement bar annotations
                if scope_dis > 3:
                    ax.annotate(f'{scope_dis}', xy=(xi + width/2, scope_dis/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                if no_overlap > 3:
                    ax.annotate(f'{no_overlap}', xy=(xi + width/2, scope_dis + no_overlap/2),
                               textcoords="data", ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
                # Total on top of disagreement bar
                total_disagree = scope_dis + no_overlap
                if total_disagree > 0:
                    ax.annotate(f'{total_disagree}', xy=(xi + width/2, total_disagree), xytext=(0, 3),
                               textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels([f'{s}\n(n={scope_rates[s]["total_instances"]})' for s in scope_types], 
                              rotation=0, ha='center', fontsize=10)
            ax.set_ylabel('Number of Cases', fontsize=12)
            ax.set_title('Evidence Scope Classification Agreement', fontsize=13, fontweight='bold')
            ax.set_xlim(-0.6, len(scope_types) - 0.4)
            
            # Two separate legend boxes stacked vertically at top right
            agreement_legend = ax.legend(handles=[Patch(facecolor='#e67e22', label='Unanimous (all 3)'),
                                                  Patch(facecolor='#3498db', label='Majority (2 of 3)')],
                                        title='Agreement', loc='upper right', fontsize=9,
                                        title_fontsize=10, frameon=True, bbox_to_anchor=(1.0, 1.0))
            ax.add_artist(agreement_legend)
            ax.legend(handles=[Patch(facecolor='#95a5a6', label='Scope Disagreement'),
                             Patch(facecolor='#bdc3c7', label='No Existence Overlap')],
                     title='Disagreement', loc='upper right', fontsize=9,
                     title_fontsize=10, frameon=True, bbox_to_anchor=(1.0, 0.82))
            
            # Pairwise Scope Agreement
            ax = axes[1]
            pairwise_scope = scope_agreement.get('pairwise_scope_agreement', {})
            
            if pairwise_scope:
                pairs = list(pairwise_scope.keys())
                # Create friendly pair labels
                pair_labels = []
                for pair in pairs:
                    ann1 = pair.split('_vs_')[0]
                    ann2 = pair.split('_vs_')[1]
                    friendly1 = results['summary']['annotator_display_names'].get(ann1, ann1.split('/')[-1])
                    friendly2 = results['summary']['annotator_display_names'].get(ann2, ann2.split('/')[-1])
                    pair_labels.append(f'{friendly1}\nvs\n{friendly2}')
                
                agreement_rates = [pairwise_scope[p]['agreement_rate'] for p in pairs]
                totals = [pairwise_scope[p]['total_comparisons'] for p in pairs]
                
                colors = ['#2ecc71' if rate > 60 else '#f39c12' if rate > 40 else '#e74c3c'
                         for rate in agreement_rates]
                
                x = np.arange(len(pairs))
                for xi, rate, total, color in zip(x, agreement_rates, totals, colors):
                    draw_rounded_bars(ax, [xi], [rate], width=0.5, color=color, alpha=0.85)
                    ax.annotate(f'{rate:.1f}%\n(n={total})', xy=(xi, rate), xytext=(0, 3),
                               textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels(pair_labels, fontsize=9)
                ax.set_ylabel('Agreement Rate (%)', fontsize=12)
                ax.set_title('Pairwise Evidence Scope Agreement', fontsize=14, fontweight='bold')
                ax.set_xlim(-0.5, len(pairs) - 0.5)
                ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.text(len(pairs)-0.6, 52, '50% threshold', fontsize=8, color='gray')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'evidence_scope_agreement.png'), dpi=150, bbox_inches='tight')
            plt.close()
            fig_count += 1
    
    # ===== 11. Relationship Type Agreement =====
    if 'relationship_type_by_voting' in results:
        type_by_voting = results['relationship_type_by_voting']
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Get all unique relationship types
        all_types = set()
        for level_data in type_by_voting.values():
            all_types.update(level_data.keys())
        
        rel_types = sorted(all_types,
                          key=lambda x: ['TREATS', 'ADVERSE_EFFECT', 'CONTRAINDICATED', 'DISCONTINUED'].index(x)
                          if x in ['TREATS', 'ADVERSE_EFFECT', 'CONTRAINDICATED', 'DISCONTINUED'] else 999)
        
        # Get counts for each voting level
        unanimous_counts = [type_by_voting['unanimous'].get(t, 0) for t in rel_types]
        majority_counts = [type_by_voting['majority'].get(t, 0) for t in rel_types]
        minority_counts = [type_by_voting['minority'].get(t, 0) for t in rel_types]
        
        x = np.arange(len(rel_types))
        width = 0.35
        
        # Draw grouped bars: Agreement (blue+orange stacked) and Disagreement (gray)
        # Agreement bar (left): Majority (blue) + Unanimous (orange) stacked
        draw_rounded_bars(ax, x - width/2, majority_counts, width=width,
                         color='#3498db', alpha=1.0)  # Blue for majority
        for i, (maj, unan) in enumerate(zip(majority_counts, unanimous_counts)):
            if unan > 0:
                draw_rounded_bars(ax, [x[i] - width/2], [unan], width=width,
                                 color='#e67e22', alpha=1.0, bottom=[maj])  # Orange for unanimous
        
        # Disagreement bar (right): Minority
        draw_rounded_bars(ax, x + width/2, minority_counts, width=width,
                         color='#95a5a6', alpha=1.0)  # Gray for disagreement
        
        # Add annotations
        for xi, maj, unan, minor in zip(x, majority_counts, unanimous_counts, minority_counts):
            # Agreement bar annotations
            if maj > 3:
                ax.annotate(f'{maj}', xy=(xi - width/2, maj/2),
                           textcoords="data", ha='center', va='center',
                           fontsize=9, fontweight='bold', color='white')
            if unan > 3:
                ax.annotate(f'{unan}', xy=(xi - width/2, maj + unan/2),
                           textcoords="data", ha='center', va='center',
                           fontsize=9, fontweight='bold', color='white')
            # Total on top of agreement bar
            total_agree = maj + unan
            if total_agree > 0:
                ax.annotate(f'{total_agree}', xy=(xi - width/2, total_agree), xytext=(0, 3),
                           textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
            
            # Disagreement bar annotation
            if minor > 3:
                ax.annotate(f'{minor}', xy=(xi + width/2, minor/2),
                           textcoords="data", ha='center', va='center',
                           fontsize=9, fontweight='bold', color='white')
            # Total on top of disagreement bar
            if minor > 0:
                ax.annotate(f'{minor}', xy=(xi + width/2, minor), xytext=(0, 3),
                           textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
        
        # Compute totals for each type
        totals = [u + m + d for u, m, d in zip(unanimous_counts, majority_counts, minority_counts)]
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t}\n(n={totals[i]})' for i, t in enumerate(rel_types)],
                          rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Number of Relationships', fontsize=12)
        ax.set_title('Relationship Type Agreement',
                    fontsize=13, fontweight='bold')
        ax.set_xlim(-0.6, len(rel_types) - 0.4)
        
        # Two separate legend boxes stacked vertically at top right
        agreement_legend = ax.legend(handles=[Patch(facecolor='#e67e22', label='Unanimous (3/3)'),
                                              Patch(facecolor='#3498db', label='Majority (2/3)')],
                                    title='Agreement', loc='upper right', fontsize=9,
                                    title_fontsize=10, frameon=True, bbox_to_anchor=(1.0, 1.0))
        ax.add_artist(agreement_legend)
        ax.legend(handles=[Patch(facecolor='#95a5a6', label='Disagreement (1/3)')],
                 title='Disagreement', loc='upper right', fontsize=9,
                 title_fontsize=10, frameon=True, bbox_to_anchor=(1.0, 0.87))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'relationship_type_agreement.png'), dpi=150, bbox_inches='tight')
        plt.close()
        fig_count += 1
    
    # ===== 12. Summary Dashboard =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Summary stats
    ax = axes[0, 0]
    ax.axis('off')
    
    fleiss = results['fleiss_kappa']
    fleiss_interp = ('Almost Perfect' if fleiss > 0.8 else 'Substantial' if fleiss > 0.6 else
                     'Moderate' if fleiss > 0.4 else 'Fair' if fleiss > 0.2 else 'Slight' if fleiss > 0 else 'Poor')
    
    summary_text = f"""
    MULTI-ANNOTATOR AGREEMENT SUMMARY
    ====================================
    
    Annotators:           {n_annotators}
    Total Files:          {results['summary']['total_files']}
    Common Files:         {results['summary']['common_files']}
    
    Fleiss' Kappa:        {fleiss:.4f} ({fleiss_interp})
    
    Total Unique Rels:    {voting['total_unique_relationships']}
    Unanimous ({n_annotators}/{n_annotators}):     {voting['unanimous_count']} ({voting['unanimous_pct']:.1f}%)
    Majority (>={int(n_annotators/2+1)}/{n_annotators}):     {voting['majority_count']} ({voting['majority_pct']:.1f}%)
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    ax.set_title('Summary', fontsize=14, fontweight='bold')
    
    # Fleiss Kappa gauge
    ax = axes[0, 1]
    
    # Create a simple gauge visualization
    kappa_normalized = (fleiss + 1) / 2  # Normalize to 0-1
    ax.barh(['Fleiss Kappa'], [kappa_normalized], color='#3498db', alpha=0.8)
    ax.barh(['Fleiss Kappa'], [1 - kappa_normalized], left=[kappa_normalized], color='#ecf0f1', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
    ax.set_xlabel('Kappa Value')
    ax.set_title(f"Fleiss' Kappa: {fleiss:.4f} ({fleiss_interp})", fontsize=14, fontweight='bold')
    
    # Pairwise summary
    ax = axes[1, 0]
    pairwise = results['pairwise_agreement']
    if pairwise:
        pairs = list(pairwise.keys())
        # Create friendly pair labels
        pair_labels = []
        for pair_key in pairs:
            metrics = pairwise[pair_key]
            ann1 = metrics['annotator_1']
            ann2 = metrics['annotator_2']
            friendly_pair = f"{results['summary']['annotator_display_names'][ann1]} vs {results['summary']['annotator_display_names'][ann2]}"
            pair_labels.append(friendly_pair)
        agreements = [pairwise[p]['agreement_pct'] for p in pairs]
        
        colors = ['#2ecc71' if a >= 60 else '#f39c12' if a >= 40 else '#e74c3c' for a in agreements]
        bars = ax.barh(pair_labels, agreements, color=colors, alpha=0.8)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Agreement %')
        ax.set_title('Pairwise Agreement', fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, agreements):
            ax.annotate(f'{val:.1f}%', xy=(val, bar.get_y() + bar.get_height()/2),
                       xytext=(5, 0), textcoords="offset points", va='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Insufficient annotators', ha='center', va='center')
        ax.axis('off')
    
    # Interpretation guide
    ax = axes[1, 1]
    ax.axis('off')
    
    interp_text = """
    KAPPA INTERPRETATION (Landis & Koch)
    ========================================
    
    < 0.00    Poor
    0.00-0.20 Slight
    0.21-0.40 Fair
    0.41-0.60 Moderate
    0.61-0.80 Substantial
    0.81-1.00 Almost Perfect
    
    VOTING RECOMMENDATION
    ========================================
    
    Unanimous: Include with HIGH confidence
    Majority:  Include with MEDIUM confidence
    Minority:  Review/Exclude
    """
    ax.text(0.1, 0.5, interp_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.suptitle('Multi-Annotator Agreement Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_annotator_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    fig_count += 1
    
    print(f"Generated {fig_count} visualization(s) in {output_dir}")


def print_multi_annotator_report(results: Dict):
    """Print formatted report to console."""
    
    print("\n" + "="*80)
    print("  MULTI-ANNOTATOR AGREEMENT ANALYSIS REPORT")
    print("="*80 + "\n")
    
    summary = results['summary']
    display_names = summary['annotator_display_names']
    
    print("[OVERVIEW]")
    print("-" * 40)
    print(f"  Number of Annotators:     {summary['n_annotators']}")
    print(f"  Annotator Names:          {', '.join([display_names[n] for n in summary['annotator_names']])}")
    print(f"  Total Files:              {summary['total_files']}")
    print(f"  Files in All Annotators:  {summary['common_files']}")
    print()
    
    print("[FLEISS' KAPPA (Multi-Rater Agreement)]")
    print("-" * 40)
    fleiss = results['fleiss_kappa']
    fleiss_interp = ('Almost Perfect' if fleiss > 0.8 else 'Substantial' if fleiss > 0.6 else
                     'Moderate' if fleiss > 0.4 else 'Fair' if fleiss > 0.2 else 'Slight' if fleiss > 0 else 'Poor')
    print(f"  Fleiss' Kappa:            {fleiss:.4f}")
    print(f"  Interpretation:           {fleiss_interp}")
    print()
    
    print("[VOTING DISTRIBUTION]")
    print("-" * 40)
    voting = results['voting_distribution']
    print(f"  Total Unique Relationships:  {voting['total_unique_relationships']}")
    print(f"  Unanimous Agreement:         {voting['unanimous_count']} ({voting['unanimous_pct']:.1f}%)")
    print(f"  Majority Agreement:          {voting['majority_count']} ({voting['majority_pct']:.1f}%)")
    print()
    print("  Vote Distribution:")
    for votes, count in sorted(voting['vote_distribution'].items()):
        marker = "[UNANIMOUS]" if votes == voting['n_annotators'] else "[MAJORITY]" if votes >= voting['n_annotators']/2 + 0.5 else "[MINORITY]"
        print(f"    {votes}/{voting['n_annotators']} annotators: {count:>5} relationships {marker}")
    print()
    
    print("[PAIRWISE AGREEMENT]")
    print("-" * 40)
    print(f"  {'Pair':<30} {'Agree%':>10} {'Kappa':>10} {'Intersection':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12}")
    for pair_key, metrics in results['pairwise_agreement'].items():
        # Get friendly names for the pair
        ann1 = metrics['annotator_1']
        ann2 = metrics['annotator_2']
        pair_name = f"{display_names[ann1]} vs {display_names[ann2]}"
        print(f"  {pair_name:<30} {metrics['agreement_pct']:>9.1f}% {metrics['cohens_kappa']:>10.4f} {metrics['intersection']:>12}")
    print()
    
    print("[PER-ANNOTATOR STATISTICS]")
    print("-" * 40)
    print(f"  {'Annotator':<20} {'Relationships':>15} {'Avg/File':>10}")
    print(f"  {'-'*20} {'-'*15} {'-'*10}")
    for ann_name, stats in results['per_annotator_stats'].items():
        friendly_name = display_names[ann_name]
        print(f"  {friendly_name:<20} {stats['total_relationships']:>15} {stats['avg_relationships_per_file']:>10.1f}")
    print()
    
    # Mention type statistics
    if 'mention_type_stats' in results:
        mention_stats = results['mention_type_stats']
        
        print("[MENTION TYPE STATISTICS]")
        print("-" * 40)
        print("\n  Per Annotator Mention Types:")
        print(f"  {'Annotator':<20} {'Table':<15} {'Mention Type':<15} {'Count':>10}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")
        
        for ann_name, stats in mention_stats['per_annotator_mention_types'].items():
            friendly_name = display_names[ann_name]
            # Medication table
            for mention_type, count in sorted(stats['medication'].items()):
                print(f"  {friendly_name:<20} {'Medication':<15} {mention_type:<15} {count:>10}")
            # Diagnosis table
            for mention_type, count in sorted(stats['diagnosis'].items()):
                print(f"  {friendly_name:<20} {'Diagnosis':<15} {mention_type:<15} {count:>10}")
        
        print("\n  Overall Mention Type Distribution:")
        print(f"  {'Table':<20} {'Mention Type':<15} {'Total Count':>15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15}")
        
        for mention_type, count in sorted(mention_stats['mention_type_per_table']['medication'].items()):
            print(f"  {'Medication Table':<20} {mention_type:<15} {count:>15}")
        
        for mention_type, count in sorted(mention_stats['mention_type_per_table']['diagnosis'].items()):
            print(f"  {'Diagnosis Table':<20} {mention_type:<15} {count:>15}")
        
        print()
    
    print("="*80)
    print("  END OF REPORT")
    print("="*80 + "\n")


def save_report(results: Dict, output_path: str):
    """Save detailed JSON report."""
    
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    serializable_results = convert_sets(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Detailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze annotation agreement across multiple annotators'
    )
    
    parser.add_argument('--input_dir', '-i', type=str,
                       default='Annotations/Individual',
                       help='Directory containing annotator folders or master JSON files (default: Annotations/Individual)')
    parser.add_argument('--output', '-o',
                       default='annotation_analysis_multi',
                       help='Output directory for results (default: annotation_analysis_multi)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip generating visualizations')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress console output')
    
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
        print("Error: Need at least 2 annotators for agreement analysis")
        return 1
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Generate analysis
    results = generate_multi_annotator_report(all_annotations, str(output_dir))
    
    # Print report
    if not args.quiet:
        print_multi_annotator_report(results)
    
    # Save JSON report
    save_report(results, str(output_dir / 'multi_annotator_analysis.json'))
    
    # Generate visualizations
    if not args.no_viz and HAS_MATPLOTLIB:
        if not args.quiet:
            print("\nGenerating visualizations...")
        create_multi_annotator_visualizations(results, str(output_dir))
    
    if not args.quiet:
        print(f"\n[SUCCESS] Analysis complete! Results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
