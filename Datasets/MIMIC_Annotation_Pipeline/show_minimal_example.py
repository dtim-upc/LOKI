#!/usr/bin/env python3
"""Show details of the minimal example."""

import json
from pathlib import Path

# Load merged annotations
file_path = Path("Annotations/Voting/merged_annotations_all.json")
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

file_id = '28979390'
ann = data[file_id]

print(f"=== MINIMAL EXAMPLE DETAILS: {file_id} ===\n")
print(f"Patient ID: {ann['patient_id']}")
print(f"Admission ID: {ann['admission_id']}")

print(f"\n--- DIAGNOSIS ROWS (1 total) ---")
for row_id, row_data in ann['row_grounding']['diagnosis'].items():
    sentences = row_data['sentences']
    mention_types = row_data['mention_types']
    print(f"  Row {row_id}: Sentences {sentences}, Mention types: {mention_types}")

print(f"\n--- MEDICATION ROWS (6 total) ---")
for row_id, row_data in ann['row_grounding']['medication'].items():
    sentences = row_data['sentences']
    mention_types = row_data['mention_types']
    print(f"  Row {row_id}: Sentences {sentences}, Mention types: {mention_types}")

print(f"\n--- RELATIONSHIPS (6 total) ---")
for i, rel in enumerate(ann['relationships'], 1):
    print(f"  {i}. Drug {rel['drug_row']} -> Diagnosis {rel['diagnosis_row']}")
    print(f"     Type: {rel['relationship_type']}")
    print(f"     Evidence: Sentences {rel['evidence_sentences']}")
    print(f"     Scope: {rel['evidence_scope']}, Confidence: {rel['confidence']}")
    print(f"     Agreement: {rel['_provenance']['agreement_level']} ({rel['_provenance']['vote_count']}/3 annotators)")
    print()

# Collect all unique sentences
all_sentences = set()
for row_data in ann['row_grounding']['diagnosis'].values():
    all_sentences.update(row_data['sentences'])
for row_data in ann['row_grounding']['medication'].values():
    all_sentences.update(row_data['sentences'])

print(f"--- UNIQUE SENTENCES (10 total) ---")
print(f"  Sentence IDs: {sorted(all_sentences)}")

print(f"\n=== SUMMARY ===")
print(f"This is the SIMPLEST example in your test set with:")
print(f"  - Only 1 diagnosis row")
print(f"  - Only 6 medication rows")
print(f"  - Only 10 unique sentences")
print(f"  - 6 drug-diagnosis relationships")
print(f"  - Perfect for visualizing attention matrices!")
