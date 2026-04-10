"""
Generate COMBINED LLM annotation prompts for LOKI evaluation.

This version pairs diagnosis and medication examples from the same admission
so annotators can see both tables when identifying relationships.

Usage:
    python generate_combined_prompts.py --all --output_dir prompts_combined/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime


def load_prompt_template(template_path: str = "annotation_prompt.md") -> str:
    """Load the annotation prompt template."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_diagnosis_table(tables: Dict[str, Any]) -> str:
    """Format diagnosis table for the prompt."""
    if "diagnosis" not in tables:
        return "No diagnosis table available."
    
    diag = tables["diagnosis"]
    rows = diag.get("rows", [])
    
    if not rows:
        return "No diagnosis rows."
    
    lines = ["| row_idx | priority | icd | diagnosis |", "|---------|----------|-----|-----------|"]
    
    for row in rows:
        row_idx = row.get("row_idx", "?")
        content = row.get("content", [])
        
        # Extract key fields
        priority = content[2] if len(content) > 2 else ""
        icd = content[3] if len(content) > 3 else ""
        diagnosis = content[-1] if content else ""
        
        if len(diagnosis) > 60:
            diagnosis = diagnosis[:57] + "..."
        
        lines.append(f"| {row_idx} | {priority} | {icd} | {diagnosis} |")
    
    return "\n".join(lines)


def format_medication_table(tables: Dict[str, Any]) -> str:
    """Format medication table for the prompt."""
    if "medication" not in tables:
        return "No medication table available."
    
    med = tables["medication"]
    rows = med.get("rows", [])
    
    if not rows:
        return "No medication rows."
    
    lines = ["| row_idx | drug | dosage | route |", "|---------|------|--------|-------|"]
    
    for row in rows:
        row_idx = row.get("row_idx", "?")
        content = row.get("content", [])
        
        drug = content[2] if len(content) > 2 else ""
        dosage = content[4] if len(content) > 4 else ""
        unit = content[5] if len(content) > 5 else ""
        route = content[-1] if content else ""
        
        if len(drug) > 40:
            drug = drug[:37] + "..."
        
        dosage_str = f"{dosage} {unit}".strip()
        
        lines.append(f"| {row_idx} | {drug} | {dosage_str} | {route} |")
    
    return "\n".join(lines)


def format_sections(sections: List[Dict]) -> str:
    """Format section summary."""
    if not sections:
        return "No sections detected."
    
    lines = ["| Section | Type | Sentences |", "|---------|------|-----------|"]
    
    for sec in sections:
        name = sec.get("section_name", "Unknown")
        sec_type = sec.get("section_type", "unknown")
        start = sec.get("start_sentence_idx", 0)
        end = sec.get("end_sentence_idx", 0)
        
        lines.append(f"| {name} | {sec_type} | {start}-{end} |")
    
    return "\n".join(lines)


def format_sentences(sentences: Dict[str, Dict], max_sentences: int = 200) -> str:
    """Format sentences with indices for the prompt."""
    lines = ["```json", "{"]
    
    sorted_indices = sorted(sentences.keys(), key=lambda x: int(x))
    
    if len(sorted_indices) > max_sentences:
        sorted_indices = sorted_indices[:max_sentences]
        truncated = True
    else:
        truncated = False
    
    for i, idx in enumerate(sorted_indices):
        sent_data = sentences[idx]
        text = sent_data.get("text", "")
        section = sent_data.get("section_name", "Unknown")
        
        text = text.replace('"', '\\"').replace('\n', ' ').replace('\r', '')
        
        if len(text) > 500:
            text = text[:497] + "..."
        
        comma = "," if i < len(sorted_indices) - 1 else ""
        lines.append(f'  "{idx}": {{"text": "{text}", "section": "{section}"}}{comma}')
    
    lines.append("}")
    lines.append("```")
    
    if truncated:
        lines.append(f"\n*Note: Showing first {max_sentences} sentences. Total: {len(sentences)}*")
    
    return "\n".join(lines)


def generate_combined_prompt(
    diagnosis_example: Dict[str, Any],
    medication_example: Dict[str, Any],
    template: str
) -> str:
    """Generate a combined annotation prompt with both tables."""
    
    patient_id = diagnosis_example.get("patient_id", "")
    admission_id = diagnosis_example.get("admission_id", "")
    
    diag_anchor_id = diagnosis_example.get("anchor_id", 0)
    med_anchor_id = medication_example.get("anchor_id", 0)
    
    # Get tables from each
    diag_tables = diagnosis_example.get("tables", {})
    med_tables = medication_example.get("tables", {})
    
    # Get document (should be same for both)
    primary_doc = diagnosis_example.get("primary_positive", {})
    sentences = primary_doc.get("sentences", {})
    sections = primary_doc.get("sections", [])
    
    # Count rows
    diag_rows = diag_tables.get("diagnosis", {}).get("rows", [])
    med_rows = med_tables.get("medication", {}).get("rows", [])
    
    data_section = f"""
---

## Data for Annotation

**Patient ID**: `{patient_id}`  
**Admission ID**: `{admission_id}`  
**Diagnosis Anchor ID**: `{diag_anchor_id}`  
**Medication Anchor ID**: `{med_anchor_id}`

### DIAGNOSIS TABLE ({len(diag_rows)} rows)
{format_diagnosis_table(diag_tables)}

### MEDICATION TABLE ({len(med_rows)} rows)
{format_medication_table(med_tables)}

### Document Sections Overview
{format_sections(sections)}

### CLINICAL NOTES (Indexed Sentences)
{format_sentences(sentences)}

---

## Your Annotation

For this admission, identify relationships between drugs (MEDICATION TABLE) and diagnoses (DIAGNOSIS TABLE).

**Important**: Use `drug_row` to reference rows from the MEDICATION TABLE and `diagnosis_row` to reference rows from the DIAGNOSIS TABLE.

```json
{{
  "patient_id": "{patient_id}",
  "admission_id": "{admission_id}",
  "diagnosis_anchor_id": {diag_anchor_id},
  "medication_anchor_id": {med_anchor_id},
  
  "row_grounding": {{
    "diagnosis": {{}},
    "medication": {{}}
  }},
  
  "relationships": [
    {{
      "id": "rel_001",
      "drug_row": 1,
      "diagnosis_row": 2,
      "relationship_type": "TREATS",
      "evidence_sentences": [],
      "evidence_scope": "section",
      "reasoning": "",
      "confidence": "high"
    }}
  ],
  
  "multi_relationship_flags": [],
  "negative_relationships": [],
  "quality_notes": null
}}
```
"""
    
    # Replace placeholder in template
    full_prompt = template.replace(
        "[DATA WILL BE INSERTED HERE BY THE ANNOTATION SCRIPT]",
        data_section
    )
    
    return full_prompt


def load_test_data(data_path: str = "mimic_data/test_row_level_v2.json") -> List[Dict]:
    """Load the v2 test data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_by_admission(data: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Group examples by admission and type."""
    grouped = defaultdict(dict)
    
    for example in data:
        admission_id = example.get("admission_id", "")
        metadata = example.get("anchor_metadata", "")
        
        if "diagnosis" in metadata:
            grouped[admission_id]["diagnosis"] = example
        elif "medication" in metadata:
            grouped[admission_id]["medication"] = example
    
    return grouped


def save_prompt(prompt: str, output_path: str) -> None:
    """Save prompt to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print(f"[OK] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate COMBINED annotation prompts (diagnosis + medication per admission)"
    )
    parser.add_argument(
        "--admission_id", type=str, default=None,
        help="Generate prompt for specific admission"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate prompts for all admissions (default)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="prompts_combined",
        help="Output directory for prompts"
    )
    parser.add_argument(
        "--data_path", type=str, default="mimic_data/test_row_level_v2.json",
        help="Path to v2 test data"
    )
    parser.add_argument(
        "--template_path", type=str, default="annotation_prompt.md",
        help="Path to prompt template"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LOKI Combined Annotation Prompt Generator")
    print("=" * 60)
    
    # Load data
    print(f"\n[LOAD] Loading data from: {args.data_path}")
    data = load_test_data(args.data_path)
    print(f"       Found {len(data)} examples")
    
    # Group by admission
    grouped = group_by_admission(data)
    print(f"       Grouped into {len(grouped)} admissions")
    
    # Load template
    print(f"[LOAD] Loading template from: {args.template_path}")
    template = load_prompt_template(args.template_path)
    
    if args.admission_id:
        if args.admission_id in grouped:
            examples = grouped[args.admission_id]
            if "diagnosis" in examples and "medication" in examples:
                patient_id = examples["diagnosis"].get("patient_id", "")
                
                prompt = generate_combined_prompt(
                    examples["diagnosis"],
                    examples["medication"],
                    template
                )
                
                filename = f"prompt_combined_{patient_id}_{args.admission_id}.md"
                output_path = Path(args.output_dir) / filename
                save_prompt(prompt, output_path)
            else:
                print(f"[ERROR] Incomplete admission {args.admission_id}")
        else:
            print(f"[ERROR] Admission not found: {args.admission_id}")
            
    else:
        # Default to ALL
        print(f"\n[GENERATE] Generating combined prompts (ALL)...")
        
        generated = 0
        for admission_id, examples in grouped.items():
            if "diagnosis" in examples and "medication" in examples:
                patient_id = examples["diagnosis"].get("patient_id", "")
                
                prompt = generate_combined_prompt(
                    examples["diagnosis"],
                    examples["medication"],
                    template
                )
                
                filename = f"prompt_combined_{patient_id}_{admission_id}.md"
                output_path = Path(args.output_dir) / filename
                save_prompt(prompt, output_path)
                generated += 1
            else:
                print(f"[WARNING] Incomplete admission {admission_id}")
        
        print(f"\n[DONE] Generated {generated} combined prompts")


if __name__ == "__main__":
    main()
