# LOKI Clinical Relationship Annotation Task

## Overview
You are annotating clinical data to create ground truth for evaluating LOKI, a system that discovers semantic join paths between structured tables (medications, diagnoses) and unstructured clinical notes.

## Your Task
Given:
1. A **DIAGNOSIS TABLE** with patient diagnoses
2. A **MEDICATION TABLE** with patient medications  
3. **CLINICAL NOTES** (indexed sentences from a discharge summary)

You must identify:
1. **Row Grounding**: Which sentences mention each row from the tables
2. **Relationships**: Drug-Diagnosis pairs with their relationship type and evidence

---

## Indexing Conventions (CRITICAL)
- **Row indices are 1-based**: Row 1, Row 2, Row 3, ...
- **Sentence indices are 0-based**: Sentence 0, Sentence 1, Sentence 2, ...
- Always use these exact indices in your output

---

## Relationship Types

You must classify each drug-diagnosis relationship into **exactly one** of these types:

### 1. TREATS (Most Common)
**Definition**: Drug is prescribed to treat, manage, or prevent the diagnosis/condition.

**Examples**:
- "Lisinopril was started for hypertension management"
- "Continue metformin for diabetes"
- "Aspirin 81mg daily for CAD prophylaxis"

**Evidence patterns**: "started for", "prescribed for", "for treatment of", "to manage", drug listed under diagnosis section header

---

### 2. ADVERSE_EFFECT (Safety Critical)
**Definition**: The diagnosis/symptom was caused BY the drug as a side effect.

**Examples**:
- "Patient developed acute kidney injury secondary to NSAIDs"
- "Cough likely due to ACE inhibitor"
- "Nausea after starting chemotherapy"

**Evidence patterns**: "secondary to", "due to", "caused by", "after starting", "side effect of", temporal association with drug initiation

---

### 3. CONTRAINDICATED (Negative Relationship)
**Definition**: Drug should NOT be given because of the diagnosis. This is a prohibition.

**Examples**:
- "Avoid NSAIDs given history of GI bleeding"
- "Metformin contraindicated with renal failure"
- "Hold anticoagulation due to active hemorrhage"

**Evidence patterns**: "avoid", "contraindicated", "do not give", "hold", "should not receive"

---

### 4. DISCONTINUED (Status Change)
**Definition**: Drug was stopped/held because of the diagnosis or clinical situation.

**Examples**:
- "Metformin was held due to acute kidney injury"
- "Patient self-discontinued lasix"
- "Warfarin stopped given upcoming surgery"

**Evidence patterns**: "held", "discontinued", "stopped", "held due to", "self-discontinued"

**Note**: A drug can have BOTH a TREATS and DISCONTINUED relationship if it was prescribed for a condition but later stopped.

---

## Evidence Scope

For each relationship, identify the **evidence scope**:

- **`explicit`**: A clear sentence directly states the relationship
  - "Lisinopril started for hypertension" → explicit TREATS
  
- **`section`**: Drug and diagnosis appear in the same clinical section (no explicit sentence)
  - Drug appears under "# Ascites" section header → section-level TREATS
  
- **`document`**: Drug and diagnosis are mentioned in different parts of the document but are related
  - Diagnosis in Past Medical History, drug in Medications → document-level inference

---

## Output Format

Provide your annotation as a JSON object with this structure:

```json
{
  "row_grounding": {
    "diagnosis": {
      "1": {
        "sentences": [3, 15, 42],
        "mention_types": ["explicit", "explicit", "context"]
      },
      "2": {
        "sentences": [8],
        "mention_types": ["explicit"]
      }
    },
    "medication": {
      "1": {
        "sentences": [25, 48, 52],
        "mention_types": ["explicit", "brand_name", "explicit"]
      }
    }
  },
  
  "relationships": [
    {
      "id": "rel_001",
      "drug_row": 1,
      "diagnosis_row": 2,
      "relationship_type": "TREATS",
      "evidence_sentences": [45, 48],
      "evidence_scope": "section",
      "reasoning": "Brief explanation of why this relationship exists",
      "confidence": "high"
    },
    {
      "id": "rel_002",
      "drug_row": 1,
      "diagnosis_row": 2,
      "relationship_type": "DISCONTINUED",
      "evidence_sentences": [10],
      "evidence_scope": "explicit",
      "reasoning": "Patient self-discontinued the medication",
      "confidence": "high",
      "temporal_note": "prior_to_admission"
    }
  ],
  
  "multi_relationship_flags": [
    {
      "drug_row": 1,
      "diagnosis_row": 2,
      "relationship_types": ["TREATS", "DISCONTINUED"],
      "note": "Same pair has multiple relationship types"
    }
  ],
  
  "negative_relationships": [
    {
      "drug_row": 5,
      "diagnosis_row": 1,
      "reason": "No clinical relationship (flu vaccine unrelated to hypertension)"
    }
  ],
  
  "quality_notes": "Any observations about data quality or difficult cases"
}
```

---

## Mention Types for Row Grounding

When grounding rows to sentences, classify the mention type:

| Type | Description | Example |
|------|-------------|---------|
| `explicit` | Entity mentioned by exact name | "Furosemide 40mg" |
| `brand_name` | Drug mentioned by brand name | "Lasix" (= Furosemide) |
| `abbreviated` | Abbreviated mention | "HTN" (= Hypertension) |
| `context` | Implied/contextual mention | "diuretics" when discussing Furosemide |
| `synonym` | Medical synonym | "kidney cancer" (= renal cell carcinoma) |

---

## Guidelines

### DO:
- ✅ Include ALL relationships you can identify, even with low confidence
- ✅ Flag multi-relationship cases (same drug-diagnosis with multiple relationship types)
- ✅ Use the exact sentence indices from the data
- ✅ Provide reasoning for each relationship
- ✅ Note temporal sequences when relevant (e.g., "prior to admission", "during stay")

### DON'T:
- ❌ Invent relationships not supported by the text
- ❌ Guess drug names or diagnoses not in the tables
- ❌ Skip a relationship because it seems "obvious"
- ❌ Change the row or sentence indices

---

## Common Drug-Diagnosis Relationships (Reference)

| Drug Class | Typical TREATS | Possible Adverse Effects |
|------------|---------------|-------------------------|
| ACE Inhibitors (Lisinopril) | Hypertension, CHF | Cough, Angioedema, AKI |
| Beta Blockers (Metoprolol) | HTN, Afib, CHF | Bradycardia, Fatigue |
| Diuretics (Furosemide, Spironolactone) | Edema, Ascites, CHF | Electrolyte abnormalities, AKI |
| Opioids (Morphine, Oxycodone) | Pain | Constipation, Respiratory depression |
| Anticoagulants (Warfarin, Heparin) | DVT, PE, Afib | Bleeding |
| Insulin | Diabetes | Hypoglycemia |
| Statins (Atorvastatin) | Hyperlipidemia | Myopathy, Liver enzyme elevation |
| PPIs (Omeprazole) | GERD, Ulcers | C. diff, Hypomagnesemia |
| Antibiotics | Infections | Allergic reactions, C. diff |

---

## Example Annotation

### Input Data

**DIAGNOSIS TABLE**:
| row_idx | diagnosis |
|---------|-----------|
| 1 | Portal hypertension |
| 2 | Ascites |
| 3 | Cirrhosis |

**MEDICATION TABLE**:
| row_idx | drug |
|---------|------|
| 1 | Furosemide |
| 2 | Spironolactone |

**CLINICAL NOTES** (selected sentences):
```
Sentence 10: "Pt reports self-discontinuing lasix and spironolactone weeks ago, because she feels like they don't do anything."
Sentence 45: "# Ascites - p/w worsening abd distension and discomfort for last week."
Sentence 48: "diuretics: > Furosemide 40 mg PO DAILY > Spironolactone 50 mg PO DAILY"
```

### Output Annotation

```json
{
  "row_grounding": {
    "diagnosis": {
      "2": {"sentences": [45], "mention_types": ["explicit"]}
    },
    "medication": {
      "1": {"sentences": [10, 48], "mention_types": ["brand_name", "explicit"]},
      "2": {"sentences": [10, 48], "mention_types": ["explicit", "explicit"]}
    }
  },
  
  "relationships": [
    {
      "id": "rel_001",
      "drug_row": 1,
      "diagnosis_row": 2,
      "relationship_type": "TREATS",
      "evidence_sentences": [45, 48],
      "evidence_scope": "section",
      "reasoning": "Furosemide listed under diuretics in Ascites treatment section",
      "confidence": "high"
    },
    {
      "id": "rel_002",
      "drug_row": 1,
      "diagnosis_row": 2,
      "relationship_type": "DISCONTINUED",
      "evidence_sentences": [10],
      "evidence_scope": "explicit",
      "reasoning": "Patient self-discontinued lasix (Furosemide) weeks ago",
      "confidence": "high",
      "temporal_note": "prior_to_admission, now restarted"
    },
    {
      "id": "rel_003",
      "drug_row": 2,
      "diagnosis_row": 2,
      "relationship_type": "TREATS",
      "evidence_sentences": [45, 48],
      "evidence_scope": "section",
      "reasoning": "Spironolactone listed under diuretics in Ascites treatment section",
      "confidence": "high"
    },
    {
      "id": "rel_004",
      "drug_row": 2,
      "diagnosis_row": 2,
      "relationship_type": "DISCONTINUED",
      "evidence_sentences": [10],
      "evidence_scope": "explicit",
      "reasoning": "Patient self-discontinued spironolactone weeks ago",
      "confidence": "high",
      "temporal_note": "prior_to_admission, now restarted"
    }
  ],
  
  "multi_relationship_flags": [
    {
      "drug_row": 1,
      "diagnosis_row": 2,
      "relationship_types": ["TREATS", "DISCONTINUED"],
      "note": "Furosemide both treats ascites and was previously discontinued"
    },
    {
      "drug_row": 2,
      "diagnosis_row": 2,
      "relationship_types": ["TREATS", "DISCONTINUED"],
      "note": "Spironolactone both treats ascites and was previously discontinued"
    }
  ],
  
  "quality_notes": "Clear case with good evidence. Patient non-compliance led to worsening ascites."
}
```

---

## Now Annotate This Example

[DATA WILL BE INSERTED HERE BY THE ANNOTATION SCRIPT]
