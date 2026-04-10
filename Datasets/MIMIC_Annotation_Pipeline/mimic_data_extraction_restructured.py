"""
MIMIC Data Extraction - Restructured by Hospital Admission ID

This script reorganizes MIMIC data by hospital admission (hadm_id) instead of just by patient (subject_id).
Each patient directory will contain subdirectories for each hospital admission with separate files.

New Structure:
    mimic_restructured/
    └── <subject_id>/
        ├── <hadm_id_1>/
        │   ├── <hadm_id_1>-diagnosis.csv
        │   ├── <hadm_id_1>-medication.csv
        │   └── <hadm_id_1>-notes.txt
        └── <hadm_id_2>/
            └── ...
"""

import numpy as np
import pandas as pd
import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "./inputs"
OUTPUT_DIR = "./mimic_admissions"
N_PATIENTS_TO_EXPORT = 10000

# ==========================================
# PHASE 1: FIND VALID PATIENTS (MINIMAL DATA)
# ==========================================
print("PHASE 1: Finding valid patients (minimal columns only)...")
print("=" * 60)

SUBSET_MULTIPLIER = 2
chunk_size = 100000  # Larger chunks for faster reading

# Track patient validity using dictionaries for O(1) lookup
diag_hadm_dict = {}  # {subject_id: set of hadm_ids}
notes_hadm_dict = {}  # {subject_id: set of hadm_ids}

print(f"Target: Find {N_PATIENTS_TO_EXPORT} valid patients")
print(f"Reading subject_id and hadm_id columns only...\n")

# Read diagnoses and discharge with ONLY needed columns
diag_reader = pd.read_csv(
    f"{DATA_DIR}/diagnoses_icd.csv", 
    usecols=["subject_id", "hadm_id"],
    chunksize=chunk_size
)
discharge_reader = pd.read_csv(
    f"{DATA_DIR}/discharge.csv",
    usecols=["subject_id", "hadm_id"],
    chunksize=chunk_size
)

chunk_num = 0
valid_patients = []

for diag_chunk, discharge_chunk in zip(diag_reader, discharge_reader):
    chunk_num += 1
    
    # Vectorized update for diagnosis hadm_ids using groupby
    for sid, group in diag_chunk.groupby("subject_id"):
        if sid not in diag_hadm_dict:
            diag_hadm_dict[sid] = set()
        diag_hadm_dict[sid].update(group["hadm_id"].dropna().values)
    
    # Vectorized update for discharge hadm_ids using groupby
    for sid, group in discharge_chunk.groupby("subject_id"):
        if sid not in notes_hadm_dict:
            notes_hadm_dict[sid] = set()
        notes_hadm_dict[sid].update(group["hadm_id"].dropna().values)
    
    # Check for valid patients - all subjects that have diagnosis records
    for sid in diag_hadm_dict:
        if sid not in valid_patients:
            diag_ids = diag_hadm_dict.get(sid, set())
            note_ids = notes_hadm_dict.get(sid, set())
            if len(diag_ids) > 0 and diag_ids.issubset(note_ids):
                valid_patients.append(sid)
    
    if len(valid_patients) >= N_PATIENTS_TO_EXPORT:
        print(f"  ✓ Found {len(valid_patients)} valid patients after {chunk_num} chunks!")
        break
    
    if chunk_num % 3 == 0:
        print(f"  Chunk {chunk_num}: {len(diag_hadm_dict)} patients seen, {len(valid_patients)} valid")

print(f"\n✓ Phase 1 complete: Found {len(valid_patients)} valid patients")

if len(valid_patients) < N_PATIENTS_TO_EXPORT:
    print(f"Warning: Only found {len(valid_patients)} valid patients")
    N_PATIENTS_TO_EXPORT = len(valid_patients)

selected_subjects = valid_patients[:N_PATIENTS_TO_EXPORT]
print(f"✓ Selected {len(selected_subjects)} patients")
print(f"Sample: {selected_subjects[:5]}")

# ==========================================
# PHASE 2: LOAD FULL DATA FOR SELECTED PATIENTS
# ==========================================
print("\n" + "=" * 60)
print("PHASE 2: Loading full data for selected patients...")
print("=" * 60)

selected_set = set(selected_subjects)

# Load reference tables
print("Loading reference tables...")
patients = pd.read_csv(f"{DATA_DIR}/patients.csv")
d_icd_diagnoses = pd.read_csv(f"{DATA_DIR}/d_icd_diagnoses.csv")

# Load full data for selected patients using chunked reading
print(f"\nLoading discharge.csv for {len(selected_subjects)} patients...")
discharge_chunks = []
for chunk in pd.read_csv(f"{DATA_DIR}/discharge.csv", chunksize=50000):
    filtered = chunk[chunk["subject_id"].isin(selected_set)]
    if len(filtered) > 0:
        discharge_chunks.append(filtered)
discharge = pd.concat(discharge_chunks, ignore_index=True)
print(f"  ✓ {discharge.shape[0]} notes")

print(f"Loading diagnoses_icd.csv for {len(selected_subjects)} patients...")
diagnoses_chunks = []
for chunk in pd.read_csv(f"{DATA_DIR}/diagnoses_icd.csv", chunksize=50000):
    filtered = chunk[chunk["subject_id"].isin(selected_set)]
    if len(filtered) > 0:
        diagnoses_chunks.append(filtered)
diagnoses_table = pd.concat(diagnoses_chunks, ignore_index=True)
print(f"  ✓ {diagnoses_table.shape[0]} records")

print(f"Loading prescriptions.csv for {len(selected_subjects)} patients...")
medication_chunks = []
for chunk in pd.read_csv(f"{DATA_DIR}/prescriptions.csv", chunksize=100000, low_memory=False):
    filtered = chunk[chunk["subject_id"].isin(selected_set)]
    if len(filtered) > 0:
        medication_chunks.append(filtered)
medication_table = pd.concat(medication_chunks, ignore_index=True) if medication_chunks else pd.DataFrame()
print(f"  ✓ {medication_table.shape[0]} records")

# Merge reference data
print("\nEnriching diagnosis table...")
diagnoses_table = diagnoses_table.merge(patients, on="subject_id", how="left")
diagnoses_table = diagnoses_table.merge(
    d_icd_diagnoses,
    on=["icd_code", "icd_version"],
    how="left"
)

print(f"\n✓ Data loading complete!")
print(f"  diagnoses_table: {diagnoses_table.shape}")
print(f"  medication_table: {medication_table.shape}")
print(f"  discharge: {discharge.shape}")

# ==========================================
# FUNCTION: FIX ENCODING ISSUES
# ==========================================
def fix_mojibake_text(text):
    """
    Repairs text containing Windows-1252 characters (like \\x95 bullets)
    that were incorrectly read as Latin-1.
    """
    if not isinstance(text, str):
        return str(text)

    chars = []
    for char in text:
        # Check for characters in the specific "Gremlins" range (0x80 - 0x9F)
        if '\x80' <= char <= '\x9f':
            try:
                # Attempt to fix the character (e.g., turn \x95 back into •)
                fixed_char = char.encode('latin1').decode('cp1252')
                chars.append(fixed_char)
            except:
                chars.append(char)
        else:
            chars.append(char)

    return "".join(chars)

# ==========================================
# EXPORT DATA BY HOSPITAL ADMISSION
# ==========================================
# If OUTPUT_DIR exists, rename it to avoid conflicts
if os.path.exists(OUTPUT_DIR):
    import time
    backup_dir = f"{OUTPUT_DIR}_backup_{int(time.time())}"
    print(f"\nWarning: Existing folder found: {OUTPUT_DIR}")
    print(f"Renaming to: {backup_dir}")
    os.rename(OUTPUT_DIR, backup_dir)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nStarting export to {OUTPUT_DIR}...")
print("=" * 60)

for idx, sid in enumerate(selected_subjects, 1):
    # Create patient folder
    patient_folder = os.path.join(OUTPUT_DIR, str(sid))
    os.makedirs(patient_folder, exist_ok=True)

    # Get all unique hadm_ids for this patient
    hadm_ids = diagnoses_table[diagnoses_table["subject_id"] == sid]["hadm_id"].unique()
    
    print(f"[{idx}/{len(selected_subjects)}] Patient {sid}: {len(hadm_ids)} admission(s)")

    # Process each hospital admission separately
    for hadm_id in hadm_ids:
        # Create hadm_id subfolder
        hadm_folder = os.path.join(patient_folder, str(hadm_id))
        os.makedirs(hadm_folder, exist_ok=True)

        # -------- Diagnosis CSV (singular) --------
        diag_df = diagnoses_table[
            (diagnoses_table["subject_id"] == sid) & 
            (diagnoses_table["hadm_id"] == hadm_id)
        ].copy()
        
        # Remove specified columns from diagnosis table
        cols_to_remove_diag = ['anchor_year', 'anchor_year_group', 'dod']
        diag_df = diag_df.drop(columns=[col for col in cols_to_remove_diag if col in diag_df.columns])
        
        # Rename columns to meaningful clinical names
        diag_df = diag_df.rename(columns={
            'seq_num': 'priority',
            'icd_code': 'icd',
            'icd_version': 'icd_ver',
            'gender': 'sex',
            'anchor_age': 'age',
            'long_title': 'diagnosis'
        })
        
        diag_path = os.path.join(hadm_folder, f"{hadm_id}-diagnosis.csv")
        diag_df.to_csv(diag_path, index=False, encoding="utf-8-sig")

        # -------- Medication CSV --------
        med_df = medication_table[
            (medication_table["subject_id"] == sid) & 
            (medication_table["hadm_id"] == hadm_id)
        ].copy()
        
        # Remove specified columns from medication table
        cols_to_remove_med = ['pharmacy_id', 'poe_id', 'poe_seq', 'order_provider_id', 
                              'starttime', 'stoptime', 'drug_type', 'formulary_drug_cd', 
                              'gsn', 'ndc', 'form_rx', 'form_unit_disp']
        med_df = med_df.drop(columns=[col for col in cols_to_remove_med if col in med_df.columns])
        
        # Rename columns to meaningful clinical names
        med_df = med_df.rename(columns={
            'prod_strength': 'contains',
            'dose_val_rx': 'dosage',
            'dose_unit_rx': 'unit',
            'form_val_disp': 'quantity',
            'doses_per_24_hrs': 'per_day'
        })
        
        med_path = os.path.join(hadm_folder, f"{hadm_id}-medication.csv")
        med_df.to_csv(med_path, index=False, encoding="utf-8-sig")

        # -------- Notes (combined into single file) --------
        notes_df = discharge[
            (discharge["subject_id"] == sid) & 
            (discharge["hadm_id"] == hadm_id)
        ].copy()

        # Drop duplicates in case same note appears more than once
        if "note_id" in notes_df.columns:
            notes_df = notes_df.drop_duplicates(subset=["note_id"])

        # Combine all notes for this admission into a single file
        combined_notes = []
        for _, row in notes_df.iterrows():
            note_text = row.get("text", "")
            if note_text:
                # Clean the text using mojibake fix function
                cleaned_text = fix_mojibake_text(note_text)
                
                # Remove ___ placeholders (hidden information)
                cleaned_text = cleaned_text.replace("___", "")
                combined_notes.append(cleaned_text)

        # Save combined notes to single file
        if combined_notes:
            notes_path = os.path.join(hadm_folder, f"{hadm_id}-notes.txt")
            # Separate multiple notes with a clear delimiter
            delimiter = "\n\n" + "=" * 80 + "\n\n"
            final_notes_text = delimiter.join(combined_notes)
            with open(notes_path, "w", encoding="utf-8-sig") as f:
                f.write(final_notes_text)

print("=" * 60)
print("\nExport complete!")
print(f"Output folder: {OUTPUT_DIR}")
print(f"\nTo create a zip file, run:")
print(f'  shutil.make_archive("{OUTPUT_DIR}", "zip", "{OUTPUT_DIR}")')
