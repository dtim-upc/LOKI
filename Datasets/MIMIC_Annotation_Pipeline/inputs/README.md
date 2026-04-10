# Inputs Directory

This directory is intended to contain the raw `.csv` table dumps directly from the PhysioNet MIMIC-IV and MIMIC-IV-Note databases. 

Due to the strict Data Use Agreement (DUA) and privacy restrictions protecting clinical health data, **no raw MIMIC data can be redistributed or uploaded to this repository.**

To reconstruct the LOKI evaluation pipeline, you must acquire the credentialed raw datasets independently:

### Required Datasets

1. **MIMIC-IV (v3.1)**  
   *Contains the structured relational tables (e.g., `diagnoses_icd.csv`, `prescriptions.csv`, `admissions.csv`)*  
   [Download MIMIC-IV via PhysioNet](https://physionet.org/content/mimiciv/3.1/)

2. **MIMIC-IV-Note (v2.2)**  
   *Contains the unstructured clinical text (`discharge.csv`)*  
   [Download MIMIC-IV-Note via PhysioNet](https://physionet.org/content/mimic-iv-note/2.2/)

### Instructions

Once you have downloaded the datasets, extract **only** the following 5 target `.csv` files and place them directly here in the `inputs/` folder:

From **MIMIC-IV**:
- `diagnoses_icd.csv`
- `d_icd_diagnoses.csv`
- `patients.csv`
- `prescriptions.csv`

From **MIMIC-IV-Note**:
- `discharge.csv`

Ensure these files are present before running `Stage 0: mimic_data_extraction_restructured.py`.
