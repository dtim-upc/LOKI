# MIMIC-IV Train/Val/Test Split Reproduction

This directory contains the anonymous test ground truth file, `Annotated_Test.json`.

To reproduce the original MIMIC train/val/test split from the source MIMIC-IV data, please follow the instructions in the upstream LOKI pipeline repository:

https://github.com/dtim-upc/LOKI/tree/main/Datasets/MIMIC_Annotation_Pipeline

That folder contains the full re-producibility workflow and dataset-splitting instructions used by the project.

Due to the strict Data Use Agreement (DUA) and privacy restrictions protecting clinical health data, no raw MIMIC data can be redistributed or uploaded to this repository.

To reconstruct the LOKI evaluation pipeline, you must acquire the credentialed raw datasets independently.

### Required datasets

1. MIMIC-IV (v3.1)  
   Contains the structured relational tables (e.g., `diagnoses_icd.csv`, `prescriptions.csv`, `admissions.csv`)  
   [Download MIMIC-IV via PhysioNet](https://physionet.org/content/mimiciv/3.1/)

2. MIMIC-IV-Note (v2.2)  
   Contains the unstructured clinical text (`discharge.csv`)  
   [Download MIMIC-IV-Note via PhysioNet](https://physionet.org/content/mimic-iv-note/2.2/)

This workspace only includes the anonymized test annotations and does not contain the raw MIMIC-IV data or the full reconstructed dataset split.
