"""
dataset_registry.py

Central registry for all datasets used in the Unified Multi-Modal Retrieval Pipeline.
Maps dataset names to their physical paths and defines their underlying schema format.
"""

import os

# Base directory for all datasets (assumes this script is in SOTA_Evaluation_New)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(_BASE_DIR, "Datasets")

DATASETS = {
    "pharma": {
        "format": "protrix",
        "native_direction": "TABLE_TO_DOC",
        "description": "PubMed abstracts <-> DrugBank tables (82 shared tables)",
        "dir": os.path.join(DATASETS_DIR, "pharma"),
        "splits": {
            "train": "train_row_level.json",
            "val": "val_row_level.json",
            "test": "test_row_level.json"
        },
        "annotations": "Annotated_Test.json"
    },
    "protrix": {
        "format": "protrix",
        "native_direction": "TABLE_TO_DOC",
        "description": "Wikipedia text <-> Wikipedia tables",
        "dir": os.path.join(DATASETS_DIR, "protrix"),
        "splits": {
            "train": "train_row_level.json",
            "val": "val_row_level.json",
            "test": "test_row_level.json"
        },
        "annotations": "Annotated_Test.json"
    },
    "totto": {
        "format": "protrix",
        "native_direction": "TABLE_TO_DOC",
        "description": "Wikipedia text <-> Wikipedia tables (Large scale)",
        "dir": os.path.join(DATASETS_DIR, "totto"),
        "splits": {
            "train": "train_row_level.json",
            "val": "val_row_level.json",
            "test": "test_row_level.json"
        },
        "annotations": None
    },
    "mimic": {
        "format": "mimic",
        "native_direction": "TABLE_TO_DOC",
        "description": "Clinical notes <-> Patient tables (MIMIC-v2 format)",
        "dir": os.path.join(DATASETS_DIR, "mimic"),
        "splits": {
            "train": "train_row_level.json",
            "val": "val_row_level.json",
            "test": "test_row_level.json"
        },
        "annotations": "Annotated_Test.json"
    },
    "multihiertt": {
        "format": "protrix",
        "native_direction": "TABLE_TO_DOC",
        "description": "Financial text <-> Hierarchical tables",
        "dir": os.path.join(DATASETS_DIR, "multihiertt"),
        "splits": {
            "train": "train_row_level.json",
            "val": "val_row_level.json",
            "test": "test_row_level.json"
        },
        "annotations": "Annotated_Test.json"
    },
    "pharma_flipped_structured": {
        "format": "other",
        "native_direction": "DOC_TO_TABLE",
        "description": "PubMed abstracts (anchors) <-> DrugBank tables (DOC_TO_TABLE structured format)",
        "dir": os.path.join(DATASETS_DIR, "pharma_flipped_structured"),
        "splits": {
            "train": "train_row_level.json",
            "val": "val_row_level.json",
            "test": "test_row_level.json"
        },
        "annotations": None
    },
    "mimic_flipped": {
        "format": "mimic",
        "native_direction": "DOC_TO_TABLE",
        "description": "Clinical discharge notes (anchors) <-> Patient tables (MIMIC flipped DOC_TO_TABLE format)",
        "dir": os.path.join(DATASETS_DIR, "mimic_flipped"),
        "splits": {
            "train": "train_row_level.json",
            "val": "val_row_level.json",
            "test": "test_row_level.json"
        },
        "annotations": "Annotated_Test.json"
    }
}

def get_dataset_info(dataset_name: str) -> dict:
    """Returns the metadata dictionary for the given dataset."""
    name_lower = dataset_name.lower()
    if name_lower not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASETS.keys())}")
    return DATASETS[name_lower]

def get_split_path(dataset_name: str, split: str = "test") -> str:
    """Returns the absolute path to the JSON file for a specific split."""
    info = get_dataset_info(dataset_name)
    if split not in info["splits"]:
        raise ValueError(f"Split '{split}' not available for {dataset_name}.")
    
    file_name = info["splits"][split]
    path = os.path.join(info["dir"], file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file missing: {path}")
    return path
