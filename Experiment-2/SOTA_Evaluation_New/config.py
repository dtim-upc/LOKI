"""
config.py — Central configuration for SOTA Evaluation.

All paths are relative to the SOTA_Evaluation/ folder.
Adjust these if your directory layout differs.
"""

import os

# ===========================================================================
# Base paths
# ===========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# Dataset Paths
# ===========================================================================

# Default test file
TEST_DATA_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "Datasets", "pharma_flipped_structured", "test_row_level.json"))

# Subsampling: 0 = use full test set, >0 = subsample to that many examples
# Both CMDL and LOKI will use the SAME subsampled examples.
MAX_TEST_EXAMPLES = 0
MAX_QUERIES = 0
SEED = 42

# ===========================================================================
# CMDL Configuration
# ===========================================================================
CMDL_MODEL_DIR = os.path.join(BASE_DIR, "models", "CMDL")
CMDL_TEXT_ENET_PATH = os.path.join(CMDL_MODEL_DIR, "text_enet_best.pt")
CMDL_COL_ENET_PATH  = os.path.join(CMDL_MODEL_DIR, "col_enet_best.pt")

# EncoderNet architecture (must match training)
CMDL_HIDDEN_SIZE = 200
CMDL_OUTPUT_SIZE = 100

# WEM (FastText) model for CMDL feature building (gensim format)
WEM_MODEL_PATH = os.path.join(
    CMDL_MODEL_DIR, "resources", "fasttext", "cc", "cc.en.300.gensim"
)
WEM_DIM = 300

# ===========================================================================
# LOKI Configuration - Bundled Runtime and Dynamic Model Discovery
# ===========================================================================
LOKI_MODEL_DIR = os.path.join(BASE_DIR, "models", "LOKI")

# Fallback mapping for config
LOKI_ARGS_PATH = os.path.join(LOKI_MODEL_DIR, "args.json")
if not os.path.exists(LOKI_ARGS_PATH):
    LOKI_ARGS_PATH = os.path.join(LOKI_MODEL_DIR, "training_config.json")

import glob
LOKI_MODELS = {}

# Recursively find standard checkpoint files in the LOKI directory.
_loki_checkpoint_patterns = ("model.pt", "*_best.pt")
_loki_checkpoint_paths = []
for _pattern in _loki_checkpoint_patterns:
    _loki_checkpoint_paths.extend(
        glob.glob(os.path.join(LOKI_MODEL_DIR, "**", _pattern), recursive=True)
    )

for pt_path in sorted(set(_loki_checkpoint_paths)):
    parent_dir = os.path.basename(os.path.dirname(pt_path))
    checkpoint_name = os.path.basename(pt_path)

    # Map the checkpoint folder name to canonical CLI keys
    if "best_model_epoch" in parent_dir or checkpoint_name.endswith("_best.pt"):
        LOKI_MODELS["best_model"] = pt_path
    elif "best_test_avg_precision" in parent_dir or "best_test_ap" in parent_dir:
        LOKI_MODELS["best_test_ap"] = pt_path
    elif "best_test_f1" in parent_dir or "best_test_overall_acc" in parent_dir or "best_test_acc" in parent_dir:
        LOKI_MODELS["best_test_acc"] = pt_path
    else:
        # Fallback for unrecognized epochs
        LOKI_MODELS[parent_dir] = pt_path

# Select which model to use (change this or pass --loki_model via CLI)
if "best_model" in LOKI_MODELS:
    LOKI_ACTIVE_MODEL = "best_model"
elif "best_test_ap" in LOKI_MODELS:
    LOKI_ACTIVE_MODEL = "best_test_ap"
elif LOKI_MODELS:
    LOKI_ACTIVE_MODEL = list(LOKI_MODELS.keys())[0]
else:
    LOKI_ACTIVE_MODEL = "best_model" # Safe fallback

# Aggregation method (must match training config)
LOKI_AGGREGATION_METHOD = "top_k_pairs"

# Evaluation mode for checkpoints with structured table features.
# None = auto-detect from args.json (default)
# False = force standard scoring
# True = force schema-aware scoring
LOKI_USE_SCHEMA_AWARE_SCORER = None

# Version marker for schema-aware scoring outputs/cache entries.
LOKI_SCHEMA_AWARE_REPRESENTATION = "column_sketch_v1"
LOKI_CELL_LEVEL_MATCHING_REPRESENTATION = "cell_match_v1"

# Retrieval for LOKI evaluation is fixed to cross-attention only.

# ===========================================================================
# TabSTAR Configuration
# ===========================================================================
TABSTAR_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "TabSTAR"))
TABSTAR_MODEL_PATH = os.path.join(BASE_DIR, "models", "TabSTAR", "tabstar_weights")
TABSTAR_E5_PATH = os.path.join(BASE_DIR, "models", "TabSTAR", "e5_small_v2")

# ===========================================================================
# TaBERT Configuration
# ===========================================================================
TABERT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "TaBERT"))
TABERT_MODEL_PATH = os.path.join(BASE_DIR, "models", "TaBERT", "tabert_large_k3", "model.bin")

# ===========================================================================
# Evaluation parameters
# ===========================================================================
K_VALUES = [1, 2, 4, 8, 16, 32]

# Scalability study: test search space sizes perfectly doubling the scope.
# 0 represents the "Full" dataset (~2240 tables) to cap the evaluation realistically.
SCALABILITY_SIZES = [50, 100, 200, 400, 800, 1600, 0]

# ===========================================================================
# Output
# ===========================================================================
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
