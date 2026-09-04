"""
download_tabstar_weights.py — One-time script to save TabSTAR model weights locally.

Downloads from HuggingFace and saves to:
    models/TabSTAR/tabstar_weights/   (full TabSTAR model — alana89/TabSTAR)
    models/TabSTAR/e5_small_v2/       (E5-small-v2 text encoder — intfloat/e5-small-v2)

After running this script, evaluate_tabstar.py will load weights from these
local directories instead of the HuggingFace cache, consistent with how
CMDL, LOKI, and TaBERT store their weights under models/.

Usage:
    python download_tabstar_weights.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABSTAR_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "TabSTAR"))
TABSTAR_SRC = os.path.join(TABSTAR_DIR, "src")
for p in (TABSTAR_DIR, TABSTAR_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoModel, AutoTokenizer
from tabstar.arch.arch import TabStarModel

def main():
    models_dir = os.path.join(SCRIPT_DIR, "models", "TabSTAR")
    os.makedirs(models_dir, exist_ok=True)

    # ── 1. TabSTAR (full model) ──────────────────────────────────────────
    tabstar_out = os.path.join(models_dir, "tabstar_weights")
    if os.path.isdir(tabstar_out) and os.listdir(tabstar_out):
        print(f"[skip] TabSTAR weights already exist at {tabstar_out}")
    else:
        print(f"[1/2] Downloading TabSTAR (alana89/TabSTAR) -> {tabstar_out}")
        model = TabStarModel.from_pretrained("alana89/TabSTAR")
        model.save_pretrained(tabstar_out)
        model.tokenizer.save_pretrained(tabstar_out)
        print("      Done.\n")

    # ── 2. E5-small-v2 (sentence encoder backbone) ───────────────────────
    e5_out = os.path.join(models_dir, "e5_small_v2")
    if os.path.isdir(e5_out) and os.listdir(e5_out):
        print(f"[skip] E5-small-v2 weights already exist at {e5_out}")
    else:
        print(f"[2/2] Downloading E5-small-v2 (intfloat/e5-small-v2) -> {e5_out}")
        e5_model = AutoModel.from_pretrained("intfloat/e5-small-v2")
        e5_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
        e5_model.save_pretrained(e5_out)
        e5_tokenizer.save_pretrained(e5_out)
        print("      Done.\n")

    print("All TabSTAR weights saved under %s" % models_dir)
    print("evaluate_tabstar.py will now load from these local paths.")


if __name__ == "__main__":
    main()
