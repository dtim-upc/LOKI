"""
download_models.py — Automatically download published evaluation models from Hugging Face.

Source repository:
    https://huggingface.co/shaoncsecu/LOKI/tree/main/Exp-2/models

Models downloaded into SOTA_Evaluation_New/models/:
    - CMDL/     (EncoderNet checkpoints + FastText WEM resources)
    - LOKI/     (Checkpoints, training configs, extracted join paths)
    - TaBERT/   (TaBERT-Large K=3 model weights and configuration)
    - TabSTAR/  (TabSTAR model weights and E5-small-v2 encoder backbone)

Usage:
    # Download all 4 models + ensure base sentence encoder:
    python download_models.py

    # Download specific models:
    python download_models.py --models TabSTAR LOKI

    # Force re-download even if already present:
    python download_models.py --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

HF_REPOSITORY_ID = "shaoncsecu/LOKI"
HF_MODELS_PATH = "Exp-2/models"
MODEL_FOLDER_NAMES = ("CMDL", "LOKI", "TaBERT", "TabSTAR")
DEFAULT_BASE_ENCODER = "sentence-transformers/embeddinggemma-300m-medical"

# Signatures for verifying whether a model folder is populated and intact
KEY_FILES_PER_MODEL = {
    "CMDL": [
        "text_enet_best.pt",
        "col_enet_best.pt",
        os.path.join("resources", "fasttext", "cc", "cc.en.300.gensim"),
    ],
    "LOKI": [
        "args.json",
    ],
    "TaBERT": [
        os.path.join("tabert_large_k3", "model.bin"),
        os.path.join("tabert_large_k3", "tb_config.json"),
    ],
    "TabSTAR": [
        os.path.join("tabstar_weights", "model.safetensors"),
        os.path.join("e5_small_v2", "model.safetensors"),
    ],
}


def _is_model_present(destination_dir: Path, model_name: str) -> bool:
    """Check if the model folder and its essential files are already present."""
    model_path = destination_dir / model_name
    if not model_path.is_dir():
        return False

    key_files = KEY_FILES_PER_MODEL.get(model_name, [])
    for rel_path in key_files:
        if not (model_path / rel_path).exists():
            return False

    # Extra check for LOKI: ensure at least one .pt checkpoint exists
    if model_name == "LOKI":
        checkpoints = list(model_path.glob("**/*.pt"))
        if not checkpoints:
            return False

    return True


def get_missing_models(destination_dir: Path, requested_models: Iterable[str]) -> List[str]:
    """Return requested model names that are not yet downloaded."""
    return [m for m in requested_models if not _is_model_present(destination_dir, m)]


def _move_or_merge_tree(src: Path, dst: Path) -> None:
    """Move files recursively from src to dst, overwriting if needed."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _move_or_merge_tree(item, target)
        else:
            if target.exists():
                target.unlink()
            shutil.move(str(item), str(target))


def download_models(
    destination: str | Path = "models",
    models: Optional[Iterable[str]] = None,
    force: bool = False,
    repo_id: str = HF_REPOSITORY_ID,
    hf_prefix: str = HF_MODELS_PATH,
    include_base_encoder: bool = True,
) -> List[Path]:
    """
    Download missing published model folders from Hugging Face into destination.
    """
    requested_names = list(models) if models is not None else list(MODEL_FOLDER_NAMES)
    invalid_names = sorted(set(requested_names) - set(MODEL_FOLDER_NAMES))
    if invalid_names:
        raise ValueError(
            f"Unknown model name(s): {', '.join(invalid_names)}. Valid options: {MODEL_FOLDER_NAMES}"
        )

    destination_path = Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)

    if force:
        to_download = requested_names
    else:
        to_download = get_missing_models(destination_path, requested_names)

    if not to_download:
        print(f"[INFO] All requested models ({', '.join(requested_names)}) are already available locally in {destination_path}.")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Model download requires huggingface_hub. Install it with: "
                "pip install huggingface_hub"
            ) from exc

        print("=" * 70)
        print(f"Downloading models from Hugging Face: {repo_id}")
        print(f"  Source path:   {hf_prefix}/")
        print(f"  Target models: {', '.join(to_download)}")
        print(f"  Destination:   {destination_path}")
        print("=" * 70)

        # Stage downloads on the same filesystem inside destination_path
        # to avoid /tmp ramdisk exhaustion and cross-device copying.
        staging_dir = destination_path / ".download_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        patterns = [f"{hf_prefix}/{m}/**" for m in to_download]

        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                token=False,  # Enforce anonymous access for public model repos
                allow_patterns=patterns,
                local_dir=str(staging_dir),
            )

            source_root = staging_dir / hf_prefix
            for name in to_download:
                source = source_root / name
                target = destination_path / name
                if not source.is_dir():
                    raise FileNotFoundError(
                        f"Downloaded repository does not contain expected folder: {hf_prefix}/{name}"
                    )
                print(f"[INFO] Materializing {name} -> {target}")
                _move_or_merge_tree(source, target)

        finally:
            # Clean up temporary staging folder
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        print(f"[SUCCESS] Finished downloading: {', '.join(to_download)}")

    # Optionally ensure the base sentence encoder is available
    if include_base_encoder and ("LOKI" in requested_names):
        try:
            script_dir = Path(__file__).resolve().parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            from hf_model_resolver import ensure_repo_local_hf_snapshot

            print("\n[INFO] Checking project-local base encoder for LOKI...")
            encoder_path, source = ensure_repo_local_hf_snapshot(
                DEFAULT_BASE_ENCODER,
                allow_online=True,
            )
            print(f"[INFO] Base encoder ready: {DEFAULT_BASE_ENCODER} -> {encoder_path} ({source})")
        except Exception as err:
            print(f"[WARN] Could not ensure base encoder: {err}")

    return [destination_path / m for m in requested_names if (destination_path / m).is_dir()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatically download evaluation models from Hugging Face (shaoncsecu/LOKI/Exp-2/models)."
    )
    script_dir = Path(__file__).resolve().parent
    default_dest = script_dir / "models"

    parser.add_argument(
        "--destination",
        default=str(default_dest),
        help=f"Local destination directory for models (default: {default_dest})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_FOLDER_NAMES,
        default=list(MODEL_FOLDER_NAMES),
        help="Models to download (choices: CMDL, LOKI, TaBERT, TabSTAR; default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model files already exist locally.",
    )
    parser.add_argument(
        "--repo-id",
        default=HF_REPOSITORY_ID,
        help=f"Hugging Face repository ID (default: {HF_REPOSITORY_ID})",
    )
    parser.add_argument(
        "--skip-base-encoder",
        action="store_true",
        help="Skip checking/downloading the base sentence encoder for LOKI.",
    )

    args = parser.parse_args()

    download_models(
        destination=args.destination,
        models=args.models,
        force=args.force,
        repo_id=args.repo_id,
        include_base_encoder=not args.skip_base_encoder,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

