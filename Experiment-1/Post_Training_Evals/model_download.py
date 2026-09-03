"""Download the published Experiment-1 model folders from Hugging Face."""

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional


HF_REPOSITORY_ID = "shaoncsecu/LOKI"
HF_INPUT_MODELS_PATH = "Exp-1/Input_Models"
MODEL_FOLDER_NAMES = ("FT-Encoder", "Uni (R-S)", "Uni (S-R)", "LOKI")


def get_missing_model_folders(destination: str, model_names: Iterable[str]) -> List[str]:
    """Return requested model folders that are not present locally."""
    destination_path = Path(destination)
    return [name for name in model_names if not (destination_path / name).is_dir()]


def download_input_models(
    destination: str = "Input_Models",
    model_names: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Download missing published model folders into the local input directory."""
    requested_names = list(model_names) if model_names is not None else list(MODEL_FOLDER_NAMES)
    invalid_names = sorted(set(requested_names) - set(MODEL_FOLDER_NAMES))
    if invalid_names:
        raise ValueError(f"Unknown model folder(s): {', '.join(invalid_names)}")

    missing_names = get_missing_model_folders(destination, requested_names)
    if not missing_names:
        print("[INFO] Requested model folders are already available locally.")
        return []

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Model download requires huggingface_hub. Install it with: "
            "pip install huggingface_hub"
        ) from exc

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading model folders from {HF_REPOSITORY_ID}: {missing_names}")

    with tempfile.TemporaryDirectory(prefix="loki_input_models_") as temporary_dir:
        snapshot_download(
            repo_id=HF_REPOSITORY_ID,
            repo_type="model",
            token=False,
            allow_patterns=[f"{HF_INPUT_MODELS_PATH}/{name}/**" for name in missing_names],
            local_dir=temporary_dir,
        )
        source_root = Path(temporary_dir) / HF_INPUT_MODELS_PATH
        downloaded_paths: List[Path] = []
        for name in missing_names:
            source = source_root / name
            target = destination_path / name
            if not source.is_dir():
                raise FileNotFoundError(f"Downloaded repository did not contain: {HF_INPUT_MODELS_PATH}/{name}")
            shutil.move(str(source), str(target))
            downloaded_paths.append(target)

    print(f"[INFO] Downloaded model folders to: {destination_path}")
    return downloaded_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download published Experiment-1 models into Input_Models."
    )
    parser.add_argument(
        "--destination",
        default="Input_Models",
        help="Local directory for downloaded model folders (default: Input_Models)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_FOLDER_NAMES,
        default=list(MODEL_FOLDER_NAMES),
        help="Model folder names to download (default: all four)",
    )
    args = parser.parse_args()
    download_input_models(destination=args.destination, model_names=args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())