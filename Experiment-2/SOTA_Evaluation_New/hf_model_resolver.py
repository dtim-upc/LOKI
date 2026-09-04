from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# Define project-local directories strictly within SOTA_Evaluation_New
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
REPO_LOCAL_HF_ROOT = PROJECT_ROOT / "models" / "hf_assets"
REPO_LOCAL_HF_SNAPSHOT_ROOT = REPO_LOCAL_HF_ROOT / "snapshots"
REPO_LOCAL_HF_CACHE_ROOT = REPO_LOCAL_HF_ROOT / "cache"

DEFAULT_SNAPSHOT_IGNORE_PATTERNS = [
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.tflite",
    "*.h5",
    "*.msgpack",
    "flax_model.msgpack",
    "tf_model.h5",
    "rust_model.ot",
]


def _repo_local_hf_root_candidates() -> List[Path]:
    """Return only project-internal directories for storing/searching model assets."""
    return [REPO_LOCAL_HF_ROOT]


def _repo_local_model_dir_name(model_name: str) -> str:
    candidate_text = str(model_name or "").strip()
    return (
        candidate_text
        .replace("\\", "--")
        .replace("/", "--")
        .replace(":", "--")
        .replace(" ", "_")
    )


def _snapshot_dir_is_usable(path: Path) -> bool:
    """Check if a directory contains actual model files (not just an arbitrary folder)."""
    if not path.exists() or not path.is_dir():
        return False

    has_config = (path / "config.json").is_file() or (path / "modules.json").is_file()
    if has_config:
        return True

    for child in path.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_file() and child.suffix in [".json", ".bin", ".pt", ".safetensors"]:
            return True
    return False


def _explicit_local_model_path(model_name: str) -> Optional[str]:
    candidate_text = str(model_name or "").strip()
    if not candidate_text:
        return None

    explicit_path = Path(candidate_text).expanduser()
    if explicit_path.exists():
        if explicit_path.is_dir():
            if _snapshot_dir_is_usable(explicit_path):
                return str(explicit_path.resolve())
        else:
            return str(explicit_path.resolve())
    return None


def get_repo_local_hf_cache_folder() -> str:
    REPO_LOCAL_HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return str(REPO_LOCAL_HF_CACHE_ROOT)


def get_repo_local_hf_snapshot_dir(model_name: str) -> Path:
    dir_name = _repo_local_model_dir_name(model_name)
    # Check project-internal candidate roots first
    for root in _repo_local_hf_root_candidates():
        candidate_snapshot = root / "snapshots" / dir_name
        if _snapshot_dir_is_usable(candidate_snapshot):
            return candidate_snapshot
    REPO_LOCAL_HF_SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    return REPO_LOCAL_HF_SNAPSHOT_ROOT / dir_name


def _offline_cache_dir_candidates() -> List[Path]:
    """Only project-internal cache directories are checked — no external user caches."""
    candidates: List[Path] = []
    for root in _repo_local_hf_root_candidates():
        cache_dir = root / "cache"
        if cache_dir.is_dir():
            candidates.append(cache_dir)
    return candidates


def _materialize_repo_local_snapshot(
    model_name: str,
    local_dir: Path,
    cache_dir: Optional[Path],
    local_files_only: bool,
    token: Optional[Any] = False,
) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[hf_model_resolver] Failed to import snapshot_download: {e}")
        return False

    local_dir.mkdir(parents=True, exist_ok=True)

    download_kwargs: Dict[str, Any] = {
        "repo_id": model_name,
        "local_dir": str(local_dir),
        "local_files_only": local_files_only,
        "ignore_patterns": DEFAULT_SNAPSHOT_IGNORE_PATTERNS,
        "token": token,
    }
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        download_kwargs["cache_dir"] = str(cache_dir)

    try:
        if not local_files_only:
            print(f"[hf_model_resolver] Downloading '{model_name}' to local project folder: {local_dir} (token={token})")
        snapshot_download(**download_kwargs)
    except Exception as e:
        if not local_files_only:
            print(f"[hf_model_resolver] Download attempt (token={token}) encountered: {e}")
        return False

    return _snapshot_dir_is_usable(local_dir)


def ensure_repo_local_hf_snapshot(
    model_name: str,
    allow_online: bool = True,
) -> Tuple[str, str]:
    """
    Ensure the model is present in a local folder within the project.
    If not already available locally, download it anonymously (token=False) directly into
    the project folder.
    """
    explicit_path = _explicit_local_model_path(model_name)
    if explicit_path is not None:
        return explicit_path, "explicit_path"

    candidate_text = str(model_name or "").strip()
    if not candidate_text:
        return candidate_text, "empty"

    repo_local_snapshot_dir = get_repo_local_hf_snapshot_dir(candidate_text)
    if _snapshot_dir_is_usable(repo_local_snapshot_dir):
        return str(repo_local_snapshot_dir), "project_local_snapshot"

    # Check project-internal cache directories first
    for cache_dir in _offline_cache_dir_candidates():
        if _materialize_repo_local_snapshot(
            candidate_text,
            repo_local_snapshot_dir,
            cache_dir=cache_dir,
            local_files_only=True,
            token=False,
        ):
            return str(repo_local_snapshot_dir), "materialized_from_local_cache"

    if allow_online:
        # Download into project local folder anonymously (token=False).
        # This prevents 400 Bad Request errors when an invalid/expired token is in the environment.
        print(f"[hf_model_resolver] Model '{candidate_text}' not found in project. Downloading into local folder...")
        local_cache = Path(get_repo_local_hf_cache_folder())

        if _materialize_repo_local_snapshot(
            candidate_text,
            repo_local_snapshot_dir,
            cache_dir=local_cache,
            local_files_only=False,
            token=False,
        ):
            return str(repo_local_snapshot_dir), "downloaded_to_project"

        # Fallback to token=None only if token=False fails (e.g. for gated/private models)
        if _materialize_repo_local_snapshot(
            candidate_text,
            repo_local_snapshot_dir,
            cache_dir=local_cache,
            local_files_only=False,
            token=None,
        ):
            return str(repo_local_snapshot_dir), "downloaded_to_project_with_token"

    raise RuntimeError(
        f"Unable to find or download model '{candidate_text}' into local project folder '{repo_local_snapshot_dir}'. "
        "The model was not found in the project directory, and download did not succeed."
    )


def bootstrap_hf_model_snapshots(
    model_names: List[str],
    allow_online: bool = True,
) -> List[Dict[str, str]]:
    bootstrap_records: List[Dict[str, str]] = []
    seen: set[str] = set()

    for raw_model_name in model_names:
        model_name = str(raw_model_name or "").strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)

        resolved_path, source = ensure_repo_local_hf_snapshot(
            model_name,
            allow_online=allow_online,
        )
        bootstrap_records.append({
            "model_name": model_name,
            "resolved_path": resolved_path,
            "source": source,
        })

    return bootstrap_records


def load_hf_model_with_cache_fallback(
    loader: Callable[..., Any],
    model_name: str,
    **kwargs: Any,
) -> Tuple[Any, str, str]:
    candidate_text = str(model_name or "").strip()
    base_kwargs = dict(kwargs)

    resolved_model_name, model_source = ensure_repo_local_hf_snapshot(
        candidate_text,
        allow_online=True,
    )

    local_kwargs = dict(base_kwargs)
    local_kwargs.setdefault("local_files_only", True)
    local_kwargs.setdefault("token", False)
    model = loader(resolved_model_name, **local_kwargs)
    return model, resolved_model_name, model_source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and resolve HuggingFace models to local project folder.")
    parser.add_argument(
        "model_name",
        nargs="?",
        default="sentence-transformers/embeddinggemma-300m-medical",
        help="Hugging Face model name to download to project local folder (default: sentence-transformers/embeddinggemma-300m-medical)",
    )
    args = parser.parse_args()
    print(f"[hf_model_resolver] Resolving model: {args.model_name}")
    path, source = ensure_repo_local_hf_snapshot(args.model_name, allow_online=True)
    print(f"[hf_model_resolver] Result: {path} ({source})")

