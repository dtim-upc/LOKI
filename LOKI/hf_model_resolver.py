from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


HF_HUB_HOME_ENV = "HF_HUB_HOME"
WORKSPACE_ROOT = Path(__file__).parent.parent
REPO_LOCAL_HF_ROOT = WORKSPACE_ROOT / "model" / "hf_assets"
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


def _dedupe_optional_paths(paths: List[Optional[Path]]) -> List[Optional[Path]]:
    deduped: List[Optional[Path]] = []
    seen: set[str] = set()
    for path in paths:
        key = "<default>" if path is None else str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _explicit_local_model_path(model_name: str) -> Optional[str]:
    candidate_text = str(model_name or "").strip()
    if not candidate_text:
        return None

    explicit_path = Path(candidate_text).expanduser()
    if explicit_path.exists():
        return str(explicit_path)
    return None


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
    if not path.exists() or not path.is_dir():
        return False

    for child in path.iterdir():
        if child.name == ".cache":
            continue
        return True
    return False


def get_repo_local_hf_cache_folder() -> str:
    REPO_LOCAL_HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return str(REPO_LOCAL_HF_CACHE_ROOT)


def get_repo_local_hf_snapshot_dir(model_name: str) -> Path:
    REPO_LOCAL_HF_SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    return REPO_LOCAL_HF_SNAPSHOT_ROOT / _repo_local_model_dir_name(model_name)


def _hf_hub_home_candidates() -> List[Path]:
    raw_value = str(os.getenv(HF_HUB_HOME_ENV) or "").strip()
    if not raw_value:
        return []

    raw_path = Path(raw_value).expanduser()
    if raw_path.name.lower() == "hub":
        return [raw_path]
    return [raw_path / "hub", raw_path]


def get_configured_hf_cache_folder() -> Optional[str]:
    candidates = _hf_hub_home_candidates()
    if not candidates:
        return None
    return str(candidates[0])


def _offline_cache_dir_candidates() -> List[Optional[Path]]:
    candidates: List[Optional[Path]] = []
    candidates.append(Path(get_repo_local_hf_cache_folder()))
    candidates.extend(_hf_hub_home_candidates())
    candidates.append(None)
    return _dedupe_optional_paths(candidates)


def _materialize_repo_local_snapshot(
    model_name: str,
    local_dir: Path,
    cache_dir: Optional[Path],
    local_files_only: bool,
) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return False

    download_kwargs: Dict[str, Any] = {
        "repo_id": model_name,
        "local_dir": str(local_dir),
        "local_files_only": local_files_only,
        "ignore_patterns": DEFAULT_SNAPSHOT_IGNORE_PATTERNS,
    }
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)

    try:
        snapshot_download(**download_kwargs)
    except Exception:
        return False

    return _snapshot_dir_is_usable(local_dir)


def ensure_repo_local_hf_snapshot(
    model_name: str,
    allow_online: bool = True,
) -> Tuple[str, str]:
    explicit_path = _explicit_local_model_path(model_name)
    if explicit_path is not None:
        return explicit_path, "explicit_path"

    candidate_text = str(model_name or "").strip()
    if not candidate_text:
        return candidate_text, "empty"

    repo_local_snapshot_dir = get_repo_local_hf_snapshot_dir(candidate_text)
    if _snapshot_dir_is_usable(repo_local_snapshot_dir):
        return str(repo_local_snapshot_dir), "repo_local_snapshot"

    for cache_dir in _offline_cache_dir_candidates():
        if _materialize_repo_local_snapshot(
            candidate_text,
            repo_local_snapshot_dir,
            cache_dir=cache_dir,
            local_files_only=True,
        ):
            if cache_dir is None:
                return str(repo_local_snapshot_dir), "repo_local_from_default_cache"
            if cache_dir == Path(get_repo_local_hf_cache_folder()):
                return str(repo_local_snapshot_dir), "repo_local_from_managed_cache"
            return str(repo_local_snapshot_dir), "repo_local_from_external_cache"

    if allow_online and _materialize_repo_local_snapshot(
        candidate_text,
        repo_local_snapshot_dir,
        cache_dir=Path(get_repo_local_hf_cache_folder()),
        local_files_only=False,
    ):
        return str(repo_local_snapshot_dir), "downloaded_to_repo"

    cached_snapshot = resolve_cached_hf_snapshot(candidate_text)
    if cached_snapshot is not None:
        return cached_snapshot, "cached_snapshot"

    raise RuntimeError(
        f"Unable to materialize a repo-local snapshot for '{candidate_text}'. "
        "The model was not found in the managed repo cache, in HF_HUB_HOME/default caches, "
        "and an online download was not successful."
    )


def resolve_cached_hf_snapshot(model_name: str) -> Optional[str]:
    explicit_path = _explicit_local_model_path(model_name)
    if explicit_path is not None:
        return explicit_path

    candidate_text = str(model_name or "").strip()
    if not candidate_text:
        return None

    repo_local_snapshot_dir = get_repo_local_hf_snapshot_dir(candidate_text)
    if _snapshot_dir_is_usable(repo_local_snapshot_dir):
        return str(repo_local_snapshot_dir)

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None

    for cache_dir in _offline_cache_dir_candidates():
        download_kwargs: Dict[str, Any] = {
            "repo_id": candidate_text,
            "local_files_only": True,
        }
        if cache_dir is not None:
            download_kwargs["cache_dir"] = str(cache_dir)

        try:
            snapshot_path = snapshot_download(**download_kwargs)
        except Exception:
            continue

        resolved_path = Path(snapshot_path)
        if resolved_path.exists():
            return str(resolved_path)

    return None


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
    model = loader(resolved_model_name, **local_kwargs)
    return model, resolved_model_name, model_source
