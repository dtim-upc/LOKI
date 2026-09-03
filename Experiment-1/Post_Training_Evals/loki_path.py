from pathlib import Path
import importlib.util
import sys
from types import ModuleType


def ensure_loki_on_path() -> Path:
    """Prepend the sibling LOKI directory to sys.path."""
    loki_dir = Path(__file__).resolve().parent.parent / "LOKI"
    if not loki_dir.is_dir():
        raise ImportError(f"LOKI directory not found at expected path: {loki_dir}")

    loki_dir_str = str(loki_dir)
    sys.path[:] = [loki_dir_str] + [entry for entry in sys.path if entry != loki_dir_str]
    return loki_dir


def load_loki_module(module_name: str) -> ModuleType:
    """Load a Python module explicitly from the sibling LOKI directory."""
    loki_dir = ensure_loki_on_path()
    module_path = loki_dir / f"{module_name}.py"
    if not module_path.is_file():
        raise ImportError(f"Module {module_name!r} not found in LOKI directory: {module_path}")

    cache_key = f"_loki_{module_name}"
    cached_module = sys.modules.get(cache_key)
    if cached_module is not None:
        return cached_module

    spec = importlib.util.spec_from_file_location(cache_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = module
    spec.loader.exec_module(module)
    return module
