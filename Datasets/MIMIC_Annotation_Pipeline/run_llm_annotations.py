"""
Run LLM-based annotation over LOKI combined prompt files.

Reads .md prompt files from prompts_combined/ (or a single file), sends each
prompt to a configurable LLM API (OpenAI, OpenRouter, Qwen/ModelStudio, or a
local LM Studio server), extracts the JSON annotation block from the response,
and saves one JSON file per admission to an output directory named after the
model.

Output JSON files are compatible with merge_annotations.py.

Usage examples:
    # OpenAI — direct API (api.openai.com)
    python run_llm_annotations.py --provider openai --model gpt-4o
    python run_llm_annotations.py --provider openai --model gpt-4o-mini
    python run_llm_annotations.py --provider openai --model o3

    # OpenRouter — any model available at openrouter.ai
    python run_llm_annotations.py --provider openrouter --model openai/gpt-5.4
    python run_llm_annotations.py --provider openrouter --model openai/gpt-oss-120b:free
    python run_llm_annotations.py --provider openrouter --model anthropic/claude-opus-4.6
    python run_llm_annotations.py --provider openrouter --model google/gemini-3.1-pro-preview

    # Qwen — Alibaba Cloud Model Studio (dashscope-intl)
    python run_llm_annotations.py --provider qwen --model qwen3.6-plus
    python run_llm_annotations.py --provider qwen --model qwen3.6-flash
    python run_llm_annotations.py --provider qwen --model qwen-max
    python run_llm_annotations.py --provider qwen --model qwen3.5-397b-a17b
    # Override region endpoint (default: Singapore)
    python run_llm_annotations.py --provider qwen --model qwen3.6-plus --base_url https://dashscope-intl.aliyuncs.com/compatible-mode/v1

    # LM Studio — local server
    python run_llm_annotations.py --provider lmstudio --model qwen3.6-35b-a3b
    python run_llm_annotations.py --provider lmstudio --model qwen3.6-35b-a3b --base_url http://10.6.144.6:1234

    # Filters / options
    python run_llm_annotations.py --provider openai --model gpt-4o --admission_id 23223704
    python run_llm_annotations.py --provider openai --model gpt-4o --dry_run

    # Create templates
    python run_llm_annotations.py --create_templates
    python run_llm_annotations.py --create_templates --templates_dir annotation_templates

Output folders are created automatically under llm_outputs/ named after the model,
e.g. llm_outputs/gpt-4o/ or llm_outputs/openai__gpt-4o/ or llm_outputs/qwen3.5-plus/.

Environment variables:
    OPENAI_API_KEY     — OpenAI API key (used for --provider openai)
                         Get one at: https://platform.openai.com/api-keys
    OPENROUTER_API_KEY — OpenRouter API key (used for --provider openrouter)
                         Get one at: https://openrouter.ai/keys
    DASHSCOPE_API_KEY  — Alibaba Cloud Model Studio API key (used for --provider qwen)
                         Get one at: https://www.alibabacloud.com/help/en/model-studio/get-api-key
    ANTHROPIC_AUTH_TOKEN — LM Studio token (optional; omit if server has no auth)
"""

import json
import os
import re
import sys
import time
import argparse
import traceback
from functools import partial
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Default models per provider
# ---------------------------------------------------------------------------
DEFAULT_MODELS = {
    "openai":     "gpt-4o",
    "openrouter": "openai/gpt-4o",
    "qwen":       "qwen3.6-plus",
    "lmstudio":   "local-model",  # must be overridden with --model <id-from-lmstudio>
}

# Default server / endpoint base URLs
OPENAI_BASE_URL           = "https://api.openai.com/v1"
OPENROUTER_BASE_URL       = "https://openrouter.ai/api/v1"
LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234"

# Alibaba Cloud Model Studio — international endpoint (Singapore region).
# Other regions:
#   US (Virginia)   : https://dashscope-us.aliyuncs.com/compatible-mode/v1
#   China (HK)      : https://dashscope-intl.aliyuncs.com/compatible-mode/v1
#   Germany (FRA)   : https://dashscope-intl.aliyuncs.com/compatible-mode/v1
QWEN_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Maps each OpenAI-compatible provider to its default base URL and API key env var.
OPENAI_COMPATIBLE_PROVIDERS = {
    "openai":     {"base_url": OPENAI_BASE_URL,           "api_key_env": "OPENAI_API_KEY"},
    "openrouter": {"base_url": OPENROUTER_BASE_URL,       "api_key_env": "OPENROUTER_API_KEY"},
    "qwen":       {"base_url": QWEN_DEFAULT_BASE_URL,     "api_key_env": "DASHSCOPE_API_KEY"},
    "lmstudio":   {"base_url": LMSTUDIO_DEFAULT_BASE_URL, "api_key_env": "ANTHROPIC_AUTH_TOKEN"},
}


# ---------------------------------------------------------------------------
# Model name → filesystem-safe folder name
# ---------------------------------------------------------------------------

def model_to_folder(model: str) -> str:
    """
    Convert a model ID to a filesystem-safe directory name.

    Examples:
        openai/gpt-4o          -> openai__gpt-4o
        ibm/granite-4-micro    -> ibm__granite-4-micro
        google/gemini-2.5-pro  -> google__gemini-2.5-pro
    """
    # Replace forward slashes (org/model) with double-underscore
    safe = model.replace("/", "__")
    # Replace any remaining characters that are problematic on Windows/Linux
    safe = re.sub(r'[<>:"|?*\\]', "_", safe)
    return safe


# ---------------------------------------------------------------------------
# Provider call functions
# ---------------------------------------------------------------------------

def call_openai_compatible(
    prompt: str,
    model: str,
    base_url: str,
    system_prompt: str = "",
    *,
    api_key_env: str,
) -> str:
    """
    Send a prompt to any OpenAI-compatible chat completions endpoint.

    Used for OpenAI, OpenRouter, and Qwen (Alibaba Cloud Model Studio). The only
    differences between providers are the base URL and the API key env var, both
    of which are supplied by the caller via functools.partial.

    Parameters
    ----------
    prompt       : The per-admission user message.
    model        : Provider model slug, e.g. "gpt-4o", "openai/gpt-4o", or "qwen3.5-plus".
    base_url     : Full endpoint URL, e.g. OPENAI_BASE_URL, OPENROUTER_BASE_URL, or
                   QWEN_DEFAULT_BASE_URL. Pass --base_url on the CLI to override.
    system_prompt: Optional static system message (sent as role="system").
    api_key_env  : Name of the environment variable that holds the API key.
                   openai     → OPENAI_API_KEY     (from platform.openai.com/api-keys)
                   openrouter → OPENROUTER_API_KEY (from openrouter.ai/keys)
                   qwen       → DASHSCOPE_API_KEY  (from alibabacloud.com)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai>=1.0"
        )

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{api_key_env} environment variable is not set. "
            f"Export it before running: set {api_key_env}=<your-key>"
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def call_lmstudio(
    prompt: str,
    model: str,
    base_url: str = LMSTUDIO_DEFAULT_BASE_URL,
    system_prompt: str = "",
) -> str:
    """
    Send prompt to a local LM Studio server and return the text response.

    Uses LM Studio's /api/v1/chat endpoint.
    Set ANTHROPIC_AUTH_TOKEN in your environment if the server requires authentication;
    leave it unset to skip the Authorization header entirely.
    Override the server address with --base_url (default: http://localhost:1234).

    When system_prompt is provided it is prepended to the input field so the
    static annotation instructions are clearly separated from the admission data.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError(
            "requests package not installed. Run: pip install requests"
        )

    api_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    # Combine system + user content for the /api/v1/chat input field.
    full_input = f"{system_prompt}\n\n---\n\n{prompt}" if system_prompt else prompt

    url = f"{base_url.rstrip('/')}/api/v1/chat"
    payload = {
        "model": model,
        "input": full_input,
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()

    # LM Studio /api/v1/chat returns {"output": "..."} for plain text responses.
    # Fall back to OpenAI-style choices if the server is in compatibility mode.
    if "output" in data:
        return data["output"]
    if "choices" in data and data["choices"]:
        return data["choices"][0].get("message", {}).get("content", "")
    raise ValueError(f"Unexpected LM Studio response structure: {list(data.keys())}")


PROVIDER_FUNCTIONS = {
    "openai":     partial(call_openai_compatible, api_key_env="OPENAI_API_KEY"),
    "openrouter": partial(call_openai_compatible, api_key_env="OPENROUTER_API_KEY"),
    "qwen":       partial(call_openai_compatible, api_key_env="DASHSCOPE_API_KEY"),
    "lmstudio":   call_lmstudio,
}


# ---------------------------------------------------------------------------
# Rate-limit error detection
# ---------------------------------------------------------------------------

def is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception looks like a provider rate-limit error."""
    msg = str(exc).lower()
    rate_limit_keywords = [
        "rate limit", "ratelimit", "rate_limit",
        "quota", "too many requests", "429",
        "resource exhausted", "resourceexhausted",
    ]
    return any(kw in msg for kw in rate_limit_keywords)


# ---------------------------------------------------------------------------
# JSON extraction from LLM response
# ---------------------------------------------------------------------------

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract and parse the JSON annotation block from the LLM response.

    Tries in order:
      1. ```json ... ``` fenced block
      2. ``` ... ``` fenced block (any language)
      3. First { ... } spanning the entire string
    """
    # Strategy 1: explicit ```json fence
    json_fence = re.search(r"```json\s*([\s\S]+?)\s*```", response_text)
    if json_fence:
        return json.loads(json_fence.group(1))

    # Strategy 2: any ``` fence
    any_fence = re.search(r"```\s*([\s\S]+?)\s*```", response_text)
    if any_fence:
        candidate = any_fence.group(1).strip()
        if candidate.startswith("{"):
            return json.loads(candidate)

    # Strategy 3: find outermost { } in the raw text
    brace_start = response_text.find("{")
    brace_end = response_text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        return json.loads(response_text[brace_start : brace_end + 1])

    raise ValueError("No JSON object found in LLM response.")


# ---------------------------------------------------------------------------
# Output filename helpers + template creation
# ---------------------------------------------------------------------------

def derive_ids_from_filename(prompt_file: Path) -> Tuple[str, str]:
    """
    Extract patient_id and admission_id from the prompt filename.

    Expected pattern: prompt_combined_<patient_id>_<admission_id>.md
    Returns ("", "") if the pattern does not match.
    """
    match = re.search(r"prompt_combined_(\d+)_(\d+)", prompt_file.name)
    if match:
        return match.group(1), match.group(2)
    return "", ""


def parse_ids_from_prompt(prompt_file: Path) -> Dict[str, Any]:
    """
    Parse patient_id, admission_id, diagnosis_anchor_id, and medication_anchor_id
    from the markdown content of a data prompt file.

    The generate_prompts.py script writes these as:
        **Patient ID**: `10000764`
        **Admission ID**: `27897940`
        **Diagnosis Anchor ID**: `12345`
        **Medication Anchor ID**: `67890`
    """
    text = prompt_file.read_text(encoding="utf-8")
    ids: Dict[str, Any] = {}

    patterns = {
        "patient_id":            r"\*\*Patient ID\*\*:\s*`([^`]+)`",
        "admission_id":          r"\*\*Admission ID\*\*:\s*`([^`]+)`",
        "diagnosis_anchor_id":   r"\*\*Diagnosis Anchor ID\*\*:\s*`?([0-9]+)`?",
        "medication_anchor_id":  r"\*\*Medication Anchor ID\*\*:\s*`?([0-9]+)`?",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            val = m.group(1)
            # anchor IDs are integers in the schema
            if "anchor_id" in key:
                try:
                    ids[key] = int(val)
                except ValueError:
                    ids[key] = val
            else:
                ids[key] = val

    return ids


def create_annotation_template(prompt_file: Path, templates_dir: Path) -> Path:
    """
    Create a skeleton annotation JSON for manual annotators.

    Reads the prompt file to extract all four IDs, then writes a pre-filled
    JSON with empty annotation fields to templates_dir. The annotator opens
    this file, fills in row_grounding and relationships, and places it in
    Annotations/Individual/<annotator>/.

    The file is written to templates_dir (not llm_outputs/) so it never
    interferes with the LLM runner's skip logic.
    """
    ids = parse_ids_from_prompt(prompt_file)

    patient_id   = ids.get("patient_id", "")
    admission_id = ids.get("admission_id", "")

    # Fall back to filename-derived IDs if parsing failed
    if not patient_id or not admission_id:
        patient_id, admission_id = derive_ids_from_filename(prompt_file)

    template: Dict[str, Any] = {
        "patient_id":           patient_id,
        "admission_id":         admission_id,
        "diagnosis_anchor_id":  ids.get("diagnosis_anchor_id", None),
        "medication_anchor_id": ids.get("medication_anchor_id", None),

        "row_grounding": {
            "diagnosis":  {},
            "medication": {},
        },

        "relationships":          [],
        "multi_relationship_flags": [],
        "negative_relationships": [],
        "quality_notes":          None,

        "_template":    True,
        "_prompt_file": prompt_file.name,
        "_created_at":  datetime.now().isoformat(),
    }

    if patient_id and admission_id:
        filename = f"annotation_{patient_id}_{admission_id}.json"
    else:
        filename = prompt_file.stem + ".json"

    out_path = templates_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(template, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def build_output_path(output_dir: Path, annotation: Dict, prompt_file: Path) -> Path:
    """
    Determine the output JSON file path.

    Prefers patient_id / admission_id from the parsed annotation; falls back to
    values derived from the prompt filename.
    """
    patient_id = str(annotation.get("patient_id", ""))
    admission_id = str(annotation.get("admission_id", ""))

    if not patient_id or not admission_id:
        patient_id, admission_id = derive_ids_from_filename(prompt_file)

    if patient_id and admission_id:
        filename = f"annotation_{patient_id}_{admission_id}.json"
    else:
        filename = prompt_file.stem + ".json"

    return output_dir / filename


# ---------------------------------------------------------------------------
# Annotation validation
# ---------------------------------------------------------------------------

VALID_RELATIONSHIP_TYPES = {"TREATS", "ADVERSE_EFFECT", "CONTRAINDICATED", "DISCONTINUED"}


def validate_annotation(annotation: Dict) -> List[str]:
    """Return a list of validation issues (empty list = valid)."""
    issues: List[str] = []

    required_top = ["row_grounding", "relationships"]
    for field in required_top:
        if field not in annotation:
            issues.append(f"Missing required field: {field}")

    if "relationships" in annotation:
        for i, rel in enumerate(annotation["relationships"]):
            for req in ("drug_row", "diagnosis_row", "relationship_type"):
                if req not in rel:
                    issues.append(f"Relationship {i}: missing '{req}'")
            rt = rel.get("relationship_type", "")
            if rt and rt not in VALID_RELATIONSHIP_TYPES:
                issues.append(
                    f"Relationship {i}: invalid relationship_type '{rt}'; "
                    f"must be one of {sorted(VALID_RELATIONSHIP_TYPES)}"
                )

    return issues


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def call_with_retry(
    provider: str,
    model: str,
    prompt: str,
    max_retries: int,
    base_delay: float,
    base_url: str = LMSTUDIO_DEFAULT_BASE_URL,
    system_prompt: str = "",
) -> str:
    """Call the LLM, retrying on transient/rate-limit errors with exponential backoff."""
    call_fn = PROVIDER_FUNCTIONS[provider]
    delay = base_delay

    for attempt in range(1, max_retries + 2):  # +1 for the initial attempt
        try:
            return call_fn(prompt, model, base_url, system_prompt)
        except Exception as exc:
            is_last = attempt == max_retries + 1
            if is_last:
                raise

            if is_rate_limit_error(exc):
                wait = delay * (2 ** (attempt - 1))
                print(
                    f"    [RATE LIMIT] attempt {attempt}/{max_retries + 1} — "
                    f"waiting {wait:.0f}s before retry..."
                )
                time.sleep(wait)
            else:
                wait = delay * (2 ** (attempt - 1))
                print(
                    f"    [ERROR] attempt {attempt}/{max_retries + 1}: {exc} — "
                    f"retrying in {wait:.0f}s..."
                )
                time.sleep(wait)

    raise RuntimeError("Unreachable")


def process_prompt_file(
    prompt_file: Path,
    provider: str,
    model: str,
    output_dir: Path,
    annotator: str,
    max_retries: int,
    base_delay: float,
    force: bool,
    dry_run: bool,
    base_url: str = LMSTUDIO_DEFAULT_BASE_URL,
    system_prompt: str = "",
) -> str:
    """
    Process a single prompt file.

    Returns one of: "skipped", "dry_run", "ok", "failed"
    """
    # Derive IDs from filename for early skip check
    patient_id_hint, admission_id_hint = derive_ids_from_filename(prompt_file)

    # Early skip check (before reading the file)
    if not force:
        candidate_path = output_dir / f"annotation_{patient_id_hint}_{admission_id_hint}.json"
        if patient_id_hint and admission_id_hint and candidate_path.exists():
            print(f"  [SKIP] {prompt_file.name} (output exists)")
            return "skipped"

    if dry_run:
        print(f"  [DRY RUN] Would process: {prompt_file.name}")
        return "dry_run"

    # Read prompt
    prompt_text = prompt_file.read_text(encoding="utf-8")

    # Call LLM
    print(f"  [CALL] {prompt_file.name} → {provider}/{model} ...", end="", flush=True)
    t0 = time.time()
    raw_response = call_with_retry(provider, model, prompt_text, max_retries, base_delay, base_url, system_prompt)
    elapsed = time.time() - t0
    print(f" ({elapsed:.1f}s)")

    # Extract JSON
    annotation = extract_json_from_response(raw_response)

    # Enrich with metadata if missing
    if "patient_id" not in annotation and patient_id_hint:
        annotation["patient_id"] = patient_id_hint
    if "admission_id" not in annotation and admission_id_hint:
        annotation["admission_id"] = admission_id_hint

    # Tag with annotator and timestamp
    annotation["_annotator"] = annotator
    annotation["_model"] = f"{provider}/{model}"
    annotation["_timestamp"] = datetime.now().isoformat()
    annotation["_prompt_file"] = prompt_file.name

    # Validate
    issues = validate_annotation(annotation)
    if issues:
        print(f"    [WARN] Validation issues:")
        for issue in issues:
            print(f"           • {issue}")

    # Save
    out_path = build_output_path(output_dir, annotation, prompt_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(annotation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"    [SAVED] → {out_path}")
    return "ok"


# ---------------------------------------------------------------------------
# Input file collection
# ---------------------------------------------------------------------------

def collect_prompt_files(
    input_dir: Optional[str],
    input_file: Optional[str],
    admission_id: Optional[str],
) -> List[Path]:
    """Collect the list of prompt .md files to process."""
    if input_file:
        path = Path(input_file)
        if not path.exists():
            print(f"[ERROR] File not found: {input_file}", file=sys.stderr)
            sys.exit(1)
        return [path]

    dir_path = Path(input_dir or "prompts_combined")
    if not dir_path.is_dir():
        print(f"[ERROR] Directory not found: {dir_path}", file=sys.stderr)
        sys.exit(1)

    files = sorted(dir_path.glob("*.md"))

    if admission_id:
        files = [f for f in files if f"_{admission_id}" in f.stem]
        if not files:
            print(
                f"[ERROR] No prompt file found for admission_id={admission_id} "
                f"in {dir_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM annotations over LOKI combined prompt files.\n\n"
            "Providers:\n"
            "  openai      — direct OpenAI API at api.openai.com (set OPENAI_API_KEY)\n"
            "  openrouter  — any model via openrouter.ai (set OPENROUTER_API_KEY)\n"
            "  qwen        — Alibaba Cloud Model Studio / DashScope (set DASHSCOPE_API_KEY)\n"
            "  lmstudio    — local LM Studio server (set ANTHROPIC_AUTH_TOKEN if needed)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Template creation (no API needed)
    parser.add_argument(
        "--create_templates",
        action="store_true",
        help=(
            "Create pre-filled annotation skeleton JSON files for manual annotators. "
            "Each file contains patient_id, admission_id, and both anchor IDs, with "
            "empty row_grounding / relationships fields ready to fill in. "
            "Files are written to --templates_dir (default: annotation_templates/) "
            "and do NOT interfere with the LLM runner's skip logic. "
            "No API calls are made. --provider is not required in this mode."
        ),
    )
    parser.add_argument(
        "--templates_dir",
        type=str,
        default="annotation_templates",
        help="Output directory for skeleton annotation files (default: annotation_templates/)",
    )

    # Provider / model
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_FUNCTIONS),
        default=None,
        help="LLM provider: openrouter | qwen | lmstudio (required unless --create_templates)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model ID to use. "
            "For openai: any OpenAI model slug, e.g. gpt-4o, gpt-4o-mini, o3. "
            "For openrouter: any slug from openrouter.ai, e.g. openai/gpt-4o, "
            "anthropic/claude-opus-4-5, google/gemini-2.5-pro. "
            "For qwen: any Qwen model slug, e.g. qwen3.5-plus, qwen3.5-flash, "
            "qwen-max, qwen3.5-397b-a17b. "
            "For lmstudio: model ID as shown in LM Studio, e.g. ibm/granite-4-micro."
        ),
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help=(
            "Override the provider's default endpoint URL. "
            f"For openai: defaults to {OPENAI_BASE_URL}. "
            "For qwen: defaults to the Singapore DashScope endpoint "
            f"({QWEN_DEFAULT_BASE_URL}). "
            "For lmstudio: defaults to http://localhost:1234. "
            "Use this to select a different Qwen region or a remote LM Studio host."
        ),
    )

    # Input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input_dir",
        type=str,
        default="prompts_combined",
        help="Directory containing prompt_combined_*.md files (default: prompts_combined/)",
    )
    input_group.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Process a single prompt .md file instead of a whole directory",
    )
    parser.add_argument(
        "--admission_id",
        type=str,
        default=None,
        help="Filter to a single admission ID (matched in filename)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="llm_outputs",
        help="Root directory for output JSON files (default: llm_outputs/)",
    )
    parser.add_argument(
        "--annotator",
        type=str,
        default=None,
        help=(
            "Annotator label written into each JSON and used as the output subfolder name. "
            "Defaults to a sanitized version of --model (e.g. openai__gpt-4o). "
            "Used by merge_annotations.py to identify the source."
        ),
    )

    # System prompt (static instructions, generated by generate_prompts.py)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="system_prompt.md",
        help=(
            "Path to the static system prompt file produced by generate_prompts.py "
            "(default: system_prompt.md). Sent as the `system` message so the static "
            "instructions are cached and not re-billed on every call. "
            "Pass an empty string to disable."
        ),
    )

    # Behaviour
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between successful API calls (default: 1.0)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Number of retries on transient errors (default: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files (default: skip already-processed prompts)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be processed without making any API calls",
    )

    args = parser.parse_args()

    # In --create_templates mode a provider is not required
    if not args.create_templates and args.provider is None:
        parser.error("--provider is required unless --create_templates is set")

    # Resolve defaults
    model = args.model or (DEFAULT_MODELS.get(args.provider, "") if args.provider else "")

    # Resolve base_url: explicit --base_url wins; otherwise use the provider default
    if args.base_url:
        base_url = args.base_url
    elif args.provider in OPENAI_COMPATIBLE_PROVIDERS:
        base_url = OPENAI_COMPATIBLE_PROVIDERS[args.provider]["base_url"]
    else:
        base_url = LMSTUDIO_DEFAULT_BASE_URL

    # Warn if provider-specific model was not explicitly specified
    if args.provider == "lmstudio" and not args.model:
        print(
            "[WARN] No --model specified for lmstudio. "
            "Using placeholder 'local-model' — set --model <model-id> to match "
            "the model loaded in LM Studio.",
            file=sys.stderr,
        )

    # Load static system prompt (produced by generate_prompts.py)
    system_prompt = ""
    if args.system_prompt:
        sp_path = Path(args.system_prompt)
        if sp_path.exists():
            system_prompt = sp_path.read_text(encoding="utf-8")
        else:
            print(
                f"[WARN] System prompt file not found: {sp_path} — "
                "falling back to user-message-only mode. "
                "Run generate_prompts.py first to create it.",
                file=sys.stderr,
            )

    # Output folder and annotator label are both derived from the model name
    # so each model gets its own subfolder automatically.
    annotator = args.annotator or model_to_folder(model)
    output_dir = Path(args.output_dir) / annotator
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOKI LLM Annotation Runner")
    print("=" * 60)
    if args.create_templates:
        print("  Mode        : CREATE TEMPLATES (no API calls)")
    else:
        print(f"  Provider    : {args.provider}")
        print(f"  Model       : {model}")
        print(f"  Base URL    : {base_url}")
        if system_prompt:
            print(f"  Sys prompt  : {args.system_prompt} ({len(system_prompt.split())} words, cached)")
        else:
            print(f"  Sys prompt  : none (full prompt sent as user message)")
        print(f"  Annotator   : {annotator}")
        print(f"  Output      : {output_dir}")
    if args.dry_run:
        print("  Mode        : DRY RUN (no API calls)")
    if args.force:
        print("  Force       : overwriting existing outputs")
    print()

    # Collect files
    prompt_files = collect_prompt_files(
        input_dir=args.input_dir if not args.input_file else None,
        input_file=args.input_file,
        admission_id=args.admission_id,
    )

    print(f"[FILES] Found {len(prompt_files)} prompt file(s) to process\n")

    # ------------------------------------------------------------------
    # --create_templates: write skeleton JSONs and exit early
    # ------------------------------------------------------------------
    if args.create_templates:
        templates_dir = Path(args.templates_dir)
        templates_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TEMPLATES] Writing skeleton annotation files to: {templates_dir}\n")
        templated = 0
        skipped   = 0
        for idx, pf in enumerate(prompt_files, start=1):
            # Derive the expected output name to check for existing file
            ids = parse_ids_from_prompt(pf)
            pid = ids.get("patient_id", "") or derive_ids_from_filename(pf)[0]
            aid = ids.get("admission_id", "") or derive_ids_from_filename(pf)[1]
            fname = (
                f"annotation_{pid}_{aid}.json"
                if pid and aid
                else pf.stem + ".json"
            )
            out = templates_dir / fname
            if out.exists() and not args.force:
                print(f"  [{idx}/{len(prompt_files)}] SKIP (exists) {fname}")
                skipped += 1
                continue
            written = create_annotation_template(pf, templates_dir)
            print(f"  [{idx}/{len(prompt_files)}] WROTE {written.name}")
            templated += 1
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Templated : {templated}")
        print(f"  Skipped   : {skipped}")
        print(f"  Total     : {len(prompt_files)}")
        print()
        print(
            f"[NEXT STEP] Open files in {templates_dir}/, fill in row_grounding "
            "and relationships, then place them in Annotations/Individual/<your_name>/"
        )
        return

    # Processing loop
    counts: Dict[str, int] = {"ok": 0, "skipped": 0, "dry_run": 0, "failed": 0}

    for idx, prompt_file in enumerate(prompt_files, start=1):
        print(f"[{idx}/{len(prompt_files)}] {prompt_file.name}")
        try:
            result = process_prompt_file(
                prompt_file=prompt_file,
                provider=args.provider,
                model=model,
                output_dir=output_dir,
                annotator=annotator,
                max_retries=args.max_retries,
                base_delay=args.delay,
                force=args.force,
                dry_run=args.dry_run,
                base_url=base_url,
                system_prompt=system_prompt,
            )
            counts[result] = counts.get(result, 0) + 1

        except Exception as exc:
            counts["failed"] += 1
            print(f"  [FAILED] {exc}")

            # Save error file so the run can be diagnosed
            error_payload = {
                "prompt_file": prompt_file.name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
            patient_id_hint, admission_id_hint = derive_ids_from_filename(prompt_file)
            if patient_id_hint and admission_id_hint:
                err_name = f"annotation_{patient_id_hint}_{admission_id_hint}.error.json"
            else:
                err_name = prompt_file.stem + ".error.json"
            err_path = output_dir / err_name
            err_path.write_text(
                json.dumps(error_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  [ERROR FILE] → {err_path}")

        # Inter-call delay (skip after last file and after skips/dry-runs)
        if result == "ok" and idx < len(prompt_files) and args.delay > 0:
            time.sleep(args.delay)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Processed : {counts.get('ok', 0)}")
    print(f"  Skipped   : {counts.get('skipped', 0)}")
    print(f"  Dry-run   : {counts.get('dry_run', 0)}")
    print(f"  Failed    : {counts.get('failed', 0)}")
    print(f"  Total     : {len(prompt_files)}")
    print()
    if counts.get("ok", 0) > 0:
        print(
            f"[NEXT STEP] Run merge_annotations.py --input_dir {output_dir} "
            f"to merge into test_annotations.json"
        )


if __name__ == "__main__":
    main()
