#!/usr/bin/env python3
"""
Reviewer 1 / Round 2 / Comment 1 multi-seed orchestration runner.

Purpose
-------
Execute the prespecified predictive-variability study over seeds 42--51
for both CSE-CIC-IDS2018 and CICIoV2024 while preserving a strict,
auditable experiment contract:

  1) Generate each dataset+seed preprocessing split exactly once.
  2) Reuse that same seed-specific split for:
       - centralized XGBoost,
       - matched non-DP hierarchical FL,
       - full FL-BC-IDS DP/evidence pipeline.
  3) Enforce the compact matched FL configuration:
       2 RSUs x 2 vehicles, 2 federated rounds, 10 local rounds.
  4) Launch every child process with PYTHONHASHSEED=0.
  5) Keep all retained outputs under:
       Reviewer1_Comment1_MultiSeed/results/
     and all process logs/status files under:
       Reviewer1_Comment1_MultiSeed/logs/
  6) Fail fast on missing prerequisites, stale source revisions, invalid
   preprocessing, or missing/invalid completion artifacts; preserve and record
   independent learner failures while allowing later independent work to continue.
  7) Support safe resume without silently reusing partial output, including
   quarantine-and-retry recovery for proven transient Ray/GCS startup failures.

This runner DOES NOT compute inferential statistics. A separate statistics
script should consume only runs that this runner marks completed/validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


# =============================================================================
# Fixed Reviewer-1 Comment-1 experimental contract
# =============================================================================

DEFAULT_SEEDS = list(range(42, 52))
DEFAULT_DATASETS = ["CSECICIDS2018", "CICIoV2024"]
DEFAULT_STAGES = ["preprocess", "centralized", "nondp", "dp"]

NUM_RSUS = 2
VEHICLES_PER_RSU = 2
NUM_ROUNDS = 2
NUM_LOCAL_ROUNDS = 10

SCRIPT_NAMES = {
    "cse_preprocess": "DatasetPreprocessing-V2_Revision2_MultiSeed.py",
    "cic_preprocess": "preprocess_ciciov2024_decimal_Revision2_MultiSeed.py",
    "centralized": "centralized_xgboost_baseline_Revision2_MultiSeed.py",
    "nondp": "FL-RSUs-Server-Non-DPBaseline_V10_Revision2_MultiSeed_FINAL_MATCHED.py",
    "dp": "FL_DP_SSI_DualMerklePoseidon_RSU+Global_ZKVerify_V10_Revision2_MultiSeed.py",
}

FROZEN_DP_HELPERS = [
    "FL_IoV_AnchorZKP_UtilsV10.py",
    "FL_IoV_CanonicalSpecV10.py",
    "FL_IoV_MerkleSSI_UtilsV10.py",
    "FL_IoV_DPxgb_UtilsV10.py",
]

DEFAULT_CSE_INPUT = Path(
    os.getenv(
        "FLBCIDS_CSE_RAW_CSV",
        "data/raw/CSE-CIC-IDS2018/CSECICIDS2018Dataset.csv",
    )
)

DEFAULT_CIC_DECIMAL_DIR = Path(
    os.getenv(
        "FLBCIDS_CICIOV_RAW_DIR",
        "data/raw/CICIoV2024",
    )
)

CIC_EXPECTED_RAW_FILES = [
    "decimal_benign.csv",
    "decimal_DoS.csv",
    "decimal_spoofing-GAS.csv",
    "decimal_spoofing-RPM.csv",
    "decimal_spoofing-SPEED.csv",
    "decimal_spoofing-STEERING_WHEEL.csv",
]

PRINCIPAL_METRICS = ["accuracy", "precision", "recall", "f1", "auc"]

# Exact immutable metadata for the CSE-CIC-IDS2018 source used by the
# completed Round-2 multi-seed runs. The row/count metadata is trusted only after
# preflight recomputes the raw-file SHA-256 and verifies this exact digest.
CSE_SOURCE_SHA256 = (
    "4335539845e880b1fb06703b5a68da0a03ed0682204bdda0863ddfc316782e3c"
)
CSE_SOURCE_ROWS = 63_195_145
CSE_SOURCE_LABEL0_COUNT = 59_353_486
CSE_SOURCE_LABEL1_COUNT = 3_841_659

# Environment keys consumed by the optimized CSE preprocessing child.
CSE_ENV_SHA256 = "REVIEWER1_CSE_SOURCE_SHA256"
CSE_ENV_ROWS = "REVIEWER1_CSE_SOURCE_ROWS"
CSE_ENV_LABEL0 = "REVIEWER1_CSE_LABEL0_COUNT"
CSE_ENV_LABEL1 = "REVIEWER1_CSE_LABEL1_COUNT"

# Infrastructure-only recovery policy. A failed FL child is retried only when
# its log proves that Ray failed during local GCS startup. Before retrying, the
# entire canonical stage output is quarantined so no partial model/evidence
# state can contaminate the fresh attempt.
TRANSIENT_RAY_STAGE_MAX_ATTEMPTS = 3
TRANSIENT_RAY_RETRY_BACKOFF_SEC = 5.0
TRANSIENT_RAY_STARTUP_EXHAUSTED_MARKER = "ray_startup_failure_exhausted"


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class Paths:
    runner_path: Path
    code_dir: Path
    revision_dir: Path
    results_dir: Path
    logs_dir: Path
    statistics_dir: Path
    project_root: Path


@dataclass
class StageResult:
    dataset: str
    seed: int
    stage: str
    status: str
    returncode: int | None
    started_utc: str
    ended_utc: str
    elapsed_sec: float
    log_path: str
    validation: Dict[str, Any]
    command: List[str]


# =============================================================================
# Generic helpers
# =============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_is_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def require_nonempty_files(paths: Sequence[Path]) -> None:
    missing = [str(p) for p in paths if not file_is_nonempty(p)]
    if missing:
        raise RuntimeError(
            "Required output file(s) are missing or empty:\n  - "
            + "\n  - ".join(missing)
        )


def parse_seed_spec(spec: str) -> List[int]:
    """
    Parse forms such as:
      42-51
      42,43,47
      42-45,48,50-51
    """
    values: List[int] = []
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            a = int(left.strip())
            b = int(right.strip())
            if b < a:
                raise ValueError(f"Invalid descending seed range: {token}")
            values.extend(range(a, b + 1))
        else:
            values.append(int(token))

    deduped: List[int] = []
    seen = set()
    for v in values:
        if v not in seen:
            deduped.append(v)
            seen.add(v)

    if not deduped:
        raise ValueError("No seeds were selected.")
    return deduped


def normalize_datasets(values: Sequence[str]) -> List[str]:
    if not values:
        return list(DEFAULT_DATASETS)

    out: List[str] = []
    for value in values:
        v = str(value).strip()
        if v.lower() == "all":
            return list(DEFAULT_DATASETS)
        if v not in DEFAULT_DATASETS:
            raise ValueError(
                f"Unknown dataset {v!r}; expected one of {DEFAULT_DATASETS} or 'all'."
            )
        if v not in out:
            out.append(v)
    return out


def normalize_stages(values: Sequence[str]) -> List[str]:
    if not values:
        return list(DEFAULT_STAGES)

    allowed = set(DEFAULT_STAGES)
    out: List[str] = []
    for value in values:
        v = str(value).strip().lower()
        if v == "all":
            return list(DEFAULT_STAGES)
        if v not in allowed:
            raise ValueError(
                f"Unknown stage {v!r}; expected one of {DEFAULT_STAGES} or 'all'."
            )
        if v not in out:
            out.append(v)

    # Always honor the canonical order regardless of CLI ordering.
    return [s for s in DEFAULT_STAGES if s in out]


def helper_dir_for_root(project_root: Path) -> Path | None:
    """Return the V10 helper directory for public or legacy repository layouts."""
    public_core = project_root / "code" / "core"
    if all((public_core / name).is_file() for name in FROZEN_DP_HELPERS):
        return public_core.resolve()

    if all((project_root / name).is_file() for name in FROZEN_DP_HELPERS):
        return project_root.resolve()

    return None


def discover_project_root(start: Path) -> Path:
    """
    Locate the FL-BC-IDS repository/project root.

    Public archive layout:
      <repo>/code/core/FL_IoV_*.py

    Historical development layout:
      <project>/FL_IoV_*.py
    """
    override = os.getenv("FLBCIDS_REPO_ROOT", "").strip()
    candidates: List[Path] = []

    if override:
        candidates.append(Path(override).expanduser())

    candidates.extend([start, *start.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)

        if helper_dir_for_root(resolved) is not None:
            return resolved

    raise RuntimeError(
        "Could not locate the FL-BC-IDS repository/project root while walking "
        f"upward from {start}. Expected the four V10 helpers either under "
        "code/core/ (public archive) or directly in the project root (legacy). "
        "Set FLBCIDS_REPO_ROOT to override discovery."
    )


def resolve_node_modules_dir(project_root: Path) -> Path:
    """Locate Node dependencies in the public environment/ or legacy root."""
    overrides = (
        os.getenv("FLBCIDS_NODE_MODULES", "").strip(),
        os.getenv("ANCHOR_ZKP_NODE_MODULES", "").strip(),
    )
    candidates: List[Path] = [
        Path(value).expanduser() for value in overrides if value
    ]
    candidates.extend(
        [
            project_root / "environment" / "node_modules",
            project_root / "node_modules",
        ]
    )

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir():
            return resolved

    raise FileNotFoundError(
        "node_modules not found. Run `npm ci` in environment/ or set "
        "FLBCIDS_NODE_MODULES / ANCHOR_ZKP_NODE_MODULES."
    )

def resolve_paths() -> Paths:
    runner_path = Path(__file__).resolve()
    code_dir = runner_path.parent

    # Intended placement:
    # Reviewer1_Comment1_MultiSeed/code/Reviewer1_Comment1_Run_MultiSeed.py
    if code_dir.name.lower() != "code":
        raise RuntimeError(
            "Place this runner inside the Revision-2 'code' directory before running it. "
            f"Current directory: {code_dir}"
        )

    revision_dir = code_dir.parent
    results_dir = revision_dir / "results"
    logs_dir = revision_dir / "logs"
    statistics_dir = revision_dir / "statistics"
    project_root = discover_project_root(revision_dir)

    return Paths(
        runner_path=runner_path,
        code_dir=code_dir,
        revision_dir=revision_dir,
        results_dir=results_dir,
        logs_dir=logs_dir,
        statistics_dir=statistics_dir,
        project_root=project_root,
    )


def stage_output_dir(paths: Paths, dataset: str, seed: int, stage: str) -> Path:
    seed_root = paths.results_dir / dataset / f"seed_{seed}"
    if stage == "preprocess":
        return seed_root / "preprocessing"
    if stage == "centralized":
        return seed_root / "centralized"
    if stage == "nondp":
        return seed_root / f"nondp_local_{NUM_LOCAL_ROUNDS}"
    if stage == "dp":
        return seed_root / "full_dp"
    raise ValueError(stage)


def log_dir_for(paths: Paths, dataset: str, seed: int) -> Path:
    return paths.logs_dir / dataset / f"seed_{seed}"


def status_path_for(paths: Paths, dataset: str, seed: int) -> Path:
    return log_dir_for(paths, dataset, seed) / "stage_status.json"


def command_string(command: Sequence[str]) -> str:
    # Windows-friendly representation; subprocess still receives an argv list.
    return subprocess.list2cmdline([str(x) for x in command])


def archive_existing_file(
    path: Path,
    tag: str = "prior",
) -> Path | None:
    """
    Move an existing file to a timestamped sibling before replacing it.

    Used for per-stage logs that would otherwise be opened with mode="w".
    Completed-stage logs are never touched because resumed valid stages do not
    call the child process.
    """
    if not path.is_file():
        return None

    stamp = now_utc_compact()

    candidate = path.with_name(
        f"{path.stem}.{tag}_{stamp}{path.suffix}"
    )

    suffix = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}.{tag}_{stamp}_{suffix}{path.suffix}"
        )
        suffix += 1

    path.replace(candidate)
    return candidate


def archive_orchestration_manifests(
    paths: Paths,
) -> Dict[str, str]:
    """
    Preserve the previous top-level preflight/master manifests before a new
    orchestration session rewrites their canonical filenames.
    """
    names = (
        "preflight_manifest.json",
        "multi_seed_master_manifest.json",
    )

    existing = [
        paths.logs_dir / name
        for name in names
        if (paths.logs_dir / name).is_file()
    ]

    if not existing:
        return {}

    archive_dir = (
        paths.logs_dir
        / "orchestration_archive"
        / now_utc_compact()
    )
    ensure_dir(archive_dir)

    archived: Dict[str, str] = {}

    for source in existing:
        destination = archive_dir / source.name
        shutil.copy2(source, destination)
        archived[source.name] = str(destination)

    return archived


# =============================================================================
# Source-contract checks
# =============================================================================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def require_source_markers(path: Path, markers: Sequence[str], label: str) -> None:
    text = read_text(path)
    missing = [m for m in markers if m not in text]
    if missing:
        raise RuntimeError(
            f"{label} does not contain the expected final Revision-2 edits.\n"
            f"File: {path}\n"
            "Missing marker(s):\n  - "
            + "\n  - ".join(missing)
        )


def validate_source_contract(
    paths: Paths,
) -> Dict[str, Any]:
    """
    Refuse to start if stale pre-edit source files are still in code/.
    These checks intentionally look only for decisive contract markers.
    """
    scripts = {k: paths.code_dir / v for k, v in SCRIPT_NAMES.items()}

    for label, path in scripts.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Revision-2 source file for {label}: {path}"
            )

    # CSE preprocessing: smoke test must be disabled for the multi-seed study.
    require_source_markers(
        scripts["cse_preprocess"],
        [
            "RUN_XGB_SMOKETEST = False",
            "f\"seed_{RANDOM_STATE}\"",
            "experiments/02_multiseed/results",
            "REVIEWER1_CSE_SOURCE_SHA256",
            "REVIEWER1_CSE_SOURCE_ROWS",
        ],
        "CSE preprocessing script",
    )

    # CIC preprocessing: seed-specific output and oversampling.
    require_source_markers(
        scripts["cic_preprocess"],
        [
            "random_state=RANDOM_STATE",
            "f\"seed_{RANDOM_STATE}\"",
            "experiments/02_multiseed/results",
            "FLBCIDS_MULTI_SEED_RESULTS_DIR",
        ],
        "CICIoV2024 preprocessing script",
    )

    # Centralized: same seed-specific split plus dynamic learner seed.
    require_source_markers(
        scripts["centralized"],
        [
            'params["seed"] = int(RANDOM_STATE)',
            'params["random_state"] = int(RANDOM_STATE)',
            "f\"seed_{RANDOM_STATE}\"",
            "/ \"centralized\"",
        ],
        "Centralized baseline",
    )

    # Non-DP: matched learner, exact two-stage hierarchy, validation-only
    # distributed evaluation, fail-hard behavior.
    require_source_markers(
        scripts["nondp"],
        [
            '"objective": "reg:squarederror"',
            '"tree_method": "hist"',
            '"learning_rate": 0.2',
            '"min_child_weight": 500',
            '"subsample": 0.2',
            "pos_chunks_rsu = np.array_split",
            "neg_chunks_rsu = np.array_split",
            "val_logloss = float(",
            'self.logger.exception("fit failed")',
            'self.logger.exception("evaluate failed")',
            "RAY_STARTUP_MAX_ATTEMPTS = 3",
            "RAY_STARTUP_FAILURE_EXHAUSTED",
            "_run_simulation_hardened(",
            '"address": "local"',
            "default=10",
        ],
        "Matched non-DP hierarchical baseline",
    )

    # DP: project-root fix, experimental learner seed, fixed anchor sampling seed.
    require_source_markers(
        scripts["dp"],
        [
            "def _find_project_root(",
            "sys.path.insert(0, _helper_dir_str)",
            '"random_state": EXPERIMENT_SEED',
            '"seed": EXPERIMENT_SEED',
            "random_state=EXPERIMENT_SEED",
            "ANCHOR_SEED = 42",
            "RAY_STARTUP_MAX_ATTEMPTS = 3",
            "RAY_STARTUP_FAILURE_EXHAUSTED",
            "_run_simulation_hardened(",
            '"address": "local"',
            'f"seed_{EXPERIMENT_SEED}"',
            '"full_dp"',
        ],
        "FL-BC-IDS DP driver",
    )

    hashes = {
        key: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for key, path in scripts.items()
    }

    return {
        "scripts": hashes,
        "contract": {
            "seeds_default": DEFAULT_SEEDS,
            "datasets_default": DEFAULT_DATASETS,
            "num_rsus": NUM_RSUS,
            "vehicles_per_rsu": VEHICLES_PER_RSU,
            "num_rounds": NUM_ROUNDS,
            "num_local_rounds": NUM_LOCAL_ROUNDS,
            "pythondeterminism": "PYTHONHASHSEED=0 in every child process",
        },
    }


# =============================================================================
# Preflight checks
# =============================================================================

def find_executable(name: str, extra_dirs: Sequence[Path] = ()) -> str:
    for d in extra_dirs:
        for suffix in ("", ".exe", ".cmd", ".bat"):
            p = d / f"{name}{suffix}"
            if p.is_file():
                return str(p.resolve())
    found = shutil.which(name)
    return str(Path(found).resolve()) if found else ""


def run_quick_command(command: Sequence[str], env: Dict[str, str]) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            [str(x) for x in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=60,
            shell=False,
        )
        return int(proc.returncode), proc.stdout or ""
    except Exception as exc:
        return 999, repr(exc)


def build_child_env(paths: Paths) -> Dict[str, str]:
    env = os.environ.copy()

    # Critical: set BEFORE each child interpreter starts.
    env["PYTHONHASHSEED"] = "0"
    env["FLWR_TELEMETRY_ENABLED"] = "0"

    # Keep helper/toolchain resolution stable for the full DP driver and Ray workers.
    helper_dir = helper_dir_for_root(paths.project_root)
    if helper_dir is None:
        raise RuntimeError(
            f"Could not resolve V10 helper directory under {paths.project_root}"
        )

    node_modules = resolve_node_modules_dir(paths.project_root)
    node_bin = node_modules / ".bin"

    env["FLBCIDS_REPO_ROOT"] = str(paths.project_root)
    env["ANCHOR_ZKP_PROJECT_ROOT"] = str(paths.project_root)
    env["FLBCIDS_NODE_MODULES"] = str(node_modules)
    env["ANCHOR_ZKP_NODE_MODULES"] = str(node_modules)
    env["ANCHOR_ZKP_CIRCOM_LINK_LIBS"] = str(node_modules)
    env["CIRCOM_LINK_LIBS"] = str(node_modules)

    # Make the helpers importable to child interpreters/Ray workers even when
    # they live under code/core in the public archive.
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(helper_dir)
        if not existing_pythonpath
        else str(helper_dir) + os.pathsep + existing_pythonpath
    )

    if node_bin.is_dir():
        env["PATH"] = str(node_bin) + os.pathsep + env.get("PATH", "")

    return env


def check_python_imports(
    python_exe: Path,
    modules: Sequence[str],
    env: Dict[str, str],
) -> Dict[str, Any]:
    code = (
        "import importlib, json\n"
        f"mods={list(modules)!r}\n"
        "out={}\n"
        "for m in mods:\n"
        "    try:\n"
        "        mod=importlib.import_module(m)\n"
        "        out[m]={'ok':True,'version':str(getattr(mod,'__version__',''))}\n"
        "    except Exception as e:\n"
        "        out[m]={'ok':False,'error':repr(e)}\n"
        "print(json.dumps(out))\n"
        "raise SystemExit(0 if all(v['ok'] for v in out.values()) else 2)\n"
    )
    rc, out = run_quick_command([str(python_exe), "-c", code], env)
    parsed: Dict[str, Any] = {}

    # Some imported libraries can emit terminal-control or informational
    # lines after the JSON result. Search backward for the last valid JSON
    # object instead of assuming stdout's final line is JSON.
    for raw_line in reversed(out.splitlines()):
        candidate = raw_line.strip()

        if not candidate:
            continue

        try:
            obj = json.loads(candidate)
        except Exception:
            continue

        if isinstance(obj, dict):
            parsed = obj
            break

    if not parsed:
        parsed = {
            "raw": out,
        }

    if rc != 0:
        raise RuntimeError(
            "Python package preflight failed for the selected interpreter.\n"
            f"Interpreter: {python_exe}\n"
            f"Result: {parsed}"
        )
    return parsed

def check_ray_local_runtime(
    python_exe: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    """
    Verify in a disposable child interpreter that a fresh local Ray runtime can
    start and shut down cleanly with the same lifecycle policy used by the FL
    drivers. No experiment artifacts are created or modified.
    """
    code = (
        "import json, ray\n"
        "out={'ok':False,'ray_version':str(getattr(ray,'__version__',''))}\n"
        "ctx=None\n"
        "try:\n"
        "    ctx=ray.init(address='local', include_dashboard=False, "
        "num_cpus=1, num_gpus=0, log_to_driver=False, logging_level=40)\n"
        "    out['ok']=bool(ray.is_initialized())\n"
        "finally:\n"
        "    try:\n"
        "        ray.shutdown()\n"
        "    except Exception as e:\n"
        "        out['shutdown_error']=repr(e)\n"
        "print(json.dumps(out))\n"
    )

    rc, out = run_quick_command(
        [str(python_exe), "-c", code],
        env,
    )

    # Ray/Windows can append terminal-control/ANSI lines after our JSON.
    # Do not assume that the final stdout line is the JSON result.
    parsed: Dict[str, Any] = {}

    for raw_line in reversed(out.splitlines()):
        candidate = raw_line.strip()

        if not candidate:
            continue

        try:
            obj = json.loads(candidate)
        except Exception:
            continue

        if isinstance(obj, dict):
            parsed = obj
            break

    if not parsed:
        parsed = {
            "ok": False,
            "raw": out,
        }

    if rc != 0 or not bool(parsed.get("ok", False)):
        raise RuntimeError(
            "Ray local-runtime preflight failed. The FL stages are not being "
            "started because a disposable fresh local Ray instance could not "
            "start and shut down cleanly.\n"
            f"Interpreter: {python_exe}\n"
            f"Result: {parsed}\n"
            f"Raw output: {out[-4000:]}"
        )

    return parsed

def preflight(
    paths: Paths,
    python_exe: Path,
    cse_input: Path,
    cic_decimal_dir: Path,
    datasets: Sequence[str],
    stages: Sequence[str],
    source_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    ensure_dir(paths.results_dir)
    ensure_dir(paths.logs_dir)
    ensure_dir(paths.statistics_dir)

    if not python_exe.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python_exe}")

    cse_source: Dict[str, Any] = {}

    if "CSECICIDS2018" in datasets and "preprocess" in stages:
        if not cse_input.is_file():
            raise FileNotFoundError(
                f"CSE-CIC-IDS2018 input CSV not found: {cse_input}"
            )

        # Hash the huge raw CSE source exactly once per orchestration session.
        # The optimized preprocessing children consume the verified metadata
        # instead of rescanning the source merely to recompute invariant
        # SHA/row/label diagnostics for every seed.
        print(
            "Preflight: verifying CSE-CIC-IDS2018 raw source SHA-256 once..."
        )

        cse_sha256 = sha256_file(cse_input)

        if cse_sha256.lower() != CSE_SOURCE_SHA256.lower():
            raise RuntimeError(
                "CSE-CIC-IDS2018 raw source SHA-256 mismatch. "
                "The cached row/label metadata must not be used with a "
                "different source file.\n"
                f"Expected: {CSE_SOURCE_SHA256}\n"
                f"Actual:   {cse_sha256}"
            )

        cse_source = {
            "path": str(cse_input),
            "sha256": cse_sha256,
            "rows": int(CSE_SOURCE_ROWS),
            "label_counts": {
                "0": int(CSE_SOURCE_LABEL0_COUNT),
                "1": int(CSE_SOURCE_LABEL1_COUNT),
            },
            "metadata_trust_basis": (
                "row and label counts are reused only after exact raw-file "
                "SHA-256 verification"
            ),
        }

    if "CICIoV2024" in datasets and "preprocess" in stages:
        if not cic_decimal_dir.is_dir():
            raise FileNotFoundError(
                f"CICIoV2024 decimal directory not found: {cic_decimal_dir}"
            )
        missing_raw = [
            name for name in CIC_EXPECTED_RAW_FILES
            if not (cic_decimal_dir / name).is_file()
        ]
        if missing_raw:
            raise FileNotFoundError(
                "CICIoV2024 is missing expected raw decimal files:\n  - "
                + "\n  - ".join(missing_raw)
            )

    env = build_child_env(paths)

    # Base packages needed by preprocessing + references.
    packages = [
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "imblearn",
    ]
    if "preprocess" in stages and "CSECICIDS2018" in datasets:
        packages.extend(["dask", "scipy"])
    if "nondp" in stages or "dp" in stages:
        packages.extend(["flwr", "ray"])
    if "dp" in stages:
        packages.extend(["dp_xgboost", "cryptography"])

    packages = list(dict.fromkeys(packages))
    package_info = check_python_imports(
        python_exe,
        packages,
        env,
    )

    ray_runtime_preflight: Dict[str, Any] = {}

    if "nondp" in stages or "dp" in stages:
        print(
            "Preflight: checking fresh local Ray startup/shutdown..."
        )

        ray_runtime_preflight = check_ray_local_runtime(
            python_exe,
            env,
        )

        print(
            "Preflight: Ray local runtime PASS"
        )

    dp_toolchain: Dict[str, Any] = {}
    if "dp" in stages:
        helper_dir = helper_dir_for_root(paths.project_root)
        if helper_dir is None:
            raise FileNotFoundError(
                "Required V10 DP helper directory was not found under "
                f"{paths.project_root}."
            )

        for helper in FROZEN_DP_HELPERS:
            p = helper_dir / helper
            if not p.is_file():
                raise FileNotFoundError(f"Required frozen DP helper not found: {p}")

        node_modules = resolve_node_modules_dir(paths.project_root)
        local_bin = node_modules / ".bin"

        def _tool_with_override(tool_name: str, env_key: str) -> str:
            override = env.get(env_key, "").strip()
            if override:
                override_path = Path(override).expanduser()
                if override_path.is_file():
                    return str(override_path.resolve())
                found_override = shutil.which(override)
                if found_override:
                    return str(Path(found_override).resolve())
                raise RuntimeError(
                    f"{env_key} is set but does not resolve to an executable: {override}"
                )
            return find_executable(tool_name, [local_bin])

        circom = _tool_with_override("circom", "ANCHOR_ZKP_CIRCOM_CMD")
        snarkjs = _tool_with_override("snarkjs", "ANCHOR_ZKP_SNARKJS_CMD")
        node = _tool_with_override("node", "ANCHOR_ZKP_NODE_CMD")

        missing_tools = [
            name for name, value in
            [("circom", circom), ("snarkjs", snarkjs), ("node", node)]
            if not value
        ]
        if missing_tools:
            raise RuntimeError(
                "Missing DP/ZKP executable(s): "
                + ", ".join(missing_tools)
                + ". They must be available on PATH or in the resolved node_modules/.bin directory."
            )

        # The frozen helper resolves PTAU next to itself first, then project root,
        # or via ANCHOR_ZKP_PTAU_PATH.
        ptau_override = env.get(
            "ANCHOR_ZKP_PTAU_PATH",
            "",
        ).strip()

        ptau_candidates: List[Path] = []

        if ptau_override:
            ptau_candidates.append(
                Path(ptau_override)
            )

        ptau_filename = "powersOfTau28_hez_final_20.ptau"
        ptau_candidates.extend(
            [
                helper_dir / ptau_filename,
                paths.project_root / ptau_filename,
                paths.project_root / "verification" / "groth16" / ptau_filename,
                paths.project_root
                / "verification"
                / "groth16"
                / "canonical"
                / ptau_filename,
            ]
        )

        ptau_path_optional: Path | None = next(
            (
                candidate.resolve()
                for candidate in ptau_candidates
                if candidate.is_file()
            ),
            None,
        )

        if ptau_path_optional is None:
            raise FileNotFoundError(
                "powersOfTau28_hez_final_20.ptau was not found "
                "in the public/legacy repository candidates and ANCHOR_ZKP_PTAU_PATH "
                "does not point to a valid file."
            )

        ptau_path: Path = ptau_path_optional

        dp_toolchain = {
            "node": node,
            "circom": circom,
            "snarkjs": snarkjs,
            "ptau": str(ptau_path),
            "ptau_sha256": sha256_file(ptau_path),
            "frozen_helpers": {
                name: {
                    "path": str(helper_dir / name),
                    "sha256": sha256_file(helper_dir / name),
                }
                for name in FROZEN_DP_HELPERS
            },
        }

    manifest = {
        "schema": "Reviewer1Comment1MultiSeedPreflightV1",
        "created_utc": now_utc(),
        "runner": {
            "path": str(paths.runner_path),
            "sha256": sha256_file(paths.runner_path),
        },
        "python_executable": str(python_exe.resolve()),
        "python_packages": package_info,
        "ray_runtime_preflight": ray_runtime_preflight,
        "paths": {
            "revision_dir": str(paths.revision_dir),
            "code_dir": str(paths.code_dir),
            "results_dir": str(paths.results_dir),
            "logs_dir": str(paths.logs_dir),
            "statistics_dir": str(paths.statistics_dir),
            "project_root": str(paths.project_root),
            "helper_dir": str(helper_dir_for_root(paths.project_root) or ""),
            "node_modules_dir": str(resolve_node_modules_dir(paths.project_root))
            if ("nondp" in stages or "dp" in stages)
            else "",
            "cse_input": str(cse_input),
            "cic_decimal_dir": str(cic_decimal_dir),
        },
        "selected": {
            "datasets": list(datasets),
            "stages": list(stages),
        },
        "source_manifest": source_manifest,
        "cse_source": cse_source,
        "dp_toolchain": dp_toolchain,
    }

    atomic_write_json(paths.logs_dir / "preflight_manifest.json", manifest)
    return manifest


# =============================================================================
# Output validation
# =============================================================================

def validate_preprocess(paths: Paths, dataset: str, seed: int) -> Dict[str, Any]:
    out = stage_output_dir(paths, dataset, seed, "preprocess")

    if dataset == "CSECICIDS2018":
        train = out / "CSECICIDS2018_train_preprocessed.csv"
        val = out / "CSECICIDS2018_val_preprocessed.csv"
        test = out / "CSECICIDS2018_test_preprocessed.csv"
        manifest_path = out / "preprocessing_manifest.json"
    else:
        train = out / "CICIoV2024_train_preprocessed.csv"
        val = out / "CICIoV2024_val_preprocessed.csv"
        test = out / "CICIoV2024_test_preprocessed.csv"
        manifest_path = out / "CICIoV2024_manifest.json"

    require_nonempty_files([train, val, test, manifest_path])

    manifest = load_json(manifest_path)
    manifest_seed = manifest.get("random_state", None)
    if int(manifest_seed) != int(seed):
        raise RuntimeError(
            f"Preprocessing manifest seed mismatch for {dataset}: "
            f"expected {seed}, got {manifest_seed}"
        )

    hashes = {
        "train_csv_sha256": sha256_file(train),
        "val_csv_sha256": sha256_file(val),
        "test_csv_sha256": sha256_file(test),
    }

    return {
        "ok": True,
        "output_dir": str(out),
        "manifest": str(manifest_path),
        "seed": seed,
        "split_hashes": hashes,
        "files": {
            "train": str(train),
            "val": str(val),
            "test": str(test),
        },
    }


def validate_run_config(
    cfg: Dict[str, Any],
    dataset: str,
    seed: int,
    *,
    dp_expected: bool,
) -> None:
    if str(cfg.get("dataset_name", "")) != dataset:
        raise RuntimeError(
            f"Run config dataset mismatch: expected {dataset}, "
            f"got {cfg.get('dataset_name')!r}"
        )
    if int(cfg.get("experiment_seed", -999999)) != int(seed):
        raise RuntimeError(
            f"Run config seed mismatch: expected {seed}, "
            f"got {cfg.get('experiment_seed')!r}"
        )
    if int(cfg.get("num_rsus", -1)) != NUM_RSUS:
        raise RuntimeError("Run config num_rsus is not the fixed matched value 2.")
    if int(cfg.get("vehicles_per_rsu", -1)) != VEHICLES_PER_RSU:
        raise RuntimeError(
            "Run config vehicles_per_rsu is not the fixed matched value 2."
        )
    if int(cfg.get("num_rounds", -1)) != NUM_ROUNDS:
        raise RuntimeError("Run config num_rounds is not the fixed matched value 2.")
    if int(cfg.get("num_local_rounds", -1)) != NUM_LOCAL_ROUNDS:
        raise RuntimeError(
            "Run config num_local_rounds is not the fixed matched value 10."
        )

    if bool(cfg.get("dp_enabled", False)) != bool(dp_expected):
        raise RuntimeError(
            f"Run config dp_enabled mismatch: expected {dp_expected}, "
            f"got {cfg.get('dp_enabled')!r}"
        )


def validate_metric_dict(metrics: Dict[str, Any], label: str) -> None:
    for key in PRINCIPAL_METRICS:
        if key not in metrics:
            raise RuntimeError(f"{label} is missing principal metric {key!r}.")
        value = float(metrics[key])
        if not (0.0 <= value <= 1.0):
            raise RuntimeError(
                f"{label}.{key} is outside [0,1]: {value}"
            )


def validate_centralized(paths: Paths, dataset: str, seed: int) -> Dict[str, Any]:
    out = stage_output_dir(paths, dataset, seed, "centralized")
    summary = out / f"centralized_xgboost_{dataset}_summary.json"
    run_summary = out / "centralized_xgboost_run_summary.json"
    model = out / f"centralized_xgboost_{dataset}.json"

    require_nonempty_files([summary, run_summary, model])

    obj = load_json(summary)
    if str(obj.get("dataset", "")) != dataset:
        raise RuntimeError("Centralized summary dataset mismatch.")
    if int(obj.get("experiment_seed", -1)) != seed:
        raise RuntimeError("Centralized summary seed mismatch.")

    test_metrics = obj.get("test_metrics", {})
    if not isinstance(test_metrics, dict):
        raise RuntimeError("Centralized test_metrics is not a dictionary.")
    validate_metric_dict(test_metrics, "centralized.test_metrics")

    return {
        "ok": True,
        "output_dir": str(out),
        "summary": str(summary),
        "model": str(model),
        "test_metrics": {
            k: float(test_metrics[k])
            for k in PRINCIPAL_METRICS
        },
    }


def validate_nondp(paths: Paths, dataset: str, seed: int) -> Dict[str, Any]:
    out = stage_output_dir(paths, dataset, seed, "nondp")
    cfg_path = out / "iov_run_config.json"
    summary = out / "iov_global_server_ensemble_summary.json"
    models = [
        out / f"iov_global_model_rsu_{rsu_id}.json"
        for rsu_id in range(1, NUM_RSUS + 1)
    ]

    require_nonempty_files([cfg_path, summary, *models])

    cfg = load_json(cfg_path)
    validate_run_config(cfg, dataset, seed, dp_expected=False)

    obj = load_json(summary)
    metrics = obj.get("ensemble_metrics", {})
    if not isinstance(metrics, dict):
        raise RuntimeError("Non-DP ensemble_metrics is not a dictionary.")
    validate_metric_dict(metrics, "nondp.ensemble_metrics")

    if int(obj.get("num_rsus_used", 0)) != NUM_RSUS:
        raise RuntimeError(
            f"Non-DP final ensemble used {obj.get('num_rsus_used')} RSUs; "
            f"expected {NUM_RSUS} for this controlled study."
        )

    return {
        "ok": True,
        "output_dir": str(out),
        "run_config": str(cfg_path),
        "summary": str(summary),
        "test_metrics": {
            k: float(metrics[k])
            for k in PRINCIPAL_METRICS
        },
    }


def validate_dp(paths: Paths, dataset: str, seed: int) -> Dict[str, Any]:
    out = stage_output_dir(paths, dataset, seed, "dp")
    cfg_path = out / "iov_run_config.json"
    summary = out / "global_ensemble_summary.json"
    gate = out / "zkp_anchor_gating_manifest.json"
    models = [
        out / f"iov_global_model_rsu_{rsu_id}.json"
        for rsu_id in range(1, NUM_RSUS + 1)
    ]

    require_nonempty_files([cfg_path, summary, gate, *models])

    cfg = load_json(cfg_path)
    validate_run_config(cfg, dataset, seed, dp_expected=True)

    obj = load_json(summary)
    metrics = obj.get("ensemble_metrics", {})
    if not isinstance(metrics, dict):
        raise RuntimeError("DP global ensemble_metrics is not a dictionary.")
    validate_metric_dict(metrics, "dp.ensemble_metrics")

    gate_obj = load_json(gate)
    if not bool(gate_obj.get("global_anchor_ok", False)):
        raise RuntimeError(
            "DP learning run finished but the GLOBAL AnchorZKP gate is not valid. "
            "For the full-system multi-seed execution this stage is not accepted "
            "as complete."
        )

    used = gate_obj.get("used_rsu_ids", [])
    if not isinstance(used, list) or sorted(int(x) for x in used) != [1, 2]:
        raise RuntimeError(
            f"DP GLOBAL gate did not include both controlled-study RSUs: {used!r}"
        )

    return {
        "ok": True,
        "output_dir": str(out),
        "run_config": str(cfg_path),
        "summary": str(summary),
        "gating_manifest": str(gate),
        "test_metrics": {
            k: float(metrics[k])
            for k in PRINCIPAL_METRICS
        },
        "global_anchor_ok": True,
        "used_rsu_ids": [int(x) for x in used],
    }


def validate_stage(
    paths: Paths,
    dataset: str,
    seed: int,
    stage: str,
) -> Dict[str, Any]:
    if stage == "preprocess":
        return validate_preprocess(paths, dataset, seed)
    if stage == "centralized":
        return validate_centralized(paths, dataset, seed)
    if stage == "nondp":
        return validate_nondp(paths, dataset, seed)
    if stage == "dp":
        return validate_dp(paths, dataset, seed)
    raise ValueError(stage)


def get_preprocess_validation(
    cache: Dict[Tuple[str, int], Dict[str, Any]],
    paths: Paths,
    dataset: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Fully validate/hash one seed's preprocessing artifacts at most once during
    this orchestration session, then reuse the immutable validation record for
    centralized, non-DP, and DP stages.
    """
    key = (dataset, int(seed))

    cached = cache.get(key)
    if cached is not None:
        return cached

    validation = validate_preprocess(
        paths,
        dataset,
        seed,
    )

    cache[key] = validation
    return validation


# =============================================================================
# Commands
# =============================================================================

def build_command(
    paths: Paths,
    python_exe: Path,
    dataset: str,
    seed: int,
    stage: str,
    cse_input: Path,
    cic_decimal_dir: Path,
) -> List[str]:
    scripts = {k: paths.code_dir / v for k, v in SCRIPT_NAMES.items()}

    common_root = str(paths.results_dir)

    if stage == "preprocess":
        if dataset == "CSECICIDS2018":
            return [
                str(python_exe),
                str(scripts["cse_preprocess"]),
                "--seed", str(seed),
                "--revision-root", common_root,
                "--input-csv", str(cse_input),
            ]
        return [
            str(python_exe),
            str(scripts["cic_preprocess"]),
            "--seed", str(seed),
            "--revision-root", common_root,
            "--decimal-dir", str(cic_decimal_dir),
        ]

    if stage == "centralized":
        return [
            str(python_exe),
            str(scripts["centralized"]),
            "--dataset", dataset,
            "--seed", str(seed),
            "--revision-root", common_root,
        ]

    if stage == "nondp":
        return [
            str(python_exe),
            str(scripts["nondp"]),
            "--dataset", dataset,
            "--seed", str(seed),
            "--revision-root", common_root,
            "--num-rsus", str(NUM_RSUS),
            "--vehicles-per-rsu", str(VEHICLES_PER_RSU),
            "--num-rounds", str(NUM_ROUNDS),
            "--num-local-rounds", str(NUM_LOCAL_ROUNDS),
        ]

    if stage == "dp":
        return [
            str(python_exe),
            str(scripts["dp"]),
            "--dataset", dataset,
            "--seed", str(seed),
            "--revision-root", common_root,
            "--num-rsus", str(NUM_RSUS),
            "--vehicles-per-rsu", str(VEHICLES_PER_RSU),
            "--num-rounds", str(NUM_ROUNDS),
            "--num-local-rounds", str(NUM_LOCAL_ROUNDS),
        ]

    raise ValueError(stage)


# =============================================================================
# Status/resume helpers
# =============================================================================

def load_status_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"schema": "Reviewer1Comment1SeedStageStatusV1", "stages": {}}
    try:
        obj = load_json(path)
        if not isinstance(obj, dict):
            raise ValueError("status JSON is not an object")
        obj.setdefault("schema", "Reviewer1Comment1SeedStageStatusV1")
        obj.setdefault("stages", {})
        return obj
    except Exception:
        return {"schema": "Reviewer1Comment1SeedStageStatusV1", "stages": {}}


def update_status(
    paths: Paths,
    dataset: str,
    seed: int,
    stage_result: StageResult,
) -> None:
    path = status_path_for(paths, dataset, seed)
    obj = load_status_file(path)
    obj["dataset"] = dataset
    obj["seed"] = int(seed)
    obj["updated_utc"] = now_utc()
    obj["stages"][stage_result.stage] = {
        "status": stage_result.status,
        "returncode": stage_result.returncode,
        "started_utc": stage_result.started_utc,
        "ended_utc": stage_result.ended_utc,
        "elapsed_sec": stage_result.elapsed_sec,
        "log_path": stage_result.log_path,
        "validation": stage_result.validation,
        "command": stage_result.command,
        "command_display": command_string(stage_result.command),
    }
    atomic_write_json(path, obj)


def stage_output_nonempty(paths: Paths, dataset: str, seed: int, stage: str) -> bool:
    out = stage_output_dir(paths, dataset, seed, stage)
    return out.is_dir() and any(out.iterdir())


def safe_quarantine_stage_output(
    paths: Paths,
    dataset: str,
    seed: int,
    stage: str,
) -> Path | None:
    """
    Preserve an invalid/partial stage directory by renaming it in place.

    Nothing is deleted. A fresh canonical stage directory can then be rebuilt
    while the interrupted artifacts remain available for audit/recovery.
    """
    target = stage_output_dir(
        paths,
        dataset,
        seed,
        stage,
    ).resolve()

    results_root = paths.results_dir.resolve()

    try:
        target.relative_to(results_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to quarantine path outside results root: {target}"
        ) from exc

    if not target.exists():
        return None

    stamp = now_utc_compact()

    candidate = target.with_name(
        f"{target.name}.partial_{stamp}"
    )

    suffix = 1
    while candidate.exists():
        candidate = target.with_name(
            f"{target.name}.partial_{stamp}_{suffix}"
        )
        suffix += 1

    target.rename(candidate)
    return candidate

def log_has_transient_ray_startup_failure(
    log_path: Path,
) -> bool:
    """
    Return True only when the FL child explicitly reports that all of its
    startup-only Ray/GCS retries were exhausted.

    We intentionally require the explicit sentinel emitted by the hardened
    FL drivers. A generic 'Simulation Engine crashed' message is not enough,
    because genuine model/training failures must never be silently retried.
    """
    if not log_path.is_file():
        return False

    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).lower()

    return (
        TRANSIENT_RAY_STARTUP_EXHAUSTED_MARKER
        in text
    )

# =============================================================================
# Process execution
# =============================================================================

def run_logged_process(
    command: Sequence[str],
    log_path: Path,
    env: Dict[str, str],
    cwd: Path,
) -> int:
    ensure_dir(log_path.parent)

    archived_log = archive_existing_file(
        log_path,
        tag="prior",
    )

    if archived_log is not None:
        print(
            "Preserved previous stage log: "
            f"{archived_log}"
        )

    with log_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
    ) as log:
        header = [
            "=" * 100,
            f"Started UTC : {now_utc()}",
            f"CWD         : {cwd}",
            f"PYHASH      : {env.get('PYTHONHASHSEED')}",
            f"Command     : {command_string(command)}",
            "=" * 100,
            "",
        ]
        for line in header:
            print(line)
            log.write(line + "\n")
        log.flush()

        proc = subprocess.Popen(
            [str(x) for x in command],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        return int(proc.wait())


def execute_stage(
    *,
    paths: Paths,
    python_exe: Path,
    dataset: str,
    seed: int,
    stage: str,
    cse_input: Path,
    cic_decimal_dir: Path,
    env: Dict[str, str],
    resume: bool,
    clean_partial: bool,
    dry_run: bool,
) -> StageResult:
    command = build_command(
        paths,
        python_exe,
        dataset,
        seed,
        stage,
        cse_input,
        cic_decimal_dir,
    )

    log_path = log_dir_for(paths, dataset, seed) / f"{stage}.log"
    started = now_utc()
    t0 = time.perf_counter()

    if clean_partial and not resume:
        raise RuntimeError(
            "--clean-partial is a recovery option and is permitted only "
            "together with --resume."
        )

    # Resume only after validating the already-existing artifacts.
    if resume:
        try:
            validation = validate_stage(paths, dataset, seed, stage)
            elapsed = time.perf_counter() - t0
            return StageResult(
                dataset=dataset,
                seed=seed,
                stage=stage,
                status="skipped_valid_existing",
                returncode=0,
                started_utc=started,
                ended_utc=now_utc(),
                elapsed_sec=elapsed,
                log_path=str(log_path),
                validation=validation,
                command=command,
            )
        except Exception:
            # Existing artifacts are absent or invalid: proceed only under the
            # normal partial-output safety rule below.
            pass

    if stage_output_nonempty(paths, dataset, seed, stage):
        if clean_partial:
            quarantined = safe_quarantine_stage_output(
                paths,
                dataset,
                seed,
                stage,
            )

            print(
                "Existing invalid/partial stage was preserved at: "
                f"{quarantined}"
            )
        else:
            raise RuntimeError(
                f"Refusing to run {dataset} seed={seed} stage={stage} because its "
                f"output directory already contains files:\n"
                f"  {stage_output_dir(paths, dataset, seed, stage)}\n"
                "Use --resume if it is a completed valid run, or use "
                "--resume --clean-partial to quarantine and rebuild an "
                "incomplete stage."
            )

    if dry_run:
        elapsed = time.perf_counter() - t0
        return StageResult(
            dataset=dataset,
            seed=seed,
            stage=stage,
            status="dry_run",
            returncode=None,
            started_utc=started,
            ended_utc=now_utc(),
            elapsed_sec=elapsed,
            log_path=str(log_path),
            validation={"ok": True, "dry_run": True},
            command=command,
        )

    max_attempts = (
        TRANSIENT_RAY_STAGE_MAX_ATTEMPTS
        if stage in {"nondp", "dp"}
        else 1
    )

    attempt = 0
    transient_retry_count = 0
    rc = 999

    while attempt < max_attempts:
        attempt += 1

        if attempt > 1:
            print(
                f"Retrying fresh stage: dataset={dataset} "
                f"seed={seed} stage={stage} "
                f"attempt={attempt}/{max_attempts}"
            )

        rc = run_logged_process(
            command=command,
            log_path=log_path,
            env=env,
            cwd=paths.project_root,
        )

        if rc == 0:
            break

        transient_ray_startup = (
            stage in {"nondp", "dp"}
            and log_has_transient_ray_startup_failure(
                log_path
            )
        )

        if (
            transient_ray_startup
            and attempt < max_attempts
        ):
            transient_retry_count += 1

            quarantined = safe_quarantine_stage_output(
                paths,
                dataset,
                seed,
                stage,
            )

            print(
                "Transient Ray GCS startup failure detected. "
                "The failed attempt was preserved at: "
                f"{quarantined}"
            )

            print(
                "A fresh stage attempt will be started after "
                "the cleanup/backoff barrier."
            )

            time.sleep(
                TRANSIENT_RAY_RETRY_BACKOFF_SEC
                * float(attempt)
            )

            continue

        break

    elapsed = time.perf_counter() - t0

    if rc != 0:
        result = StageResult(
            dataset=dataset,
            seed=seed,
            stage=stage,
            status="failed_process",
            returncode=rc,
            started_utc=started,
            ended_utc=now_utc(),
            elapsed_sec=elapsed,
            log_path=str(log_path),
            validation={
                "ok": False,
                "reason": (
                    f"child process exited with rc={rc}"
                ),
                "execution_attempts": int(attempt),
                "transient_ray_retries": int(
                    transient_retry_count
                ),
            },
            command=command,
        )

        update_status(
            paths,
            dataset,
            seed,
            result,
        )

        raise RuntimeError(
            f"{dataset} seed={seed} stage={stage} "
            f"failed with exit code {rc} after "
            f"{attempt} attempt(s). See log: {log_path}"
        )

    try:
        validation = validate_stage(
            paths,
            dataset,
            seed,
            stage,
        )

        validation = dict(validation)

        validation["execution_attempts"] = int(
            attempt
        )

        validation["transient_ray_retries"] = int(
            transient_retry_count
        )

    except Exception as exc:
        result = StageResult(
            dataset=dataset,
            seed=seed,
            stage=stage,
            status="failed_validation",
            returncode=rc,
            started_utc=started,
            ended_utc=now_utc(),
            elapsed_sec=elapsed,
            log_path=str(log_path),
            validation={
                "ok": False,
                "reason": repr(exc),
                "execution_attempts": int(
                    attempt
                ),
                "transient_ray_retries": int(
                    transient_retry_count
                ),
            },
            command=command,
        )
        update_status(paths, dataset, seed, result)
        raise RuntimeError(
            f"{dataset} seed={seed} stage={stage} exited successfully but "
            f"its completion artifacts failed validation: {exc}"
        ) from exc

    return StageResult(
        dataset=dataset,
        seed=seed,
        stage=stage,
        status="completed_validated",
        returncode=rc,
        started_utc=started,
        ended_utc=now_utc(),
        elapsed_sec=elapsed,
        log_path=str(log_path),
        validation=validation,
        command=command,
    )


# =============================================================================
# Cross-method paired-input verification
# =============================================================================

def write_seed_input_contract(
    paths: Paths,
    dataset: str,
    seed: int,
    prep: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if prep is None:
        prep = validate_preprocess(
            paths,
            dataset,
            seed,
        )

    obj = {
        "schema": "Reviewer1Comment1SeedInputContractV1",
        "dataset": dataset,
        "seed": int(seed),
        "created_utc": now_utc(),
        "preprocessing_dir": prep["output_dir"],
        "split_files": prep["files"],
        "split_hashes": prep["split_hashes"],
        "paired_methods": [
            "centralized",
            "nondp",
            "dp",
        ],
        "statement": (
            "All three learning configurations are launched with this exact "
            "seed-specific preprocessing directory and therefore consume the "
            "same train/validation/test CSV artifacts."
        ),
    }

    path = (
        log_dir_for(paths, dataset, seed)
        / "paired_input_contract.json"
    )

    # Preserve an already-valid contract from an earlier orchestration session.
    if path.is_file():
        existing = load_json(path)

        if (
            isinstance(existing, dict)
            and str(existing.get("dataset", "")) == dataset
            and int(existing.get("seed", -1)) == int(seed)
            and existing.get("split_hashes") == prep["split_hashes"]
        ):
            return existing

        raise RuntimeError(
            "Existing paired_input_contract.json disagrees with the validated "
            f"preprocessing artifacts for {dataset} seed={seed}: {path}"
        )

    atomic_write_json(path, obj)
    return obj


# =============================================================================
# Run manifest
# =============================================================================

def build_plan(
    datasets: Sequence[str],
    seeds: Sequence[int],
    stages: Sequence[str],
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            for stage in stages:
                plan.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "stage": stage,
                    }
                )
    return plan


def update_master_manifest(
    path: Path,
    *,
    base: Dict[str, Any],
    results: Sequence[StageResult],
    state: str,
    error: str = "",
) -> None:
    obj = dict(base)
    obj["updated_utc"] = now_utc()
    obj["state"] = state
    obj["error"] = error
    obj["stage_results"] = [
        {
            "dataset": r.dataset,
            "seed": r.seed,
            "stage": r.stage,
            "status": r.status,
            "returncode": r.returncode,
            "started_utc": r.started_utc,
            "ended_utc": r.ended_utc,
            "elapsed_sec": r.elapsed_sec,
            "log_path": r.log_path,
            "validation": r.validation,
            "command": r.command,
        }
        for r in results
    ]
    atomic_write_json(path, obj)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Reviewer-1 Comment-1 seed-repartitioned statistical "
            "experiment over the fixed compact FL configuration."
        )
    )

    parser.add_argument(
        "--python-exe",
        type=Path,
        default=Path(sys.executable),
        help=(
            "Python interpreter used for every child process. "
            "Default: the interpreter running this runner."
        ),
    )

    parser.add_argument(
        "--seeds",
        default="42-51",
        help="Seed specification, e.g. 42-51 or 42,43,47.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=(
            "Datasets to run: CSECICIDS2018 CICIoV2024, or 'all'. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help=(
            "Stages to execute: preprocess centralized nondp dp, or 'all'. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--cse-input-csv",
        type=Path,
        default=DEFAULT_CSE_INPUT,
        help="Original CSE-CIC-IDS2018 CSV.",
    )

    parser.add_argument(
        "--cic-decimal-dir",
        type=Path,
        default=DEFAULT_CIC_DECIMAL_DIR,
        help="Directory containing the six CICIoV2024 decimal CSV files.",
    )

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume safely by default. A stage is skipped only if its completion "
            "artifacts pass validation. Use --no-resume only for a deliberately "
            "fresh invocation."
        ),
    )

    parser.add_argument(
        "--clean-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Safe recovery is enabled by default. With resume enabled, any "
            "non-empty but invalid stage directory is preserved under a "
            "timestamped *.partial_* directory and the canonical stage directory "
            "is rebuilt. Use --no-clean-partial to disable automatic quarantine."
        ),
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run source/toolchain/input checks and exit without experiments.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate preflight and print/record commands without executing them.",
    )

    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Continue independent work by default after a non-preprocessing "
            "stage failure, so one infrastructure fault cannot terminate the "
            "remaining multi-seed batch. Use --no-continue-on-error for "
            "deliberate fail-fast behavior."
        ),
    )

    args = parser.parse_args()

    if args.clean_partial and not args.resume:
        parser.error(
            "--clean-partial may only be used together with --resume. "
            "Valid completed stages are always validated and skipped first."
        )

    return args


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    paths = resolve_paths()
    seeds = parse_seed_spec(args.seeds)
    datasets = normalize_datasets(args.datasets)
    stages = normalize_stages(args.stages)
    python_exe = Path(args.python_exe).expanduser().resolve()
    cse_input = Path(args.cse_input_csv).expanduser().resolve()
    cic_decimal_dir = Path(args.cic_decimal_dir).expanduser().resolve()

    ensure_dir(paths.results_dir)
    ensure_dir(paths.logs_dir)
    ensure_dir(paths.statistics_dir)

    print("=" * 100)
    print("REVIEWER 1 / ROUND 2 / COMMENT 1 — MULTI-SEED ORCHESTRATOR")
    print("=" * 100)
    print(f"Revision dir : {paths.revision_dir}")
    print(f"Results dir  : {paths.results_dir}")
    print(f"Logs dir     : {paths.logs_dir}")
    print(f"Project root : {paths.project_root}")
    print(f"Python       : {python_exe}")
    print(f"Datasets     : {datasets}")
    print(f"Seeds        : {seeds}")
    print(f"Stages       : {stages}")
    print(
        "FL contract  : "
        f"{NUM_RSUS} RSUs x {VEHICLES_PER_RSU} vehicles, "
        f"{NUM_ROUNDS} rounds, {NUM_LOCAL_ROUNDS} local rounds"
    )
    print("PYTHONHASHSEED for children: 0")
    print(f"Safe resume   : {bool(args.resume)}")
    print(f"Quarantine    : {bool(args.clean_partial)}")
    print(f"Continue error: {bool(args.continue_on_error)}")
    print(
        "Transient FL retries: "
        f"{TRANSIENT_RAY_STAGE_MAX_ATTEMPTS} total attempts for proven "
        "Ray GCS startup failures"
    )
    print("=" * 100)

    # Preserve the previous top-level orchestration manifests before this
    # session writes the canonical manifest filenames again.
    prior_orchestration_archive = archive_orchestration_manifests(
        paths
    )

    if prior_orchestration_archive:
        print(
            "Previous orchestration manifests preserved under: "
            f"{Path(next(iter(prior_orchestration_archive.values()))).parent}"
        )

    # 1) Verify the exact final source revisions before any expensive work.
    source_manifest = validate_source_contract(paths)

    # 2) Verify raw inputs, Python packages, DP helper/toolchain prerequisites.
    preflight_manifest = preflight(
        paths=paths,
        python_exe=python_exe,
        cse_input=cse_input,
        cic_decimal_dir=cic_decimal_dir,
        datasets=datasets,
        stages=stages,
        source_manifest=source_manifest,
    )

    print("Preflight: PASS")
    print(f"Preflight manifest: {paths.logs_dir / 'preflight_manifest.json'}")

    if args.preflight_only:
        print("Preflight-only requested; no experiment was started.")
        return

    plan = build_plan(datasets, seeds, stages)
    master_path = paths.logs_dir / "multi_seed_master_manifest.json"

    master_base = {
        "schema": "Reviewer1Comment1MultiSeedMasterManifestV1",
        "created_utc": now_utc(),
        "purpose": (
            "Ten-seed independently repartitioned statistical characterization "
            "of principal learning configurations."
        ),
        "datasets": datasets,
        "seeds": seeds,
        "stages": stages,
        "fixed_fl_contract": {
            "num_rsus": NUM_RSUS,
            "vehicles_per_rsu": VEHICLES_PER_RSU,
            "num_rounds": NUM_ROUNDS,
            "num_local_rounds": NUM_LOCAL_ROUNDS,
        },
        "results_dir": str(paths.results_dir),
        "logs_dir": str(paths.logs_dir),
        "statistics_dir": str(paths.statistics_dir),
        "python_executable": str(python_exe),
        "pythonhashseed": "0",
        "preflight_manifest": str(paths.logs_dir / "preflight_manifest.json"),
        "source_hashes": source_manifest["scripts"],
        "source_hash_semantics": (
            "These are the source files present for this orchestration "
            "session. Stages marked skipped_valid_existing were produced by "
            "a prior session; the previous top-level manifests are preserved "
            "in prior_orchestration_archive."
        ),
        "prior_orchestration_archive": prior_orchestration_archive,
        "plan": plan,
        "resume": bool(args.resume),
        "clean_partial": bool(args.clean_partial),
        "continue_on_error": bool(
            args.continue_on_error
        ),
        "transient_ray_stage_max_attempts": int(
            TRANSIENT_RAY_STAGE_MAX_ATTEMPTS
        ),
        "dry_run": bool(args.dry_run),
    }

    completed_results: List[StageResult] = []
    update_master_manifest(
        master_path,
        base=master_base,
        results=completed_results,
        state="running" if not args.dry_run else "dry_run",
    )

    env = build_child_env(paths)

    cse_source = preflight_manifest.get(
        "cse_source",
        {},
    )

    if isinstance(cse_source, dict) and cse_source:
        env[CSE_ENV_SHA256] = str(
            cse_source["sha256"]
        )
        env[CSE_ENV_ROWS] = str(
            cse_source["rows"]
        )

        label_counts = cse_source.get(
            "label_counts",
            {},
        )

        if isinstance(label_counts, dict):
            env[CSE_ENV_LABEL0] = str(
                label_counts.get("0", "")
            )
            env[CSE_ENV_LABEL1] = str(
                label_counts.get("1", "")
            )

    # Avoid repeatedly SHA-hashing the same very large preprocessing CSVs
    # before centralized, non-DP, and DP stages.
    preprocess_validation_cache: Dict[
        Tuple[str, int],
        Dict[str, Any],
    ] = {}

    failures: List[str] = []

    try:
        for dataset in datasets:
            for seed in seeds:
                print("\n" + "#" * 100)
                print(f"# DATASET={dataset} | SEED={seed}")
                print("#" * 100)

                seed_failed = False

                for stage in stages:
                    # Learning methods require preprocessing even if preprocessing
                    # was not selected in this invocation.
                    if stage != "preprocess":
                        try:
                            prep_validation = get_preprocess_validation(
                                preprocess_validation_cache,
                                paths,
                                dataset,
                                seed,
                            )

                            write_seed_input_contract(
                                paths,
                                dataset,
                                seed,
                                prep=prep_validation,
                            )

                        except Exception as exc:
                            raise RuntimeError(
                                f"{dataset} seed={seed} cannot start {stage}: "
                                "seed-specific preprocessing is missing or invalid. "
                                "Run/select the preprocess stage first."
                            ) from exc

                    print("\n" + "-" * 100)
                    print(f"Starting stage: dataset={dataset} seed={seed} stage={stage}")
                    print("-" * 100)

                    try:
                        result = execute_stage(
                            paths=paths,
                            python_exe=python_exe,
                            dataset=dataset,
                            seed=seed,
                            stage=stage,
                            cse_input=cse_input,
                            cic_decimal_dir=cic_decimal_dir,
                            env=env,
                            resume=bool(args.resume),
                            clean_partial=bool(args.clean_partial),
                            dry_run=bool(args.dry_run),
                        )

                        completed_results.append(result)
                        update_status(paths, dataset, seed, result)

                        print(
                            f"Stage status: {result.status} | "
                            f"elapsed={result.elapsed_sec:.2f}s"
                        )

                        # After preprocessing, immediately retain the validated
                        # split hashes and persist the paired-input contract.
                        if stage == "preprocess" and result.status != "dry_run":
                            key = (
                                dataset,
                                int(seed),
                            )

                            preprocess_validation_cache[
                                key
                            ] = result.validation

                            write_seed_input_contract(
                                paths,
                                dataset,
                                seed,
                                prep=result.validation,
                            )

                            print(
                                "Paired input contract ready: "
                                f"{log_dir_for(paths, dataset, seed) / 'paired_input_contract.json'}"
                            )

                        update_master_manifest(
                            master_path,
                            base=master_base,
                            results=completed_results,
                            state="running" if not args.dry_run else "dry_run",
                        )

                    except Exception as exc:
                        msg = (
                            f"FAILED dataset={dataset} seed={seed} "
                            f"stage={stage}: {exc}"
                        )
                        failures.append(msg)
                        seed_failed = True
                        print("\n" + "!" * 100)
                        print(msg)
                        print("!" * 100)

                        update_master_manifest(
                            master_path,
                            base=master_base,
                            results=completed_results,
                            state="failed",
                            error=msg,
                        )

                        if not args.continue_on_error:
                            raise

                        # A preprocessing failure invalidates all learning
                        # methods for this seed, so move to the next seed.
                        if stage == "preprocess":
                            break

                        # Centralized, non-DP, and DP are independent learning
                        # methods once the seed-specific preprocessing exists.
                        # Therefore one method failure must not terminate the
                        # remaining methods or the rest of the batch.
                        continue

                if seed_failed and args.continue_on_error:
                    continue
    except Exception:
        # The stage-level failure path has already updated the master
        # manifest. Record the orchestration abort with its traceback,
        # then propagate the original exception unchanged.
        logging.exception(
            "Multi-seed orchestration aborted before normal completion."
        )
        raise


    if args.dry_run:
        final_state = "dry_run"
    elif failures:
        final_state = "completed_with_failures"
    else:
        final_state = "completed"
    update_master_manifest(
        master_path,
        base=master_base,
        results=completed_results,
        state=final_state,
        error="\n".join(failures),
    )

    print("\n" + "=" * 100)
    print("MULTI-SEED ORCHESTRATOR FINISHED")
    print("=" * 100)
    print(f"State           : {final_state}")
    print(f"Stage records   : {len(completed_results)}")
    print(f"Failures        : {len(failures)}")
    print(f"Master manifest : {master_path}")
    print(f"Results root    : {paths.results_dir}")
    print(f"Logs root       : {paths.logs_dir}")
    print(f"Statistics root : {paths.statistics_dir}")
    print("=" * 100)

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
