"""
run_logger.py 

Saves:
- params/<run_id>.csv   
- figures/<run_id>/...
- logs/run_log.csv      (index of all runs)

Designed for a repo where run_logger.py is inside dual_tracer_sims/utils/
"""

import csv
import subprocess
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


# ---------- Path configuration ----------
# run_logger.py lives in dual_tracer_sims/utils/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURES_DIR = PROJECT_ROOT / "figures"
PARAMS_DIR = PROJECT_ROOT / "params"
LOGS_DIR = PROJECT_ROOT / "logs"
RUN_LOG_PATH = LOGS_DIR / "run_log.csv"


# ---------- Data structure for run metadata ----------
@dataclass
class RunInfo:
    run_id: str
    git_sha: str
    params_path: str
    description: str
    start_time: str
    figure_dir: str


# ---------- Internal helpers ----------

def _ensure_base_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _get_git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        sha = "not_a_git_repo"
    return sha


def _generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"sim_{ts}_{rand}"


def _write_params_csv(run_id: str, params: Dict, git_sha: str) -> Path:
    """
    Save parameters as a CSV file:
        key,value
        tau_fdg,12.0
        tau_pbr,25.0
        ...
    Automatically injects run_id and git_sha.
    """
    params_copy = dict(params)
    params_copy["run_id"] = run_id
    params_copy["git_sha"] = git_sha

    path = PARAMS_DIR / f"{run_id}.csv"

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in params_copy.items():
            writer.writerow([k, v])

    return path


def _append_run_log(row_dict: Dict) -> None:
    df_new = pd.DataFrame([row_dict])
    if RUN_LOG_PATH.exists():
        df_new.to_csv(RUN_LOG_PATH, mode="a", index=False, header=False)
    else:
        df_new.to_csv(RUN_LOG_PATH, index=False)


# ---------- Public API ----------

def start_run(param_dict: Dict, description: str = "") -> RunInfo:
    """
    Creates run_id, figure folder, CSV param file.
    """
    _ensure_base_dirs()

    git_sha = _get_git_sha()
    run_id = _generate_run_id()

    # create figures/<run_id>/
    fig_dir = FIGURES_DIR / run_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    # write params/<run_id>.csv
    params_path = _write_params_csv(run_id, param_dict, git_sha)

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return RunInfo(
        run_id=run_id,
        git_sha=git_sha,
        params_path=str(params_path.relative_to(PROJECT_ROOT)),
        description=description,
        start_time=start_time,
        figure_dir=str(fig_dir.relative_to(PROJECT_ROOT)),
    )


def get_figure_path(run_info: RunInfo, filename: str) -> Path:
    """
    Returns the full path to a figure inside this run’s folder.
    """
    fig_dir = PROJECT_ROOT / run_info.figure_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir / filename



def finalize_run(
    run_info: RunInfo,
    figure_filenames: Union[str, List[str]],
    key_params_str: str = "",
    notes: str = "",
) -> Dict:
    """
    Writes a row to logs/run_log.csv for Excel tracking.
    """
    if isinstance(figure_filenames, str):
        figure_filenames = [figure_filenames]

    normalized = []
    for f in figure_filenames:
        p = Path(f)
        try:
            normalized.append(str(p.relative_to(PROJECT_ROOT)))
        except ValueError:
            normalized.append(str(p))

    row = {
        "Run ID": run_info.run_id,
        "Date": run_info.start_time,
        "Figure Filenames": "; ".join(normalized),
        "Description": run_info.description,
        "Key Parameters": key_params_str,
        "Full Param File": run_info.params_path,
        "Code Version (Git SHA)": run_info.git_sha,
        "Notes": notes,
    }

    _append_run_log(row)
    return row


def runinfo_to_dict(run_info: RunInfo) -> Dict:
    return asdict(run_info)