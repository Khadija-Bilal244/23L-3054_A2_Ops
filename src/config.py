"""Shared settings. Secrets come ONLY from environment variables (.env)."""
import os
import subprocess

from dotenv import load_dotenv

load_dotenv()

ROLL = os.getenv("ROLL_NUMBER", "00000000")
PROJECT = f"{ROLL}_A2_Ops"
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
LABELS = {0: "real", 1: "ai_generated"}  # positive class (1) = AI-generated


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dvc_data_version(dvc_file: str = "data/raw.dvc") -> str:
    """md5 stored in the .dvc pointer file = the exact dataset version."""
    try:
        import yaml

        with open(dvc_file) as f:
            return str(yaml.safe_load(f)["outs"][0]["md5"])
    except Exception:
        return "unknown"


def setup_mlflow(experiment: str | None = None):
    """Point MLflow at DagsHub (if env vars set) or at local ./mlruns."""
    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI")
    user, token = os.getenv("DAGSHUB_USER"), os.getenv("DAGSHUB_TOKEN")
    if user and token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or PROJECT)
    return mlflow
