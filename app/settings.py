"""Task 4.4 - every setting comes from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/champion/model.keras")
    THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/champion/threshold.json")
    THRESHOLD = os.getenv("THRESHOLD")  # optional override of threshold.json
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///history.db")
    MODEL_NAME = os.getenv("MODEL_NAME", "visionguard")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
    INFERENCE_URL = os.getenv("INFERENCE_URL", "")  # empty -> local model
    INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")
    API_KEY = os.getenv("API_KEY", "")  # used by app.api (server side)
    PORT = int(os.getenv("PORT", "5000"))
