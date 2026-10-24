"""Task 8 - client for the hosted inference API (the model lives in the cloud)."""
import requests

from app.settings import Settings


class RemotePredictor:
    def __init__(self, url=None, api_key=None, timeout=30):
        self.url = (url or Settings.INFERENCE_URL).rstrip("/")
        self.key = api_key or Settings.INFERENCE_API_KEY
        self.timeout = timeout
        self.model_name, self.model_version, self.threshold = Settings.MODEL_NAME, "remote", None

    def predict_bytes(self, data: bytes, filename="upload.jpg") -> dict:
        r = requests.post(f"{self.url}/predict", headers={"X-API-Key": self.key},
                          files={"image": (filename, data)}, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Inference API returned {r.status_code}: {r.text[:200]}")
        return r.json()

    def healthy(self) -> bool:
        try:
            return requests.get(f"{self.url}/health", timeout=5).ok
        except requests.RequestException:
            return False
