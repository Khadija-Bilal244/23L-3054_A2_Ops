"""Task 7 - model inference API (runs in the cloud).  POST /predict  (multipart field 'image').
Protected by a shared secret in the X-API-Key header.  python -m app.api
"""
import hmac

from flask import Flask, jsonify, request

from app.settings import Settings
from app.validation import ValidationError, read_upload


def create_api(predictor=None):
    api = Flask(__name__)
    api.config["MAX_CONTENT_LENGTH"] = (Settings.MAX_UPLOAD_MB + 1) * 1024 * 1024
    if predictor is None:
        from app.inference import Predictor
        predictor = Predictor()
    api.predictor = predictor

    @api.get("/health")
    def health():
        return jsonify(status="ok", model=predictor.model_name, version=predictor.model_version)

    @api.post("/predict")
    def predict():
        key = request.headers.get("X-API-Key", "")
        if not Settings.API_KEY or not hmac.compare_digest(key, Settings.API_KEY):
            return jsonify(error="invalid or missing API key"), 401
        try:
            _, data = read_upload(request.files.get("image"), Settings.MAX_UPLOAD_MB * 1024 * 1024)
        except ValidationError as e:
            return jsonify(error=e.message), e.status
        return jsonify(predictor.predict_bytes(data))

    return api


if __name__ == "__main__":
    create_api().run(host="0.0.0.0", port=Settings.PORT, debug=False)
