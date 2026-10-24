"""Flask web app: upload page, result, history, health (Tasks 4 and 5, 8)."""
import base64
import hashlib
import logging

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from app.db import Database
from app.settings import Settings
from app.validation import ValidationError, read_upload

log = logging.getLogger("visionguard")
CAVEAT = ("This is a statistical estimate, not proof. Compression, screenshots, edits and unusual art "
          "styles can fool the model. Do not use it as the only basis for a serious decision.")


def create_app(predictor=None, db=None):
    app = Flask(__name__)
    app.config["SECRET_KEY"] = Settings.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = (Settings.MAX_UPLOAD_MB + 1) * 1024 * 1024
    logging.basicConfig(level=logging.INFO)

    if predictor is None:  # remote API (Task 8) or local model (Task 4)
        if Settings.INFERENCE_URL:
            from app.remote import RemotePredictor
            predictor = RemotePredictor()
        else:
            from app.inference import Predictor  # imported lazily: TensorFlow only when needed
            predictor = Predictor()
    db = db or Database(Settings.DATABASE_URL)
    app.predictor, app.db = predictor, db

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=Settings.MAX_UPLOAD_MB)

    @app.post("/predict")
    def predict():
        max_bytes = Settings.MAX_UPLOAD_MB * 1024 * 1024
        name, data = read_upload(request.files.get("image"), max_bytes)
        res = app.predictor.predict_bytes(data, name) if hasattr(app.predictor, "healthy") else app.predictor.predict_bytes(data)
        app.db.add(filename=name, file_sha256=hashlib.sha256(data).hexdigest(), label=res["label"],
                   prob_ai=res["prob_ai"], confidence=res["confidence"], threshold=res.get("threshold"),
                   model_name=res.get("model_name", "unknown"), model_version=str(res.get("model_version", "")),
                   latency_ms=res.get("latency_ms"))
        preview = "data:image/jpeg;base64," + base64.b64encode(data).decode() if len(data) < 1_500_000 else None
        return render_template("result.html", res=res, name=name, preview=preview, caveat=CAVEAT)

    @app.get("/history")
    def history():
        return render_template("history.html", rows=app.db.recent())

    @app.get("/health")
    def health():
        out = {"status": "ok", "database": "ok", "model": "ok"}
        code = 200
        try:
            app.db.ping()
        except Exception as e:  # noqa: BLE001
            out.update(status="degraded", database=f"error: {type(e).__name__}")
            code = 503
        if hasattr(app.predictor, "healthy") and not app.predictor.healthy():
            out.update(status="degraded", model="remote API unreachable")
            code = 503
        return jsonify(out), code

    # ---------- friendly error pages ----------
    def err(code, title, msg):
        return render_template("error.html", code=code, title=title, msg=msg), code

    @app.errorhandler(ValidationError)
    def _val(e):
        return err(e.status, "Can't use that file", e.message)

    @app.errorhandler(413)
    def _big(_):
        return err(413, "File too large", f"Upload an image under {Settings.MAX_UPLOAD_MB} MB.")

    @app.errorhandler(404)
    def _404(_):
        return err(404, "Page not found", "That page does not exist.")

    @app.errorhandler(Exception)
    def _500(e):
        if isinstance(e, HTTPException):
            return err(e.code, e.name, e.description)
        log.exception("unhandled error: %s", e)  # details go to logs, not to the user
        return err(500, "Something went wrong", "The verdict could not be produced. Try again in a moment.")

    return app


if __name__ == "__main__":
    # Flask's built-in server, debug OFF: debug mode exposes the interactive debugger
    # (remote code execution if reachable) and the auto-reloader starts the app twice
    # (the model would be loaded twice).
    create_app().run(host="0.0.0.0", port=Settings.PORT, debug=False)
