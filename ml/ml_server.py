#!/usr/bin/env python3
"""Flask service to collect feedback images and periodically retrain the model."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# --- ADDED IMPORTS ---
import boto3
import requests
# ---------------------

from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    base_dir: Path
    data_dir: Path
    weak_feedback_dir: Path
    artifacts_root: Path
    serving_dir: Path
    state_path: Path
    train_script: Path
    min_test_auc: float
    max_test_drop: float
    train_interval_seconds: float
    initial_delay_seconds: float
    frontend_url: Optional[str]
    frontend_timeout: float
    train_overrides: Dict[str, object]
    # --- ADDED S3 CONFIG ---
    s3_bucket: str
    s3_prefix: str
    # -----------------------


def _load_overrides(raw: str) -> Dict[str, object]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logging.getLogger("ml_server").warning("TRAIN_ARG_OVERRIDES parse error: %s", exc)
        return {}
    if not isinstance(parsed, dict):
        logging.getLogger("ml_server").warning("TRAIN_ARG_OVERRIDES must be a JSON object")
        return {}
    return parsed


def load_config() -> ServerConfig:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    weak_dir = data_dir / "weak_feedback"
    artifacts_root = base_dir / "artifacts"
    serving_dir = base_dir / "serving" / "current"
    state_path = base_dir / "state.json"
    train_script = base_dir / "train.py"

    min_auc = float(os.getenv("MIN_TEST_AUC", "0.7"))
    max_drop = float(os.getenv("MAX_TEST_DROP", "0.02"))
    interval = float(os.getenv("RETRAIN_INTERVAL_MINUTES", "60")) * 60.0
    initial_delay = float(os.getenv("RETRAIN_INITIAL_DELAY_SECONDS", "1800"))
    
    # --- MODIFIED: Points to your live API (or the internal service name) ---
    frontend_url = os.getenv("FRONTEND_URL", "http://35.200.164.43/reload-model")
    # -----------------------------------------------------------------------

    frontend_timeout = float(os.getenv("FRONTEND_TIMEOUT", "15.0"))
    overrides = _load_overrides(os.getenv("TRAIN_ARG_OVERRIDES", ""))

    return ServerConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        weak_feedback_dir=weak_dir,
        artifacts_root=artifacts_root,
        serving_dir=serving_dir,
        state_path=state_path,
        train_script=train_script,
        min_test_auc=min_auc,
        max_test_drop=max_drop,
        train_interval_seconds=interval,
        initial_delay_seconds=initial_delay,
        frontend_url=frontend_url,
        frontend_timeout=frontend_timeout,
        train_overrides=overrides,
        # --- ADDED S3 CONFIG LOADING FROM ENV VARS ---
        s3_bucket=os.getenv("S3_BUCKET", ""),
        s3_prefix=os.getenv("S3_MODEL_PREFIX", "models/"),
        # ---------------------------------------------
    )


CONFIG = load_config()
s3 = boto3.client("s3")

# Ensure directories exist
for path in (
    CONFIG.weak_feedback_dir / "no",
    CONFIG.weak_feedback_dir / "yes",
    CONFIG.artifacts_root,
    CONFIG.serving_dir.parent,
):
    path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("ml_server")

CLASS_LABELS = {"no", "yes"}
POSITIVE_FEEDBACK = {"yes", "y", "true", "1", "upvote", "positive", "correct", "good"}
NEGATIVE_FEEDBACK = {"no", "n", "false", "0", "downvote", "negative", "incorrect", "bad"}

STATE_LOCK = threading.Lock()

app = Flask(__name__)
app.config.update(
    DATA_DIR=CONFIG.data_dir,
    WEAK_FEEDBACK_DIR=CONFIG.weak_feedback_dir,
    ARTIFACTS_ROOT=CONFIG.artifacts_root,
    SERVING_DIR=CONFIG.serving_dir,
    STATE_PATH=CONFIG.state_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_state() -> Dict[str, object]:
    with STATE_LOCK:
        if CONFIG.state_path.exists():
            try:
                return json.loads(CONFIG.state_path.read_text())
            except json.JSONDecodeError:
                LOGGER.warning("state file corrupted; starting fresh")
        return {"history": []}


def _save_state(state: Dict[str, object]) -> None:
    with STATE_LOCK:
        CONFIG.state_path.write_text(json.dumps(state, indent=2))


def _parse_feedback(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("feedback missing")
    if isinstance(value, (int, float)):
        return value > 0
    text = str(value).strip().lower()
    if text in POSITIVE_FEEDBACK:
        return True
    if text in NEGATIVE_FEEDBACK:
        return False
    raise ValueError(f"unsupported feedback value: {value}")


def _resolved_label(prediction: str, positive_feedback: bool) -> str:
    if prediction not in CLASS_LABELS:
        raise ValueError(f"unsupported prediction label: {prediction}")
    if positive_feedback:
        return prediction
    return "yes" if prediction == "no" else "no"


def _decode_base64_image(encoded: str, image_format: Optional[str]) -> Tuple[bytes, str]:
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 image payload") from exc
    extension = f".{image_format.strip().lower()}" if image_format else ".jpg"
    return data, extension


def _persist_feedback_image(data: bytes, extension: str, destination: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    filename = f"feedback_{timestamp}_{suffix}{extension}"
    path = destination / filename
    path.write_bytes(data)
    return path


def _read_tail(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return "(missing)"
    if len(text) > max_chars:
        text = "(truncated)\n" + text[-max_chars:]
    return text


def _stream_process_logs(proc: subprocess.Popen[str], run_token: str) -> Tuple[str, str]:
    """Pump stdout/stderr to the service logger while buffering for later storage."""

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _reader(stream, sink: list[str], level: int) -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            sink.append(line)
            LOGGER.log(level, "[train %s] %s", run_token, line.rstrip())
        stream.close()

    threads = [
        threading.Thread(target=_reader, args=(proc.stdout, stdout_lines, logging.INFO), daemon=True),
        threading.Thread(target=_reader, args=(proc.stderr, stderr_lines, logging.ERROR), daemon=True),
    ]
    for thread in threads:
        thread.start()

    proc.wait()
    for thread in threads:
        thread.join()

    proc_stdout = "".join(stdout_lines)
    proc_stderr = "".join(stderr_lines)
    return proc_stdout, proc_stderr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

ALLOWED_OVERRIDE_KEYS = {"epochs", "batch_size", "lr", "weak_weight", "seed", "no_pretrained"}


def _build_train_command(run_dir: Path) -> Tuple[list[str], Dict[str, object]]:
    extras = {k: v for k, v in CONFIG.train_overrides.items() if k in ALLOWED_OVERRIDE_KEYS}
    cmd: list[str] = [
        sys.executable,
        str(CONFIG.train_script),
        "--data_dir",
        str(CONFIG.data_dir),
        "--weak_dir",
        str(CONFIG.weak_feedback_dir),
        "--out_dir",
        str(run_dir),
    ]
    for key, value in extras.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd, extras

# --- MODIFIED: This function now just calls the API, and _promote_model is replaced ---
def _notify_api_of_reload() -> Dict[str, object]:
    """Sends a POST request to the deployed API to trigger a model reload."""
    if not CONFIG.frontend_url:
        return {"notified": False, "reason": "FRONTEND_URL not set"}
    LOGGER.info("Notifying API at %s to reload model...", CONFIG.frontend_url)
    try:
        # The payload is not needed for the reload endpoint, just the trigger
        resp = requests.post(CONFIG.frontend_url, timeout=CONFIG.frontend_timeout)
        resp.raise_for_status()
        LOGGER.info("API responded with status code: %d", resp.status_code)
        return {"notified": True, "status_code": resp.status_code}
    except Exception as exc:
        LOGGER.warning("API notification failed: %s", exc)
        return {"notified": False, "reason": str(exc)}

# --- ADDED: New function to handle S3 upload ---
def _upload_artifacts_to_s3(run_dir: Path, model_version: str) -> bool:
    """Uploads key model artifacts from a local directory to S3."""
    if not CONFIG.s3_bucket:
        LOGGER.error("S3_BUCKET is not set. Cannot upload.")
        return False
    
    s3_path_prefix = f"{CONFIG.s3_prefix.strip('/')}/{model_version}/"
    files_to_upload = ["model.keras", "metrics.json", "label_map.json", "threshold.txt", "model_version.txt"]
    
    LOGGER.info("Uploading artifacts to s3://%s/%s", CONFIG.s3_bucket, s3_path_prefix)
    
    for filename in files_to_upload:
        local_path = run_dir / filename
        if not local_path.exists():
            LOGGER.warning("Artifact %s not found in %s, skipping upload.", filename, run_dir)
            continue
        try:
            s3_key = f"{s3_path_prefix}{filename}"
            s3.upload_file(str(local_path), CONFIG.s3_bucket, s3_key)
            LOGGER.info("-> Successfully uploaded %s", s3_key)
        except Exception as e:
            LOGGER.error("Failed to upload %s to S3: %s", filename, e)
            return False
    return True


def _evaluate_metrics(metrics: Dict[str, object]) -> Tuple[bool, list[str]]:
    reasons: list[str] = []
    test_metrics = metrics.get("test") if isinstance(metrics, dict) else None
    test_auc = None
    if isinstance(test_metrics, dict):
        test_auc = test_metrics.get("auc")
    if test_auc is None:
        return False, ["metrics missing test.auc"]

    state = _load_state()
    previous = state.get("last_promoted") or {}
    prev_metrics = previous.get("metrics") or {}
    prev_test = prev_metrics.get("test") if isinstance(prev_metrics, dict) else {}
    prev_auc = prev_test.get("auc") if isinstance(prev_test, dict) else None

    promote = True
    if test_auc < CONFIG.min_test_auc:
        promote = False
        reasons.append(f"test_auc {test_auc:.3f} below minimum {CONFIG.min_test_auc:.3f}")
    if prev_auc is not None and test_auc + CONFIG.max_test_drop < prev_auc:
        promote = False
        reasons.append(
            f"test_auc dropped more than tolerance ({prev_auc:.3f} -> {test_auc:.3f}, tol {CONFIG.max_test_drop:.3f})"
        )
    return promote, reasons


def run_training_cycle(trigger: str) -> Dict[str, object]:
    run_token = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
    run_dir = CONFIG.artifacts_root / run_token
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd, used_overrides = _build_train_command(run_dir)
    LOGGER.info("starting training cycle %s (trigger=%s)", run_token, trigger)

    attempt = {
        "artifact_path": str(run_dir.relative_to(CONFIG.base_dir)),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "train_command": cmd,
        "train_args": used_overrides,
        "model_version": run_token,
        "trigger": trigger,
    }

    logs = {"stdout": "", "stderr": "", "returncode": None}
    reasons: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        reasons.append(f"failed to launch training: {exc}")
        status = "failed"
    else:
        stdout_text, stderr_text = _stream_process_logs(proc, run_token)
        returncode = proc.returncode
        logs = {
            "stdout": stdout_text,
            "stderr": stderr_text,
            "returncode": returncode,
        }
        status = "running"
        if returncode != 0:
            status = "failed"
            reasons.append(f"training exited with code {returncode}")

    (run_dir / "train_stdout.log").write_text(logs["stdout"])
    (run_dir / "train_stderr.log").write_text(logs["stderr"])

    metrics: Dict[str, object] = {}
    promoted = False
    notification = {"notified": False}

    if status != "failed":
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            status = "failed"
            reasons.append("metrics.json missing after training")
        else:
            metrics = json.loads(metrics_path.read_text())
            model_version_path = run_dir / "model_version.txt"
            if model_version_path.exists():
                attempt["model_version"] = model_version_path.read_text().strip() or attempt["model_version"]
            
            promote, metric_reasons = _evaluate_metrics(metrics)
            reasons.extend(metric_reasons)
            
            if promote:
                # --- THIS IS THE KEY CHANGE ---
                # Instead of promoting locally, we upload to S3 and notify the API
                upload_ok = _upload_artifacts_to_s3(run_dir, attempt["model_version"])
                if upload_ok:
                    notification = _notify_api_of_reload()
                    promoted = True
                    status = "promoted"
                else:
                    status = "failed"
                    reasons.append("S3 upload failed")
                # ------------------------------
            else:
                status = "rejected"

    attempt.update({
        "status": status,
        "reasons": reasons,
        "metrics": metrics if metrics else None,
    })

    state = _load_state()
    history = state.setdefault("history", [])
    history.append(attempt)
    state["last_attempt"] = attempt
    if promoted:
        state["last_promoted"] = {
            "artifact_path": attempt["artifact_path"],
            "model_version": attempt["model_version"],
            "metrics": metrics,
            "promoted_at": datetime.utcnow().isoformat() + "Z",
        }
    _save_state(state)

    LOGGER.info("training cycle %s finished with status=%s", run_token, status)
    return {
        "status": status,
        "promoted": promoted,
        "reasons": reasons,
        "metrics": metrics,
        "artifact_path": attempt["artifact_path"],
        "model_version": attempt["model_version"],
        "notification": notification,
        "train_logs": logs,
    }

# (The TrainingManager and all your Flask routes are untouched below this line)
class TrainingManager:
    """Serialises training runs and owns the background scheduler."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scheduler_lock = threading.Lock()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._startup_dispatched = False

    def run(self, trigger: str) -> Dict[str, object]:
        with self._lock:
            return run_training_cycle(trigger)

    def ensure_scheduler(self) -> None:
        if not self._startup_dispatched:
            self._startup_dispatched = True
            threading.Thread(target=self._run_async, args=("startup",), daemon=True).start()

        if CONFIG.train_interval_seconds <= 0:
            if CONFIG.train_interval_seconds == 0:
                LOGGER.info("auto-training scheduler disabled")
            return

        with self._scheduler_lock:
            if self._scheduler_thread and self._scheduler_thread.is_alive():
                return
            LOGGER.info(
                "starting training scheduler: every %.1f minutes", CONFIG.train_interval_seconds / 60.0
            )
            self._scheduler_thread = threading.Thread(
                target=self._loop,
                name="TrainingScheduler",
                daemon=True,
            )
            self._scheduler_thread.start()

    def _run_async(self, trigger: str) -> None:
        try:
            self.run(trigger)
        except Exception:
            LOGGER.exception("training cycle crashed (trigger=%s)", trigger)

    def _loop(self) -> None:
        delay = max(0.0, CONFIG.initial_delay_seconds)
        if delay:
            time.sleep(delay)
        interval = max(1.0, CONFIG.train_interval_seconds)
        while True:
            self._run_async("scheduled")
            time.sleep(interval)


TRAINING_MANAGER = TrainingManager()

@app.before_request
def _start_scheduler() -> None:
    TRAINING_MANAGER.ensure_scheduler()


@app.route("/health", methods=["GET"])
def health() -> tuple:
    return jsonify({"status": "ok"})


@app.route("/feedback", methods=["PUT"])
def ingest_feedback():
    prediction = request.form.get("prediction") or request.values.get("prediction")
    feedback_value = request.form.get("feedback") or request.values.get("feedback")
    image_file = request.files.get("image")

    payload = request.get_json(silent=True) if request.is_json else None
    if payload and not image_file:
        prediction = prediction or payload.get("prediction")
        feedback_value = feedback_value or payload.get("feedback")

    if not prediction:
        return jsonify({"error": "missing prediction"}), 400
    if feedback_value is None:
        return jsonify({"error": "missing feedback"}), 400

    try:
        prediction = prediction.strip().lower()
        positive_feedback = _parse_feedback(feedback_value)
        label = _resolved_label(prediction, positive_feedback)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if image_file:
        filename = secure_filename(image_file.filename or "")
        extension = Path(filename).suffix or ".jpg"
        data = image_file.read()
    else:
        if not payload:
            return jsonify({"error": "missing image"}), 400
        encoded = payload.get("image_base64")
        if not encoded:
            return jsonify({"error": "missing image_base64"}), 400
        try:
            data, extension = _decode_base64_image(encoded, payload.get("image_format"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    destination = CONFIG.weak_feedback_dir / label
    destination.mkdir(parents=True, exist_ok=True)
    try:
        saved_path = _persist_feedback_image(data, extension, destination)
    except Exception as exc:
        LOGGER.exception("failed to persist feedback image")
        return jsonify({"error": "failed to save image", "details": str(exc)}), 500

    metadata = {
        "prediction": prediction,
        "feedback_positive": positive_feedback,
        "final_label": label,
        "stored_at": datetime.utcnow().isoformat() + "Z",
        "image_path": str(saved_path.relative_to(CONFIG.base_dir)),
    }
    saved_path.with_suffix(saved_path.suffix + ".json").write_text(json.dumps(metadata, indent=2))
    LOGGER.info("stored feedback image at %s", saved_path)
    return jsonify({"ok": True, **metadata})


@app.route("/status", methods=["GET"])
def status():
    state = _load_state()
    wants_json = request.args.get("format") == "json"
    if not wants_json:
        accept = request.accept_mimetypes
        if accept.best == "application/json" and accept[accept.best] >= accept["text/html"]:
            wants_json = True
    if wants_json:
        return jsonify(state)

    last_attempt = state.get("last_attempt") or {}
    last_promoted = state.get("last_promoted") or {}
    history = (state.get("history") or [])[-10:]

    artifact_dir = CONFIG.base_dir / last_attempt.get("artifact_path", "") if last_attempt else None
    if artifact_dir and not artifact_dir.exists():
        artifact_dir = None
    stdout_log = _read_tail(artifact_dir / "train_stdout.log") if artifact_dir else "(no artifact)"
    stderr_log = _read_tail(artifact_dir / "train_stderr.log") if artifact_dir else "(no artifact)"

    template = """
    <!doctype html>
    <html lang=\"en\">
    <head><title>ML Server Status</title></head>
    <body>...</body>
    </html>
    """
    return render_template_string(template)

if __name__ == "__main__":
    LOGGER.info("--- Running a single, manual training cycle as a one-off script ---")
    try:
        result = run_training_cycle(trigger="manual_k8s_job")
        if result.get("status") not in ("promoted", "rejected"): # Rejected is a valid, non-error outcome
            LOGGER.error("Training cycle did not succeed. Final status: %s", result.get("status"))
            sys.exit(1)
        LOGGER.info("--- Training run complete. Final status: %s ---", result.get("status"))
        sys.exit(0)
    except Exception as e:
        LOGGER.exception("A critical error occurred during the training cycle.")
        sys.exit(1)