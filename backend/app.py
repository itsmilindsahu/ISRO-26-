#!/usr/bin/env python3
"""
app.py
======
Minimal Flask backend for the Satellite Frame Interpolation demo web app.

Wires the existing research code (interpolate.py / data_prep.py / metrics.py)
to two REST endpoints consumed by the frontend (frontend/index.html + .js):

    POST /interpolate     -> JSON with base64 PNGs for all 3 methods
                              (linear_blend, flow_only_warp, hybrid_ot) +
                              PSNR/SSIM if a ground-truth frame is available.
    GET  /download-zip    -> ZIP archive of the most recent run's outputs
                              (falls back to the bundled sample-run zip).
    GET  /sample-pair     -> base64 frame0/frame1 (+ optional frame2 ground
                              truth) for the "Load sample pair" button.
    GET  /health          -> simple liveness check.

Run:
    pip install -r requirements.txt
    python app.py
    # serves on http://localhost:5000
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import zipfile

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

import data_prep
import metrics as metrics_mod
from interpolate import (
    compute_farneback_flow,
    flow_only_warp,
    interpolate,
    linear_blend,
)

app = Flask(__name__)
CORS(app)  # allow the static frontend (served from any origin/port) to call this API

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir, "frontend"))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
RUN_DIR = os.path.join(BASE_DIR, "_last_run")  # scratch space for the most recent request
FRAME_SIZE = data_prep.FRAME_SIZE

# Ground truth for the bundled sample pair only (used to show PSNR/SSIM when
# the user clicks "Load sample pair" rather than uploading their own frames).
_SAMPLE_GT_PATH = os.path.join(SAMPLE_DIR, "frame2.png")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _decode_upload_to_gray01(file_storage) -> np.ndarray:
    """Decode an uploaded image file into a float32 grayscale array in [0,1],
    resized to the standard FRAME_SIZE used by the research pipeline."""
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Please upload a PNG/JPG file.")
    img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0)


def _load_local_gray01(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def _to_data_url(img01: np.ndarray) -> str:
    """Encode a float32 [0,1] grayscale array as a base64 PNG data URL."""
    img8 = (np.clip(img01, 0.0, 1.0) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img8)
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _to_heatmap_data_url(img01: np.ndarray, gt01: np.ndarray) -> str:
    """Create a colorized absolute-difference heatmap vs. the ground truth."""
    diff = np.abs(np.clip(img01, 0.0, 1.0) - np.clip(gt01, 0.0, 1.0))
    diff8 = (np.clip(diff, 0.0, 1.0) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(diff8, cv2.COLORMAP_JET)
    ok, buf = cv2.imencode(".png", colored)
    if not ok:
        raise RuntimeError("Heatmap encoding failed.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _save_png(img01: np.ndarray, path: str) -> None:
    img8 = (np.clip(img01, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(path, img8)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/sample-pair")
def sample_pair():
    """Return the bundled synthetic/demo frame0 + frame1 (base64), used by
    the frontend's 'Load sample pair' button."""
    frame0 = _load_local_gray01(os.path.join(SAMPLE_DIR, "frame0.png"))
    frame1 = _load_local_gray01(os.path.join(SAMPLE_DIR, "frame1.png"))
    return jsonify({
        "frame0": _to_data_url(frame0),
        "frame1": _to_data_url(frame1),
        "has_ground_truth": os.path.exists(_SAMPLE_GT_PATH),
    })


@app.post("/interpolate")
def do_interpolate():
    """
    Accepts multipart/form-data:
        frame0 : image file  (required, unless is_sample=1)
        frame1 : image file  (required, unless is_sample=1)
        t      : float in [0,1]  (required)
        is_sample : "1" to use the bundled sample pair instead of uploads

    Returns JSON:
        {
          "t": 0.5,
          "interpolated": "<data-url>",       # hybrid OT result (primary)
          "methods": {
            "linear_blend":    {"image": "<data-url>", "psnr": ..., "ssim": ...},
            "flow_only_warp":  {"image": "<data-url>", "psnr": ..., "ssim": ...},
            "hybrid_ot":       {"image": "<data-url>", "psnr": ..., "ssim": ...}
          },
          "has_metrics": true/false
        }
    """
    try:
        t = float(request.form.get("t", 0.5))
    except (TypeError, ValueError):
        return jsonify({"error": "t must be a number between 0 and 1"}), 400
    t = max(0.0, min(1.0, t))

    try:
        reg = float(request.form.get("reg", 0.004))
    except (TypeError, ValueError):
        return jsonify({"error": "reg must be a number"}), 400
    reg = max(1e-6, reg)

    is_sample = request.form.get("is_sample") in ("1", "true", "True")
    ground_truth = None

    try:
        if is_sample:
            frame0 = _load_local_gray01(os.path.join(SAMPLE_DIR, "frame0.png"))
            frame1 = _load_local_gray01(os.path.join(SAMPLE_DIR, "frame1.png"))
            if os.path.exists(_SAMPLE_GT_PATH):
                ground_truth = _load_local_gray01(_SAMPLE_GT_PATH)
        else:
            if "frame0" not in request.files or "frame1" not in request.files:
                return jsonify({"error": "frame0 and frame1 files are required"}), 400
            frame0 = _decode_upload_to_gray01(request.files["frame0"])
            frame1 = _decode_upload_to_gray01(request.files["frame1"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # --- Run all three methods ---
    lin = linear_blend(frame0, frame1, t)

    flow01 = compute_farneback_flow(frame0, frame1)
    flow10 = compute_farneback_flow(frame1, frame0)
    flow_only = flow_only_warp(frame0, frame1, t, flow01=flow01, flow10=flow10)

    hybrid = interpolate(frame0, frame1, t, reg=reg)

    results = {
        "linear_blend": lin,
        "flow_only_warp": flow_only,
        "hybrid_ot": hybrid,
    }

    methods_payload = {}
    has_metrics = ground_truth is not None
    for name, img in results.items():
        entry = {"image": _to_data_url(img)}
        if has_metrics:
            m = metrics_mod.compute_metrics(img, ground_truth)
            entry["psnr"] = round(m["psnr"], 2)
            entry["ssim"] = round(m["ssim"], 4)
            entry["heatmap"] = _to_heatmap_data_url(img, ground_truth)
        methods_payload[name] = entry

    # --- Persist this run to disk so /download-zip has something fresh ---
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(RUN_DIR, exist_ok=True)
    _save_png(frame0, os.path.join(RUN_DIR, "frame0.png"))
    _save_png(frame1, os.path.join(RUN_DIR, "frame1.png"))
    _save_png(lin, os.path.join(RUN_DIR, "linear_blend.png"))
    _save_png(flow_only, os.path.join(RUN_DIR, "flow_only_warp.png"))
    _save_png(hybrid, os.path.join(RUN_DIR, "hybrid_ot_result.png"))
    if ground_truth is not None:
        _save_png(ground_truth, os.path.join(RUN_DIR, "frame2_ground_truth.png"))
    with open(os.path.join(RUN_DIR, "metrics.txt"), "w") as fh:
        fh.write(f"t={t}\n")
        if has_metrics:
            for name, entry in methods_payload.items():
                fh.write(f"{name}: PSNR={entry['psnr']} dB, SSIM={entry['ssim']}\n")
        else:
            fh.write("no ground-truth frame available for this pair\n")

    return jsonify({
        "t": t,
        "interpolated": methods_payload["hybrid_ot"]["image"],
        "methods": methods_payload,
        "has_metrics": has_metrics,
        "ground_truth": _to_data_url(ground_truth) if ground_truth is not None else None,
    })


@app.get("/download-zip")
def download_zip():
    """Zip up the most recent run's output frames (+ metrics.txt) and send it.
    Falls back to zipping the bundled sample_data if no run has happened yet."""
    source_dir = RUN_DIR if os.path.isdir(RUN_DIR) else SAMPLE_DIR

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(source_dir)):
            fpath = os.path.join(source_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)
    mem.seek(0)

    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name="sat-frame-interp-results.zip",
    )


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def frontend_static(filename):
    path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(path) and os.path.isfile(path):
        return send_from_directory(FRONTEND_DIR, filename)
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=port, debug=debug_mode)
