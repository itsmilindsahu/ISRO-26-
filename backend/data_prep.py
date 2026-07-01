"""
data_prep.py
============
Acquire a pair (optionally a triplet) of grayscale "satellite-like" frames
to interpolate between.

Acquisition strategy (in order of preference):
  1. Local files already present in ``/data`` (frame0.png, frame1.png,
     optional frame2.png as a held-out ground-truth middle frame).
  2. A live download of two consecutive GOES-16 CONUS Band-13 (clean
     longwave IR) browse images from NOAA's public "Open Data" S3 bucket
     (``noaa-goes16``, no auth / no AWS keys required). We deliberately
     avoid a hard dependency on netCDF4/xarray (kept out of the minimal
     dependency list) by pulling small PNG/JPG quick-look renders of the
     band, which are plain raster images OpenCV can decode directly.
  3. A synthetic "moving blob" toy pair, generated procedurally, used
     whenever the network is unavailable/blocked or the remote layout
     changes. This keeps the whole pipeline runnable offline.

All frames are returned as float32 grayscale arrays in [0, 1], resized to
``FRAME_SIZE x FRAME_SIZE``.
"""

from __future__ import annotations

import os
import urllib.request
import urllib.error
from typing import Optional, Tuple

import cv2
import numpy as np

FRAME_SIZE = 192

# Two consecutive GOES-16 CONUS Band-13 quick-look images on NOAA's public
# STAR/CIRA browse-imagery mirror. These are ordinary PNGs (no netCDF
# decoding required). Timestamps are illustrative; if unreachable we fall
# back automatically.
GOES_CANDIDATE_URLS = [
    (
        "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/13/"
        "20240601180031_GOES16-ABI-CONUS-13-1250x750.jpg"
    ),
    (
        "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/13/"
        "20240601181031_GOES16-ABI-CONUS-13-1250x750.jpg"
    ),
]


def _load_local(data_dir: str) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Load frame0/frame1(/frame2) from ``data_dir`` if all required files exist."""
    p0 = os.path.join(data_dir, "frame0.png")
    p1 = os.path.join(data_dir, "frame1.png")
    p2 = os.path.join(data_dir, "frame2.png")

    if not (os.path.exists(p0) and os.path.exists(p1)):
        return None

    f0 = cv2.imread(p0, cv2.IMREAD_GRAYSCALE)
    f1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
    f2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE) if os.path.exists(p2) else None
    if f0 is None or f1 is None:
        return None
    return f0, f1, f2


def _download_image(url: str, timeout: float = 6.0) -> Optional[np.ndarray]:
    """Download a single image from ``url`` and decode it as grayscale."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sat-frame-interp/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        return img
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return None


def _try_download_goes() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Attempt to fetch two consecutive GOES-16 Band-13 browse frames.

    Returns None if the network is unavailable, the host is blocked, or the
    remote objects have moved -- callers should fall back to synthetic data.
    """
    imgs = []
    for url in GOES_CANDIDATE_URLS:
        img = _download_image(url)
        if img is None:
            return None
        imgs.append(img)
    return imgs[0], imgs[1]


def _make_synthetic_pair(size: int = FRAME_SIZE, seed: int = 0):
    """Generate a synthetic "moving blob" toy triplet (t=0, t=0.5, t=1).

    Simulates a cloud-like Gaussian blob (with slowly evolving mass, similar
    to convective growth/decay in IR imagery) translating and slightly
    deforming across three time steps, so the pipeline can be exercised and
    quantitatively evaluated (frame2 as held-out ground truth) even fully
    offline.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)

    def blob_field(cx, cy, sx, sy, amp, theta=0.0):
        ct, st = np.cos(theta), np.sin(theta)
        xr = ct * (x - cx) + st * (y - cy)
        yr = -st * (x - cx) + ct * (y - cy)
        return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

    # Two blobs drifting in different directions to create nontrivial,
    # partially divergent motion (good stress-test for pure optical flow).
    start1, end1 = np.array([55.0, 70.0]), np.array([130.0, 100.0])
    start2, end2 = np.array([140.0, 140.0]), np.array([90.0, 60.0])

    frames = []
    for t in (0.0, 0.5, 1.0):
        c1 = start1 + t * (end1 - start1)
        c2 = start2 + t * (end2 - start2)
        sx1 = 18 + 4 * t
        sy1 = 14 + 2 * t
        sx2 = 12 - 3 * t
        sy2 = 16 + 3 * t
        amp1 = 0.85 - 0.15 * t   # slowly dissipating
        amp2 = 0.55 + 0.25 * t   # slowly intensifying (mass "merging" look)
        field = blob_field(c1[0], c1[1], sx1, sy1, amp1, theta=0.3)
        field += blob_field(c2[0], c2[1], sx2, sy2, amp2, theta=-0.5)
        field += 0.03 * rng.standard_normal((size, size)).astype(np.float32)
        field = np.clip(field, 0, 1)
        frames.append(field.astype(np.float32))

    return frames[0], frames[1], frames[2]


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Resize to FRAME_SIZE x FRAME_SIZE and normalize to [0, 1] float32."""
    img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


def get_frames(data_dir: str = "/data"):
    """Top-level acquisition entry point.

    Returns
    -------
    frame0, frame1 : np.ndarray (float32, HxW, [0, 1])
        The two frames to interpolate between.
    frame2 : np.ndarray or None
        Optional held-out ground-truth frame for metric evaluation.
    source : str
        Human-readable description of where the data came from.
    """
    os.makedirs(data_dir, exist_ok=True)

    local = _load_local(data_dir)
    if local is not None:
        f0, f1, f2 = local
        f2 = _preprocess(f2) if f2 is not None else None
        return _preprocess(f0), _preprocess(f1), f2, "local files in /data"

    remote = _try_download_goes()
    if remote is not None:
        f0, f1 = remote
        f0p, f1p = _preprocess(f0), _preprocess(f1)
        cv2.imwrite(os.path.join(data_dir, "frame0.png"), (f0p * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(data_dir, "frame1.png"), (f1p * 255).astype(np.uint8))
        return f0p, f1p, None, "downloaded GOES-16 CONUS Band-13 browse imagery"

    f0, f1, f2 = _make_synthetic_pair()
    cv2.imwrite(os.path.join(data_dir, "frame0.png"), (f0 * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(data_dir, "frame1.png"), (f1 * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(data_dir, "frame2.png"), (f2 * 255).astype(np.uint8))
    return f0, f1, f2, "synthetic moving-blob toy pair (network unavailable)"
