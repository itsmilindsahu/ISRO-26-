#!/usr/bin/env python3
"""
interpolate.py
===============
Satellite-image frame interpolation via a *flow-warped Wasserstein-barycenter
hybrid*.

Pipeline
--------
1.  Farneback optical flow gives a coarse, dense motion prior between
    frame0 and frame1.
2.  Both frames are pre-warped toward the target time ``t`` using that flow
    (frame0 forward by ``t``, frame1 backward by ``1-t``). This handles bulk
    advection cheaply but cannot represent mass that appears, disappears, or
    splits (e.g. a convective cell growing, or one cloud cluster separating
    into two) -- flow-only warping "smears" or "ghosts" in those regions.
3.  The two pre-warped frames are treated as unnormalized mass
    distributions and refined with an entropic-regularized Wasserstein
    (Sinkhorn) barycenter, computed in convolutional/heat-kernel form via
    POT's ``convolutional_barycenter2d``, using barycentric weights
    ``[1-t, t]``. This corrects residual warp artifacts and lets mass
    split/merge/redistribute in a way that is optimal under a transport
    cost, rather than being simply blended.

Also provided for comparison:
    * ``linear_blend``    -- naive pixel-wise cross-dissolve.
    * ``flow_only_warp``  -- pure optical-flow warp, no OT refinement.

Public API
----------
    interpolate(frame0, frame1, t, reg=0.004) -> np.ndarray

CLI
---
    python interpolate.py --t 0.5
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from typing import Tuple

import cv2
import numpy as np

try:
    from ot.bregman import convolutional_barycenter2d
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "POT (Python Optimal Transport) is required: pip install POT"
    ) from exc

import data_prep
import metrics as metrics_mod

EPS = 1e-8


# --------------------------------------------------------------------------
# Optical flow + warping
# --------------------------------------------------------------------------

def compute_farneback_flow(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """Coarse dense motion prior from ``frame_a`` to ``frame_b``.

    Parameters
    ----------
    frame_a, frame_b : np.ndarray, float32 in [0, 1]

    Returns
    -------
    flow : np.ndarray, shape (H, W, 2)
        Per-pixel (dx, dy) displacement such that frame_a warped by this
        flow approximates frame_b.
    """
    a8 = (np.clip(frame_a, 0, 1) * 255).astype(np.uint8)
    b8 = (np.clip(frame_b, 0, 1) * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        a8, b8, None,
        pyr_scale=0.5, levels=4, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow


def warp_by_flow(img: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
    """Forward-warp ``img`` along ``flow`` scaled by ``scale`` (e.g. ``t``).

    Uses inverse (remap-based) sampling: for each destination pixel we look
    up the source location ``pos - scale * flow`` -- a standard
    approximation that avoids splatting holes for small-to-moderate motion.
    """
    h, w = img.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x - scale * flow[..., 0]
    map_y = grid_y - scale * flow[..., 1]
    warped = cv2.remap(
        img.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


# --------------------------------------------------------------------------
# Baseline methods
# --------------------------------------------------------------------------

def linear_blend(frame0: np.ndarray, frame1: np.ndarray, t: float) -> np.ndarray:
    """Naive pixel-wise cross-dissolve baseline: ``(1-t)*frame0 + t*frame1``."""
    return ((1.0 - t) * frame0 + t * frame1).astype(np.float32)


def flow_only_warp(frame0: np.ndarray, frame1: np.ndarray, t: float,
                    flow01: np.ndarray = None, flow10: np.ndarray = None) -> np.ndarray:
    """Pure optical-flow warp interpolation, without any OT refinement.

    Warps frame0 forward by ``t`` and frame1 backward by ``1-t`` using
    bidirectional Farneback flow, then blends the two warped estimates.
    """
    if flow01 is None:
        flow01 = compute_farneback_flow(frame0, frame1)
    if flow10 is None:
        flow10 = compute_farneback_flow(frame1, frame0)

    warped0 = warp_by_flow(frame0, flow01, scale=t)
    warped1 = warp_by_flow(frame1, flow10, scale=(1.0 - t))
    return ((1.0 - t) * warped0 + t * warped1).astype(np.float32)


# --------------------------------------------------------------------------
# OT barycenter refinement (Benamou-Brenier / Wasserstein-geodesic hybrid)
# --------------------------------------------------------------------------

def _to_distribution(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalize a nonnegative image into a probability distribution.

    Returns the normalized distribution and the original total mass, so the
    barycenter result can later be rescaled back to a comparable photometric
    range.
    """
    img = np.clip(img, 0.0, None).astype(np.float64) + EPS
    mass = float(img.sum())
    return img / mass, mass


def ot_barycenter_refine(warped0: np.ndarray, warped1: np.ndarray, t: float,
                          reg: float = 0.004) -> np.ndarray:
    """Refine two flow-pre-warped frames via a convolutional Wasserstein
    (Sinkhorn) barycenter with weights ``[1-t, t]``.

    This is the discrete, entropic-regularized analogue of a
    Benamou-Brenier / Wasserstein-geodesic interpolation: rather than
    linearly blending intensities (which "ghosts" wherever mass moves,
    grows, or splits), the barycenter redistributes mass along
    approximately optimal-transport paths between the two pre-warped
    distributions, which is well suited to correcting residual warp
    artifacts and handling mass splitting/merging (e.g. clouds separating
    or merging) that pure flow-warping cannot represent.

    Parameters
    ----------
    warped0, warped1 : np.ndarray
        The flow-pre-warped estimates of frame0 and frame1 at time ``t``.
    t : float
        Interpolation parameter in [0, 1]; barycentric weights are
        ``[1-t, t]``.
    reg : float
        Entropic regularization strength (heat-kernel scale). Smaller is
        sharper but slower/less stable; larger is smoother/faster.

    Returns
    -------
    np.ndarray, float32 in [0, 1]
        The OT-refined interpolated frame.
    """
    dist0, mass0 = _to_distribution(warped0)
    dist1, mass1 = _to_distribution(warped1)

    A = np.stack([dist0, dist1], axis=0)
    weights = np.array([1.0 - t, t], dtype=np.float64)

    bary = convolutional_barycenter2d(A, reg=reg, weights=weights)

    # Rescale the (unit-mass) barycenter back to a comparable photometric
    # range by matching the barycentrically-interpolated total mass.
    target_mass = (1.0 - t) * mass0 + t * mass1
    result = bary * target_mass
    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    return result


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def interpolate(frame0: np.ndarray, frame1: np.ndarray, t: float,
                 reg: float = 0.004) -> np.ndarray:
    """Hybrid flow-warp + Wasserstein-barycenter frame interpolation.

    Parameters
    ----------
    frame0, frame1 : np.ndarray
        Grayscale frames, float32, values in [0, 1], identical shape.
    t : float
        Interpolation parameter in [0, 1] (0 -> frame0, 1 -> frame1).
    reg : float
        Entropic regularization for the Sinkhorn barycenter refinement.

    Returns
    -------
    np.ndarray, float32 in [0, 1]
        The interpolated frame at time ``t``.
    """
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 1e-6:
        return frame0.astype(np.float32).copy()
    if t >= 1.0 - 1e-6:
        return frame1.astype(np.float32).copy()

    flow01 = compute_farneback_flow(frame0, frame1)
    flow10 = compute_farneback_flow(frame1, frame0)

    warped0 = warp_by_flow(frame0, flow01, scale=t)
    warped1 = warp_by_flow(frame1, flow10, scale=(1.0 - t))

    refined = ot_barycenter_refine(warped0, warped1, t, reg=reg)
    return refined


# --------------------------------------------------------------------------
# Visualization / evaluation helpers
# --------------------------------------------------------------------------

def _save_frame(img: np.ndarray, path: str) -> None:
    cv2.imwrite(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))


def _make_comparison_grid(images: dict, metrics_by_name: dict, out_path: str, t: float) -> None:
    """Save a labeled comparison grid PNG using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(images.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        ax.imshow(images[name], cmap="gray", vmin=0, vmax=1)
        title = name
        if name in metrics_by_name:
            m = metrics_by_name[name]
            title += f"\nPSNR {m['psnr']:.2f} dB | SSIM {m['ssim']:.3f}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Satellite Frame Interpolation Comparison (t={t:.2f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _zip_output(project_dir: str, output_dir: str, zip_path: str) -> None:
    """Zip the /output folder plus all source (.py) files and README."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for f in files:
                full = os.path.join(root, f)
                arcname = os.path.join("output", os.path.relpath(full, output_dir))
                zf.write(full, arcname)

        for f in sorted(os.listdir(project_dir)):
            full = os.path.join(project_dir, f)
            if os.path.isfile(full) and (f.endswith(".py") or f == "README.md"):
                zf.write(full, os.path.join("src", f))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Satellite frame interpolation via flow-warp + Wasserstein-barycenter refinement."
    )
    parser.add_argument("--t", type=float, default=0.5, help="Interpolation parameter in [0, 1].")
    parser.add_argument("--reg", type=float, default=0.004, help="Sinkhorn entropic regularization.")
    parser.add_argument("--data-dir", type=str, default="/data", help="Directory with frame0/frame1(/frame2).png.")
    parser.add_argument("--output-dir", type=str, default="/output", help="Directory to write outputs to.")
    parser.add_argument("--zip", type=str, default=None,
                         help="Path for the final zip archive (default: <project_dir>/sat-frame-interp.zip).")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output_dir, exist_ok=True)

    frame0, frame1, frame2, source = data_prep.get_frames(args.data_dir)
    print(f"[data] source: {source}")
    print(f"[data] frame shape: {frame0.shape}, t={args.t}, reg={args.reg}")

    # --- Run all three methods ---
    lin = linear_blend(frame0, frame1, args.t)

    flow01 = compute_farneback_flow(frame0, frame1)
    flow10 = compute_farneback_flow(frame1, frame0)
    flow_only = flow_only_warp(frame0, frame1, args.t, flow01=flow01, flow10=flow10)

    # Use the public API directly so CLI behavior always matches `interpolate()`.
    hybrid = interpolate(frame0, frame1, args.t, reg=args.reg)

    # --- Metrics against held-out true frame, if available ---
    metrics_by_name = {}
    if frame2 is not None:
        for name, img in [("linear_blend", lin), ("flow_only_warp", flow_only), ("hybrid_ot", hybrid)]:
            metrics_by_name[name] = metrics_mod.compute_metrics(img, frame2)
            m = metrics_by_name[name]
            print(f"[metrics] {name:16s} PSNR={m['psnr']:6.2f} dB  SSIM={m['ssim']:.4f}")
    else:
        print("[metrics] no held-out frame2 available -- skipping PSNR/SSIM.")

    # --- Save individual outputs ---
    _save_frame(frame0, os.path.join(args.output_dir, "frame0.png"))
    _save_frame(frame1, os.path.join(args.output_dir, "frame1.png"))
    _save_frame(lin, os.path.join(args.output_dir, "linear_blend.png"))
    _save_frame(flow_only, os.path.join(args.output_dir, "flow_only_warp.png"))
    _save_frame(hybrid, os.path.join(args.output_dir, "hybrid_ot_result.png"))
    if frame2 is not None:
        _save_frame(frame2, os.path.join(args.output_dir, "frame2_ground_truth.png"))

    # --- Comparison grid ---
    grid_images = {
        "frame0 (t=0)": frame0,
        "linear blend": lin,
        "flow-only warp": flow_only,
        "hybrid OT (ours)": hybrid,
        "frame1 (t=1)": frame1,
    }
    if frame2 is not None:
        grid_images["ground truth"] = frame2
    _make_comparison_grid(grid_images, metrics_by_name,
                           os.path.join(args.output_dir, "comparison_grid.png"), args.t)

    # --- Write a small metrics summary text file ---
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as fh:
        fh.write(f"source: {source}\n")
        fh.write(f"t={args.t}  reg={args.reg}\n")
        if metrics_by_name:
            for name, m in metrics_by_name.items():
                fh.write(f"{name}: PSNR={m['psnr']:.4f} dB, SSIM={m['ssim']:.4f}\n")
        else:
            fh.write("no held-out ground-truth frame available\n")

    print(f"[output] saved images + comparison grid to {args.output_dir}")

    # --- Package everything ---
    zip_path = args.zip or os.path.join(project_dir, "sat-frame-interp.zip")
    _zip_output(project_dir, args.output_dir, zip_path)
    print(f"[package] wrote {zip_path}")


if __name__ == "__main__":
    main()
