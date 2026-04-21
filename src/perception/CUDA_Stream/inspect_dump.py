"""Inspect a pose dump (.npz) for anatomical anomalies.

Scans every frame and reports how often each keypoint-level sanity
check fails. Useful for tuning the gate thresholds used by
``view_sagittal --display-mode ... --sanity``.

Usage::

    PYTHONPATH=src python3 -m perception.CUDA_Stream.inspect_dump \\
        data/recordings/2026-04-21/walk_01_pose.npz

The script is pure-Python / numpy so it runs on Mac just like view_sagittal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .keypoint_config import get_schema


# Default anatomical bounds (meters). Generous — tune from dump statistics.
DEFAULT_BOUNDS = {
    "hip_width":  (0.15, 0.60),   # L-hip ↔ R-hip
    "thigh":      (0.25, 0.60),   # hip ↔ knee, each side
    "shank":      (0.20, 0.55),   # knee ↔ ankle, each side
    "asym_max":   1.30,           # max(L, R) / min(L, R)
}


def print_stat(name: str, arr: np.ndarray, unit_cm: bool = True) -> None:
    a = np.asarray(arr, dtype=np.float64)
    k = 100.0 if unit_cm else 1.0
    suf = "cm" if unit_cm else ""
    print(f"  {name:18s}: mean={a.mean()*k:6.1f}{suf}   "
          f"median={np.median(a)*k:6.1f}{suf}   "
          f"p5={np.percentile(a, 5)*k:6.1f}{suf}   "
          f"p95={np.percentile(a, 95)*k:6.1f}{suf}   "
          f"min={a.min()*k:6.1f}   max={a.max()*k:6.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="path to .npz dump")
    ap.add_argument("--schema", default="lowlimb6", choices=["coco17", "lowlimb6"])
    ap.add_argument("--hip-width", nargs=2, type=float, metavar=("LO", "HI"),
                    default=DEFAULT_BOUNDS["hip_width"])
    ap.add_argument("--thigh", nargs=2, type=float, metavar=("LO", "HI"),
                    default=DEFAULT_BOUNDS["thigh"])
    ap.add_argument("--shank", nargs=2, type=float, metavar=("LO", "HI"),
                    default=DEFAULT_BOUNDS["shank"])
    ap.add_argument("--asym-max", type=float, default=DEFAULT_BOUNDS["asym_max"])
    ap.add_argument("--knee-angle", nargs=2, type=float,
                    metavar=("LO", "HI"), default=(30.0, 185.0),
                    help="knee flexion angle acceptable range, degrees")
    ap.add_argument("--ank-above-knee-tol", type=float, default=0.10,
                    help="ankle allowed to sit above knee by this many metres")
    ap.add_argument("--knee-above-hip-tol", type=float, default=0.20,
                    help="knee allowed to sit above hip by this many metres")
    ap.add_argument("--valid-only", dest="valid_only", action="store_true",
                    help="ignore SHM-invalid frames (where valid=False). "
                         "Recommended — invalid frames have zero'd keypoints "
                         "which skew every distance statistic to 0.")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 2

    d = np.load(path, allow_pickle=True)
    kpts_3d_all = d["kpts_3d"].astype(np.float64)      # (N, K, 3)
    kpt_conf_all = d["kpt_conf"].astype(np.float64)    # (N, K)
    valid_all = d["valid"]                             # (N,)
    N_all, K, _ = kpts_3d_all.shape

    schema = get_schema(args.schema)
    if K != schema.num_keypoints:
        print(f"schema mismatch: dump K={K}, schema K={schema.num_keypoints}",
              file=sys.stderr)
        return 3
    idx = {name: i for i, name in enumerate(schema.keypoints)}

    # Filter to valid-only frames if requested. Invalid frames have zero'd
    # keypoints that contaminate every distance statistic.
    if args.valid_only:
        mask = np.asarray(valid_all, dtype=bool)
        kpts_3d = kpts_3d_all[mask]
        kpt_conf = kpt_conf_all[mask]
        valid = valid_all[mask]
    else:
        kpts_3d = kpts_3d_all
        kpt_conf = kpt_conf_all
        valid = valid_all
    N = kpts_3d.shape[0]

    print(f"File       : {path}")
    print(f"Frames     : total={N_all}   valid_from_SHM={int(valid_all.sum())}"
          f"   analysed={N}"
          f"{'   (valid-only)' if args.valid_only else ''}")
    print(f"Keypoints  : {K}  ({', '.join(schema.keypoints)})")
    print()

    if N == 0:
        print("No frames to analyse.", file=sys.stderr)
        return 0

    def dist(a: str, b: str) -> np.ndarray:
        ia, ib = idx[a], idx[b]
        return np.linalg.norm(kpts_3d[:, ia] - kpts_3d[:, ib], axis=1)

    # ── Bone lengths ──────────────────────────────────────────────────────
    hw = dist("left_hip", "right_hip")
    lt = dist("left_hip", "left_knee")
    rt = dist("right_hip", "right_knee")
    ls = dist("left_knee", "left_ankle")
    rs = dist("right_knee", "right_ankle")

    print("Bone length statistics (all frames):")
    print_stat("hip_width",  hw)
    print_stat("L_thigh",    lt)
    print_stat("R_thigh",    rt)
    print_stat("L_shank",    ls)
    print_stat("R_shank",    rs)
    print()

    # ── Bilateral asymmetry ──────────────────────────────────────────────
    thigh_asym = np.maximum(lt, rt) / np.maximum(np.minimum(lt, rt), 1e-6)
    shank_asym = np.maximum(ls, rs) / np.maximum(np.minimum(ls, rs), 1e-6)
    print("Bilateral asymmetry (max/min ratio, 1.0 = symmetric):")
    print_stat("thigh_asym", thigh_asym, unit_cm=False)
    print_stat("shank_asym", shank_asym, unit_cm=False)
    print()

    # ── Knee flexion angles and orientation ───────────────────────────────
    def knee_angle(side: str) -> np.ndarray:
        ih = idx[f"{side}_hip"]
        ik = idx[f"{side}_knee"]
        ia = idx[f"{side}_ankle"]
        v1 = kpts_3d[:, ih] - kpts_3d[:, ik]
        v2 = kpts_3d[:, ia] - kpts_3d[:, ik]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        ok = (n1 > 1e-3) & (n2 > 1e-3)
        dot = np.einsum('ij,ij->i', v1, v2)
        cos = np.divide(dot, n1 * n2, out=np.zeros_like(dot), where=ok)
        cos = np.clip(cos, -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        ang[~ok] = np.nan
        return ang

    lka = knee_angle("left")
    rka = knee_angle("right")
    print("Knee flexion angle (degrees, 180 = straight leg):")
    if np.isfinite(lka).any():
        print_stat("L_knee_angle", lka[np.isfinite(lka)], unit_cm=False)
    if np.isfinite(rka).any():
        print_stat("R_knee_angle", rka[np.isfinite(rka)], unit_cm=False)
    print()

    # Y ordering: world +Y = down, so ankle_y > knee_y > hip_y when standing.
    ilh, irh = idx["left_hip"], idx["right_hip"]
    ilk, irk = idx["left_knee"], idx["right_knee"]
    ila, ira = idx["left_ankle"], idx["right_ankle"]
    l_ank_above_knee_m = kpts_3d[:, ilk, 1] - kpts_3d[:, ila, 1]   # positive if violation
    r_ank_above_knee_m = kpts_3d[:, irk, 1] - kpts_3d[:, ira, 1]
    l_knee_above_hip_m = kpts_3d[:, ilh, 1] - kpts_3d[:, ilk, 1]
    r_knee_above_hip_m = kpts_3d[:, irh, 1] - kpts_3d[:, irk, 1]
    print("Y-inversion margins (metres; positive = anatomically impossible):")
    print_stat("L_ank_above_knee", l_ank_above_knee_m, unit_cm=False)
    print_stat("R_ank_above_knee", r_ank_above_knee_m, unit_cm=False)
    print_stat("L_knee_above_hip", l_knee_above_hip_m, unit_cm=False)
    print_stat("R_knee_above_hip", r_knee_above_hip_m, unit_cm=False)
    print()

    # ── Confidence per keypoint ──────────────────────────────────────────
    print("Keypoint confidence (mean / p5 / p95 / fraction<0.3):")
    for name in schema.keypoints:
        i = idx[name]
        c = kpt_conf[:, i]
        print(f"  {name:15s}: mean={c.mean():.3f}  p5={np.percentile(c,5):.3f}  "
              f"p95={np.percentile(c,95):.3f}  <0.3: {(c<0.3).mean()*100:5.1f}%")
    print()

    # ── Apply gate and count failures ────────────────────────────────────
    hw_lo, hw_hi = args.hip_width
    t_lo, t_hi = args.thigh
    s_lo, s_hi = args.shank
    a_max = args.asym_max

    ka_lo, ka_hi = args.knee_angle
    ank_tol = args.ank_above_knee_tol
    knee_tol = args.knee_above_hip_tol

    failures = {
        "hip_width_oor":  (hw < hw_lo) | (hw > hw_hi),
        "L_thigh_oor":    (lt < t_lo) | (lt > t_hi),
        "R_thigh_oor":    (rt < t_lo) | (rt > t_hi),
        "L_shank_oor":    (ls < s_lo) | (ls > s_hi),
        "R_shank_oor":    (rs < s_lo) | (rs > s_hi),
        "thigh_asym":     thigh_asym > a_max,
        "shank_asym":     shank_asym > a_max,
        "L_ank_above_knee": l_ank_above_knee_m > ank_tol,
        "R_ank_above_knee": r_ank_above_knee_m > ank_tol,
        "L_knee_above_hip": l_knee_above_hip_m > knee_tol,
        "R_knee_above_hip": r_knee_above_hip_m > knee_tol,
        "L_knee_angle_oor": np.isfinite(lka) & ((lka < ka_lo) | (lka > ka_hi)),
        "R_knee_angle_oor": np.isfinite(rka) & ((rka < ka_lo) | (rka > ka_hi)),
    }
    any_fail = np.zeros(N, dtype=bool)

    print("Per-check failure rates (gate below):")
    print(f"  hip_width        : {hw_lo*100:.0f}–{hw_hi*100:.0f} cm")
    print(f"  thigh            : {t_lo*100:.0f}–{t_hi*100:.0f} cm   (each side)")
    print(f"  shank            : {s_lo*100:.0f}–{s_hi*100:.0f} cm   (each side)")
    print(f"  asym_max         : {a_max:.2f}   (L/R ratio)")
    print(f"  knee_angle       : {ka_lo:.0f}–{ka_hi:.0f}°")
    print(f"  ank_above_knee   : reject if ankle sits > {ank_tol*100:.0f} cm above knee")
    print(f"  knee_above_hip   : reject if knee  sits > {knee_tol*100:.0f} cm above hip")
    print()
    for k, v in failures.items():
        any_fail |= v
        pct = v.mean() * 100
        bar = "█" * int(pct / 2)
        print(f"  {k:18s}: {int(v.sum()):4d} / {N}  ({pct:5.1f}%) {bar}")
    print()
    pct_any = any_fail.mean() * 100
    print(f"  ANY FAILURE       : {int(any_fail.sum()):4d} / {N}  ({pct_any:5.1f}%)")
    print(f"  CLEAN             : {int((~any_fail).sum()):4d} / {N}  "
          f"({100-pct_any:5.1f}%)")
    print()

    # ── Worst 5 frames (hint for debugging) ──────────────────────────────
    scores = (
        np.maximum(0, hw - hw_hi) + np.maximum(0, hw_lo - hw)
        + np.maximum(0, lt - t_hi) + np.maximum(0, t_lo - lt)
        + np.maximum(0, rt - t_hi) + np.maximum(0, t_lo - rt)
        + np.maximum(0, ls - s_hi) + np.maximum(0, s_lo - ls)
        + np.maximum(0, rs - s_hi) + np.maximum(0, s_lo - rs)
        + np.maximum(0, thigh_asym - a_max)
        + np.maximum(0, shank_asym - a_max)
    )
    worst = np.argsort(-scores)[:5]
    print("Worst 5 frames by violation magnitude:")
    for i in worst:
        reasons = [k for k, v in failures.items() if v[i]]
        print(f"  idx={int(i):4d}  hw={hw[i]*100:5.1f}cm  "
              f"Lt={lt[i]*100:5.1f}  Rt={rt[i]*100:5.1f}  "
              f"Ls={ls[i]*100:5.1f}  Rs={rs[i]*100:5.1f}  "
              f"TA={thigh_asym[i]:.2f}  SA={shank_asym[i]:.2f}   "
              f"[{', '.join(reasons)}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
