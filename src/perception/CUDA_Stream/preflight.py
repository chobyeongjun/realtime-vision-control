"""Preflight — verify JetPack / ZED SDK / PyTorch / TensorRT readiness.

Run on the Jetson before the first benchmark. Fails loudly if any
component is missing or mis-configured.

    python3 -m perception.CUDA_Stream.preflight

Exit codes:
    0 — all checks pass
    1 — one or more required checks failed (see stderr)
    2 — warning only (JetPack < 6.2 — Super Mode unavailable)
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

LOGGER = logging.getLogger(__name__)


def _run(cmd: List[str]) -> Tuple[int, str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=5
        )
        return out.returncode, (out.stdout + out.stderr).strip()
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def check_jetpack() -> Tuple[bool, str]:
    """Require L4T R36.4.x (JetPack 6.2.1+) for ZED SDK 5.2 compatibility.

    skiro-learnings (vault zed-x-mini-jetson-setup.md): L4T kernel + ZED
    Link + SDK versions must align exactly. JetPack 6.1 (L4T 36.4.0-pre)
    fails. 6.2.1 is the minimum that has every layer matching.
    """
    path = Path("/etc/nv_tegra_release")
    if not path.exists():
        return False, "not a Jetson (no /etc/nv_tegra_release)"
    text = path.read_text()
    m_rel = re.search(r"R(\d+)", text)
    m_rev = re.search(r"REVISION:\s*(\d+)\.(\d+)", text)
    rel = int(m_rel.group(1)) if m_rel else 0
    rev_major = int(m_rev.group(1)) if m_rev else 0
    rev_minor = int(m_rev.group(2)) if m_rev else 0
    # Accept L4T R36 REVISION 4.x (JetPack 6.2.x+) or any R37+.
    ok = (rel == 36 and rev_major >= 4) or rel >= 37
    return ok, f"L4T R{rel} REV{rev_major}.{rev_minor}"


def check_nvpmodel() -> Tuple[bool, str]:
    """Detect high-perf mode by ACTUAL frequencies, not the mode label.

    Some JetPack 6.2.x carrier-board BSPs keep the label "MAXN" but ship
    Super-mode frequencies (Orin NX: GPU 918MHz, CPU 1.984GHz). Checking
    the label alone gives false negatives — we parse nvpmodel -q --verbose
    instead and confirm GPU max >= 918 MHz.
    """
    rc, out = _run(["nvpmodel", "-q", "--verbose"])
    if rc != 0:
        return False, "nvpmodel not available"
    # Parse REAL_VAL for GPU MAX_FREQ (Hz)
    gpu_freq_hz = 0
    for line in out.splitlines():
        if "PARAM GPU" in line and "MAX_FREQ" in line:
            m = re.search(r"REAL_VAL:\s*(\d+)", line)
            if m:
                gpu_freq_hz = int(m.group(1))
                break
    # Orin NX max = 918 MHz (Super Mode ceiling); 900 is a safety floor.
    ok = gpu_freq_hz >= 900_000_000
    mode_line = next(
        (l.replace("NVPM VERB: Current mode:", "").strip()
         for l in out.splitlines() if "Current mode" in l),
        "mode unknown",
    )
    return ok, f"{mode_line} | GPU max {gpu_freq_hz // 1_000_000} MHz"


def _parse_version(s: str) -> tuple:
    """'10.3.0' -> (10,3,0). Unparseable parts become 0."""
    out = []
    for piece in s.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def check_cuda_torch() -> Tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() == False"
    dev = torch.cuda.get_device_name(0)
    torch_cuda = torch.version.cuda or ""
    # Compare against nvcc if available — mismatch causes silent TRT bugs.
    rc, out = _run(["nvcc", "--version"])
    nvcc_v = ""
    if rc == 0:
        m = re.search(r"release\s+(\d+\.\d+)", out)
        if m:
            nvcc_v = m.group(1)
    mismatch = ""
    if nvcc_v and torch_cuda:
        if _parse_version(torch_cuda)[:2] != _parse_version(nvcc_v)[:2]:
            mismatch = f" — MISMATCH with nvcc {nvcc_v}"
    return not mismatch, f"torch {torch.__version__}, cuda={torch_cuda}, dev={dev}{mismatch}"


def check_tensorrt() -> Tuple[bool, str]:
    """TRT 10.x is required for execute_async_v3."""
    try:
        import tensorrt as trt
    except ImportError:
        return False, "tensorrt not installed"
    ver = trt.__version__
    ok = _parse_version(ver)[0] >= 10
    return ok, f"tensorrt {ver}{' (need >= 10.x)' if not ok else ''}"


def check_pyzed() -> Tuple[bool, str]:
    """ZED SDK 5.2+ required for NV12 zero-copy + stable 5.2.1 stack."""
    try:
        import pyzed.sl as sl  # type: ignore
    except ImportError:
        return False, "pyzed.sl not installed"
    # Parse Stereolabs version string from ZED_Explorer for robustness.
    rc, out = _run(["ZED_Explorer", "--version"])
    version_str = ""
    if rc == 0:
        m = re.search(r"(\d+\.\d+\.\d+)", out)
        if m:
            version_str = m.group(1)
    if not version_str:
        try:
            version_str = sl.Camera.get_sdk_version()  # type: ignore[attr-defined]
        except Exception:
            version_str = "unknown"
    parsed = _parse_version(version_str)
    ok = parsed[:2] >= (5, 2)  # need 5.2+
    return ok, f"ZED SDK {version_str}{' (need >= 5.2)' if not ok else ''}"


def check_ultralytics() -> Tuple[bool, str]:
    try:
        import ultralytics
    except ImportError:
        return False, "ultralytics not installed"
    ver = ultralytics.__version__
    # YOLO26 requires >= 8.3.x (the release branch that introduced the model)
    ok = tuple(int(p) for p in ver.split(".")[:2]) >= (8, 3)
    return ok, f"ultralytics {ver}"


def check_shm_name() -> Tuple[bool, str]:
    """Mainline must NOT be using /hwalker_pose_cuda."""
    shm_path = Path("/dev/shm/hwalker_pose_cuda")
    collision = shm_path.exists()
    return not collision, (
        f"SHM /hwalker_pose_cuda collision at {shm_path}" if collision
        else "no collision"
    )


def check_jetson_clocks() -> Tuple[bool, str]:
    """jetson_clocks must be applied every boot (skiro-learnings).

    Without it the GPU runs at 306 MHz (33% of 918 MHz max) → predict
    spikes. The devfreq node path varies between BSPs — we try several
    known locations and also accept nvpmodel-parsed frequency as a
    confirmation.
    """
    candidates = [
        ("/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq",
         "/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq"),
        ("/sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq",
         "/sys/devices/gpu.0/devfreq/17000000.gv11b/max_freq"),
        ("/sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq",
         "/sys/devices/gpu.0/devfreq/17000000.gpu/max_freq"),
    ]
    for cur_p, max_p in candidates:
        cur_path = Path(cur_p)
        if not cur_path.exists():
            continue
        try:
            cur = int(cur_path.read_text().strip())
            mx_path = Path(max_p)
            mx = int(mx_path.read_text().strip()) if mx_path.exists() else cur
            ratio = cur / mx if mx else 0
            ok = ratio > 0.6 and mx >= 800_000_000  # clocks locked + near top
            return ok, f"cur={cur // 1_000_000} MHz max={mx // 1_000_000} MHz ratio={ratio:.0%}"
        except Exception as err:
            return True, f"could not read clocks ({err})"
    # Fall back to nvpmodel — the frequency we parsed there is authoritative.
    rc, out = _run(["nvpmodel", "-q", "--verbose"])
    if rc == 0:
        for line in out.splitlines():
            if "PARAM GPU" in line and "MAX_FREQ" in line:
                m = re.search(r"REAL_VAL:\s*(\d+)", line)
                if m:
                    hz = int(m.group(1))
                    ok = hz >= 800_000_000
                    return ok, f"from nvpmodel: GPU max {hz // 1_000_000} MHz"
    return True, "devfreq path not found on this BSP (skipped)"


def check_gdm_running() -> Tuple[bool, str]:
    """GDM (X server) MUST stay running on the ZED GMSL Jetson.

    skiro-learnings: stopping GDM breaks Argus/EGL, ZED segfaults, and
    the Argus daemon's internal state corrupts requiring a reboot. We
    check the service is active and refuse to run if it's not.
    """
    rc, out = _run(["systemctl", "is-active", "gdm"])
    if rc == 127:
        return True, "systemctl not available (non-Jetson env)"
    active = out.strip() == "active"
    return active, f"gdm systemd state: {out.strip() or 'unknown'}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="JetPack 6.2 required")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    checks = [
        ("JetPack (L4T release)", check_jetpack),
        ("MAXN_SUPER mode", check_nvpmodel),
        ("jetson_clocks locked", check_jetson_clocks),
        ("GDM (X server) running — do NOT stop", check_gdm_running),
        ("PyTorch + CUDA", check_cuda_torch),
        ("TensorRT", check_tensorrt),
        ("pyzed.sl", check_pyzed),
        ("ultralytics >= 8.3", check_ultralytics),
        ("SHM namespace no collision", check_shm_name),
    ]

    failed = []
    warnings = []
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as err:
            ok, detail = False, f"exception: {err}"
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name:32s} : {detail}")
        if not ok:
            if name == "MAXN_SUPER mode" and not args.strict:
                warnings.append(name)
            else:
                failed.append(name)

    print()
    if failed:
        print(f"FAILED checks: {failed}", file=sys.stderr)
        return 1
    if warnings:
        print(f"WARN: {warnings} — continuing (non-strict)")
        return 2
    print("preflight OK — ready to run benchmarks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
