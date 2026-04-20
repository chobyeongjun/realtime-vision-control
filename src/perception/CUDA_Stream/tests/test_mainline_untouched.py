"""Contract test — CUDA_Stream module must not import-for-write from mainline.

Scans every .py under CUDA_Stream and asserts that we don't import from the
forbidden mainline packages with modifications in mind. (Read-only helpers
are allowed but we don't reference them here yet, so the list is empty.)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_WRITE_TARGETS = {
    "perception.realtime",
    "perception.benchmarks.run_benchmark",
    "perception.benchmarks.zed_camera",
    "perception.benchmarks.pose_models",
}
SHM_REGEX = re.compile(r"['\"]/?hwalker_pose(?!_cuda)")


@pytest.mark.parametrize(
    "path",
    [p for p in ROOT.rglob("*.py") if "tests" not in p.parts],
)
def test_no_mainline_shm_write(path: Path):
    source = path.read_text()
    # allow /hwalker_pose_cuda (our name) — forbid bare /hwalker_pose usage
    for match in SHM_REGEX.finditer(source):
        # check it's not inside a comment about the mainline name
        line_start = source.rfind("\n", 0, match.start()) + 1
        line = source[line_start : source.find("\n", match.end())]
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "MAINLINE_SHM" in line or "FORBIDDEN_NAMES" in line or "forbidden" in line.lower():
            continue
        if "mainline" in line.lower() and "collision" in line.lower():
            continue
        if "must never" in line.lower() or "must not" in line.lower():
            continue
        raise AssertionError(
            f"{path}:{source[:match.start()].count(chr(10)) + 1} "
            f"references mainline SHM: {line.strip()}"
        )


def test_forbidden_imports_absent():
    for path in ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        src = path.read_text()
        for target in FORBIDDEN_WRITE_TARGETS:
            if f"from {target} import" in src or f"import {target}" in src:
                # we allow pure reads of constants, but flag any modification.
                # since none of our modules import mainline right now, this
                # is a hard no.
                raise AssertionError(
                    f"{path} imports from forbidden mainline target {target}"
                )
