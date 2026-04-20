"""CUDA_Stream — H-Walker Perception GPU pipeline (isolated experimental module).

Mainline files under ``src/perception/realtime/`` and
``src/perception/benchmarks/`` are READ-ONLY from this module.
SHM publish namespace is ``/hwalker_pose_cuda`` — must never collide with
mainline's ``/hwalker_pose``.
"""

__all__ = [
    "stream_manager",
    "trt_runner",
    "zed_gpu_bridge",
    "gpu_preprocess",
    "gpu_postprocess",
    "pipeline",
    "cuda_graph",
    "shm_publisher",
    "watchdog",
]

__version__ = "0.1.0"
SHM_NAMESPACE = "/hwalker_pose_cuda"
MAINLINE_SHM = "/hwalker_pose"  # for collision checks only — never write here
