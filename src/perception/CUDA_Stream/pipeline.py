"""Triple-buffer 3-stage overlapped pipeline.

At any instant:
    capture_stream   : grabbing frame N+1
    preproc_stream   : letterbox/normalize frame N+1 (after capture done)
    infer_stream     : TRT inference on frame N
    post_stream      : 3D/filter/publish frame N-1

Cross-stream dependencies flow via ``torch.cuda.Event`` only. Host thread
steps once per frame: pick latest ZEDFrame → advance events → wait on the
post_stream event that corresponds to the frame we want to return.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import torch

from .constraints import ConstraintStack
from .gpu_postprocess import GpuPostprocessor, PoseResult
from .gpu_preprocess import GpuPreprocessor, LetterboxParams
from .stream_manager import StreamManager
from .tracer import PipelineTracer
from .trt_runner import TRTRunner
from .zed_gpu_bridge import ZEDFrame, ZEDGpuBridge

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineTick:
    """Output of one pipeline step — what the host consumer receives."""

    frame_id: int
    ts_ns: int
    result: PoseResult
    latency_ms: dict  # {"grab": ..., "preproc": ..., "infer": ..., "post": ..., "e2e": ...}


class StreamedPosePipeline:
    """3-stage GPU pipeline with 4 streams."""

    def __init__(
        self,
        bridge: ZEDGpuBridge,
        runner: TRTRunner,
        preprocessor: GpuPreprocessor,
        postprocessor: GpuPostprocessor,
        streams: StreamManager,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        constraints: Optional[ConstraintStack] = None,
        tracer: Optional[PipelineTracer] = None,
    ) -> None:
        self.bridge = bridge
        self.runner = runner
        self.pre = preprocessor
        self.post = postprocessor
        self.sm = streams
        # Default: no constraints (OFF). Callers opt in by passing a
        # ConstraintStack with calibrated bone_length / joint_velocity.
        # See constraints.py for the rationale.
        self.constraints = constraints or ConstraintStack()
        # Tracer is OFF by default — pass one from run_stream_demo/benchmark_stream
        # with --trace to enable per-stage CUDA-event timing + CSV dump.
        self.tracer = tracer or PipelineTracer(enabled=False)

        self._input = input_name or runner.input_names[0]
        self._output = output_name or runner.output_names[0]

        # in-flight bookkeeping for overlap (not true triple buffer — we
        # keep 3 "tokens" representing capture/infer/post frames)
        self._pending: Deque[dict] = deque(maxlen=3)

        # Prime every stream's done_event once so any consumer that calls
        # ``wait_for(X)`` on frame 0 has a defined event to wait on.
        for bundle in streams.streams.values():
            bundle.record_done()

        # (Reserved for future fallback use — currently constraint rejects
        # emit zeros+valid=False instead of using stale data. Keeping this
        # as a field documented for future work; DO NOT read it elsewhere
        # without first deciding how it interacts with the valid=False
        # safety contract.)

    # ------------------------------------------------------------------
    # Single step (for benchmarks / reference correctness)
    # ------------------------------------------------------------------
    def run_once(self, frame: ZEDFrame) -> PipelineTick:
        """Run a single frame end-to-end in a way the caller can verify.

        This serializes stages but still uses explicit streams / events.
        Used by tests and the `--no-overlap` flag in the benchmark.
        """
        t_start = time.perf_counter()
        cap = self.sm.bundle("capture")
        pre = self.sm.bundle("preproc")
        inf = self.sm.bundle("infer")
        po = self.sm.bundle("post")

        # Preproc must wait for ZED's private H2D stream to finish the copy.
        if frame.ready_event is not None:
            pre.stream.wait_event(frame.ready_event)
        else:
            pre.wait_for(cap)
        _, lb = self.pre(frame.rgb_gpu, stream=pre.stream)
        # bind preproc output as TRT input (zero-copy)
        self.runner.bind_input_address(self._input, self.pre.out)
        pre.record_done()

        inf.wait_for(pre)
        with torch.cuda.stream(inf.stream):
            self.runner.infer_async(self.sm.stream_ptr("infer"))
        inf.record_done()

        po.wait_for(inf)
        result = self.post(
            raw_output=self.runner.get_output(self._output),
            depth_hw=frame.depth_gpu,
            lb_params=lb,
            calibration=frame.calibration,
            stream=po.stream,
            ts_s=frame.ts_ns * 1e-9,
        )
        po.record_done()
        po.stream.synchronize()

        t_end = time.perf_counter()
        return PipelineTick(
            frame_id=frame.frame_id,
            ts_ns=frame.ts_ns,
            result=result,
            latency_ms={"e2e": (t_end - t_start) * 1e3},
        )

    # ------------------------------------------------------------------
    # Overlapped run — the real deal
    # ------------------------------------------------------------------
    def run_overlapped_step(self) -> Optional[PipelineTick]:
        """Consume the latest ZED frame, advance streams, return last finished."""
        frame = self.bridge.latest(timeout=0.5)
        if frame is None:
            return None

        cap = self.sm.bundle("capture")
        pre = self.sm.bundle("preproc")
        inf = self.sm.bundle("infer")
        po = self.sm.bundle("post")

        self.tracer.begin(frame_id=frame.frame_id, ts_ns=frame.ts_ns)
        t_start = time.perf_counter()

        # --- stage A: preproc (on preproc_stream; rgb is already GPU)
        # Wait on the ZED bridge's H2D completion event, not on cap.
        if frame.ready_event is not None:
            pre.stream.wait_event(frame.ready_event)
        else:
            pre.wait_for(cap)
        # NOTE: cap_ms is intentionally NOT tracked here — the ZED H2D
        # happens on a stream inside ZEDGpuBridge that we don't own.
        # e2e_ms minus (pre+inf+post) approximates the capture overhead.
        self.tracer.mark_start("pre", pre.stream)
        _, lb = self.pre(frame.rgb_gpu, stream=pre.stream)
        self.tracer.mark_end("pre", pre.stream)
        pre.record_done()

        # --- stage B: infer (binds preproc output, waits on preproc)
        inf.wait_for(pre)
        self.runner.bind_input_address(self._input, self.pre.out)
        self.tracer.mark_start("inf", inf.stream)
        with torch.cuda.stream(inf.stream):
            self.runner.infer_async(self.sm.stream_ptr("infer"))
        self.tracer.mark_end("inf", inf.stream)
        inf.record_done()

        # --- stage C: post (waits on infer)
        po.wait_for(inf)
        self.tracer.mark_start("post", po.stream)
        result = self.post(
            raw_output=self.runner.get_output(self._output),
            depth_hw=frame.depth_gpu,
            lb_params=lb,
            calibration=frame.calibration,
            stream=po.stream,
            ts_s=frame.ts_ns * 1e-9,
        )
        self.tracer.mark_end("post", po.stream)
        po.record_done()
        po.stream.synchronize()  # only sync point in the hot path

        # --- stage D: optional constraint gate + occlusion fallback
        final_result = self._apply_constraints_and_fallback(
            result, ts_s=frame.ts_ns * 1e-9
        )
        t_end = time.perf_counter()

        # Emit trace AFTER synchronize so elapsed_time is safe to read.
        self.tracer.set_result_meta(
            valid=final_result.valid,
            occluded_count=int((final_result.kpt_conf < self.post.kpt_conf_threshold).sum().item())
            if final_result.valid else 0,
            depth_invalid_ratio=final_result.depth_invalid_ratio,
            box_conf=final_result.box_conf,
        )
        trace = self.tracer.end()

        tick = PipelineTick(
            frame_id=frame.frame_id,
            ts_ns=frame.ts_ns,
            result=final_result,
            latency_ms={
                "e2e": (t_end - t_start) * 1e3,
                **{f"{k}_ms": v for k, v in trace.stage_ms.items()},
            },
        )
        return tick

    def _apply_constraints_and_fallback(
        self, result: PoseResult, ts_s: float
    ) -> PoseResult:
        """Run opt-in constraints; fallback to last accepted if invalid.

        This keeps bad frames (occluded / teleported joints) from
        reaching the SHM with ``valid=True``. Constraint failures are
        converted into ``valid=False`` publishes so the control loop
        retreats to 0 N on the AK60.
        """
        if not result.valid:
            return result

        # calibration observation
        self.constraints.observe(result.kpts_3d_m)
        new_kpts, decision = self.constraints.apply(result.kpts_3d_m, ts_s=ts_s)

        if not decision.accept:
            # Hard reject — mark invalid AND zero the keypoint arrays so a
            # downstream consumer that erroneously ignores `valid` can't
            # drive AK60 using the last bad sample. Never overwrite the
            # constraint's internal prev-state.
            zeros_3d = torch.zeros_like(result.kpts_3d_m)
            zeros_2d = torch.zeros_like(result.kpts_2d_px)
            zeros_c = torch.zeros_like(result.kpt_conf)
            return PoseResult(
                kpts_2d_px=zeros_2d,
                kpts_3d_m=zeros_3d,
                kpt_conf=zeros_c,
                box_conf=result.box_conf,
                valid=False,
                depth_invalid_ratio=result.depth_invalid_ratio,
            )

        return PoseResult(
            kpts_2d_px=result.kpts_2d_px,
            kpts_3d_m=new_kpts,
            kpt_conf=result.kpt_conf,
            box_conf=result.box_conf,
            valid=True,
            depth_invalid_ratio=result.depth_invalid_ratio,
        )

    def shutdown(self) -> None:
        self.sm.synchronize_all()
