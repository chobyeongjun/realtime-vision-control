# Architecture — CUDA_Stream pipeline

## Stream topology
```
                    torch.cuda.Event
          ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
ZED grab →│ capture_str │  →   │ preproc_str │  →   │ infer_str   │──┐
          │ (bg thread) │      │ letterbox   │      │ execute_v3  │  │
          └─────────────┘      │ +normalize  │      └─────────────┘  │
                               └─────────────┘                        │
                                                                      ▼
                                                           ┌─────────────┐
                                                           │ post_str    │
                                                           │ 3D+filter   │
                                                           │ → pinned    │
                                                           └─────┬───────┘
                                                                 │
                                                                 ▼
                                                           SHM /hwalker_pose_cuda
```

## Frame N pipeline
| t | capture | preproc | infer | post |
|---|--------:|--------:|------:|-----:|
| 0 | N+1     |   N     |  N-1  |  N-2 |
| 1 | N+2     |  N+1    |   N   |  N-1 |

At steady state, host only waits on post_str.synchronize() (≈ 1-2ms).

## Event contract
- `capture.done_event.record(capture_stream)` — fired when ZED frame copied to GPU
- `preproc.wait_event(capture.done)` — preproc waits for capture
- `infer.wait_event(preproc.done)` — infer waits for preproc
- `post.wait_event(infer.done)` — post waits for infer

No `torch.cuda.synchronize()` in hot path. Only `post_stream.synchronize()`
once per published tick (cost ≈ 0.5ms to retire).

## Memory model
- ZED Mat host-side copy (deep_copy=True) → torch.from_numpy → H2D on capture_stream
- TRT input bound directly to `GpuPreprocessor.out` (no intermediate copy)
- TRT output stays on GPU, consumed by `GpuPostprocessor`
- Only the final 17×3 keypoint array (~200 B) is D2H'd to pinned host

## Fallback hierarchy
1. **sdk_cuda_ctx share** (ZED SDK 5.2+) → truly zero-copy
2. **copy_async** (default) → ZED CPU → torch H2D via capture_stream — still overlaps with other streams
3. **Webcam stub** → only for dev machines without ZED

## CUDA Graph
When shapes are fixed (imgsz=640, batch=1), `cuda_graph.GraphedStep` captures
the preproc+infer sub-graph and replays it with one launch. Failure is
silent — falls back to eager.

## Invariants
- Mainline (`realtime/`, `benchmarks/`) is READ-ONLY to this module
- SHM name is `/hwalker_pose_cuda` and never `/hwalker_pose`
- `stream_manager.py` is the ONLY owner of `torch.cuda.Stream` instances
- No module calls `torch.cuda.synchronize()` globally in hot paths
