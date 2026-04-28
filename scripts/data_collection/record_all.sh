#!/usr/bin/env bash
# 2026-04-28 데이터 수집 — 복장 3종 × 속도 3종 + Exosuit 3종
# Jetson에서 실행. 각 세션 사이에 Enter 눌러서 진행.

set -e

DATE="2026-04-28"
OUTDIR="recordings/${DATE}"
DURATION=60   # 초 (필요시 수정)

mkdir -p "$OUTDIR"

run() {
    local name="$1"
    echo ""
    echo "============================================"
    echo "  다음: ${name}.svo2"
    echo "  준비되면 Enter"
    echo "============================================"
    read -r
    python3 -m perception.CUDA_Stream.record_svo \
        --out "${OUTDIR}/${name}.svo2" \
        --duration "$DURATION"
}

# ── 추리닝 ────────────────────────────────────────
run "walk_sweats_3kmh"
run "walk_sweats_4p5kmh"
run "walk_sweats_6kmh"

# ── 쫄쫄이 ────────────────────────────────────────
run "walk_tight_3kmh"
run "walk_tight_4p5kmh"
run "walk_tight_6kmh"

# ── 헐렁한 반바지 ──────────────────────────────────
run "walk_shorts_3kmh"
run "walk_shorts_4p5kmh"
run "walk_shorts_6kmh"

# ── Exosuit ───────────────────────────────────────
run "walk_exosuit_3kmh"
run "walk_exosuit_4p5kmh"
run "walk_exosuit_6kmh"

echo ""
echo "모든 녹화 완료 → ${OUTDIR}/"
ls -lh "${OUTDIR}/"
