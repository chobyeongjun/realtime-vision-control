#!/bin/bash
# research-vault (~/research-vault) → realtime-vision-control/docs/ 동기화
# Vault가 single source of truth. docs/는 GitHub 공식 스냅샷.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAULT="$HOME/research-vault"

if [ ! -d "$VAULT" ]; then
    echo "ERROR: Vault not found at $VAULT"
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p docs/evolution docs/handovers docs/experiments

echo "[1/3] Wiki master document..."
[ -f "$VAULT/10_Wiki/perception-evolution-master.md" ] && \
    rsync -a "$VAULT/10_Wiki/perception-evolution-master.md" "docs/evolution/perception-evolution.md"

echo "[2/3] Handovers..."
for f in "$VAULT/00_Raw/"2026-*-perception-*.md \
         "$VAULT/00_Raw/"2026-*-cuda-stream-*.md \
         "$VAULT/00_Raw/"2026-*-p0-*.md; do
    [ -f "$f" ] && rsync -a "$f" docs/handovers/ 2>/dev/null || true
done

echo "[3/3] Experiments..."
[ -d "$VAULT/experiments/realtime-vision-control" ] && \
    rsync -av --delete "$VAULT/experiments/realtime-vision-control/" docs/experiments/

echo ""
echo "✓ Sync complete. Review with: git diff docs/"
