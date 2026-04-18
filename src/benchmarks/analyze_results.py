#!/usr/bin/env python3
"""
벤치마크 결과 분석 및 시각화
============================
benchmark JSON 파일을 읽어 모델 비교 차트 + HTML 리포트를 생성합니다.

사용법:
    python3 analyze_results.py results/                          # 디렉토리 내 모든 JSON
    python3 analyze_results.py results/benchmark_*.json          # glob 패턴
    python3 analyze_results.py results/ --output report.html     # HTML 리포트 생성
"""

import argparse
import json
import os
import sys
import glob
import base64
from io import BytesIO
from datetime import datetime

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib 미설치 - 차트 생성 불가. pip install matplotlib")


# ============================================================================
# 데이터 로드
# ============================================================================
def load_results(paths):
    """벤치마크 JSON 파일 로드"""
    all_results = []
    for path in paths:
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "benchmark_*.json")))
        else:
            files = sorted(glob.glob(path))

        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    data["_source_file"] = os.path.basename(f)
                    all_results.append(data)
            except Exception as e:
                print(f"  [WARN] {f} 로드 실패: {e}")

    return all_results


def merge_results(all_data):
    """여러 파일의 결과를 모델별로 병합 (최신 우선)"""
    merged = {}
    configs = []
    for data in all_data:
        configs.append(data.get("config", {}))
        for model_name, stats in data.get("results", {}).items():
            if "error" not in stats:
                merged[model_name] = stats
    return merged, configs


# ============================================================================
# 모델 비교표 (터미널 출력용)
# ============================================================================
def print_model_comparison_table(merged):
    """모델별 상세 비교표 출력"""
    if not merged:
        print("  결과 없음")
        return

    print()
    print("=" * 140)
    print("  모델 비교표 (Model Comparison Table)")
    print("=" * 140)

    # 헤더 (스코어 제거, raw 메트릭만)
    headers = [
        ("모델", 28),
        ("FPS", 6),
        ("Infer", 7),
        ("E2E", 7),
        ("P95", 7),
        ("<50ms%", 7),
        ("인식률", 6),
        ("하체%", 6),
        ("Conf", 5),
        ("Foot", 4),
    ]
    header_line = "  ".join(f"{h:<{w}}" if i == 0 else f"{h:>{w}}" for i, (h, w) in enumerate(headers))
    print(header_line)
    print("-" * 110)

    for name, stats in merged.items():
        fps = stats.get('avg_fps', 0)
        infer = stats.get('avg_inference_time_ms', stats.get('avg_latency_ms', 0))
        e2e = stats.get('avg_e2e_latency_ms', 0)
        p95 = stats.get('p95_e2e_latency_ms', 0)
        under50 = stats.get('e2e_under_50ms_rate', 0)
        det = stats.get('detection_rate', 0)
        ll = stats.get('lower_limb_rate', 0)
        conf = stats.get('avg_lower_limb_conf', 0)
        has_foot = "Y" if stats.get("joint_angle_stats", {}).get("left_ankle_dorsiflexion") else "N"

        print(f"  {name:<28}"
              f"{fps:>6.1f}  "
              f"{infer:>7.1f}  "
              f"{e2e:>7.1f}  "
              f"{p95:>7.1f}  "
              f"{under50:>7.1f}  "
              f"{det:>6.1f}  "
              f"{ll:>6.1f}  "
              f"{conf:>5.2f}  "
              f"{has_foot:>4}")

    print("-" * 110)

    # 요약 (스코어 없이)
    valid = {n: s for n, s in merged.items()}
    if valid:
        best_fps = max(valid, key=lambda n: valid[n].get('avg_fps', 0))
        best_e2e = min(valid, key=lambda n: valid[n].get('avg_e2e_latency_ms', float('inf')))
        best_det = max(valid, key=lambda n: valid[n].get('lower_limb_rate', 0))
        print(f"\n  최고 FPS:       {best_fps} ({valid[best_fps].get('avg_fps', 0):.1f})")
        print(f"  최저 E2E:       {best_e2e} ({valid[best_e2e].get('avg_e2e_latency_ms', 0):.1f}ms)")
        print(f"  최고 하체 인식:  {best_det} ({valid[best_det].get('lower_limb_rate', 0):.1f}%)")

    # 관절 각도 비교표
    print_joint_angle_table(merged)


def print_joint_angle_table(merged):
    """관절 각도 비교표"""
    angle_names = set()
    for stats in merged.values():
        angle_names.update(stats.get("joint_angle_stats", {}).keys())

    if not angle_names:
        return

    angle_names = sorted(angle_names)
    print()
    print("=" * 120)
    print("  관절 각도 비교 (degrees)")
    print("=" * 120)
    print(f"  {'모델':<28}", end="")
    for an in angle_names:
        short = an.replace("_flexion", "").replace("_dorsiflexion", "_df")
        print(f"  {short:>14}", end="")
    print()
    print("-" * 120)

    for name, stats in merged.items():
        ja = stats.get("joint_angle_stats", {})
        print(f"  {name:<28}", end="")
        for an in angle_names:
            if an in ja:
                m = ja[an]["mean"]
                s = ja[an]["std"]
                print(f"  {m:>6.1f}+{s:<5.1f}", end="")
            else:
                print(f"  {'N/A':>14}", end="")
        print()

    print("-" * 120)


# ============================================================================
# 추천 스코어 계산
# ============================================================================
def compute_recommendation_scores(merged):
    """모델 추천 스코어 계산"""
    scores = {}
    for name, stats in merged.items():
        e2e = stats.get('avg_e2e_latency_ms', 100)
        ll_rate = stats.get('lower_limb_rate', 0)
        has_foot = 1.0 if stats.get("joint_angle_stats", {}).get("left_ankle_dorsiflexion") else 0.0
        fps = stats.get('avg_fps', 0)

        latency_score = max(0, (50 - e2e) / 50) if e2e < 50 else -0.3
        detect_score = ll_rate / 100.0
        foot_score = has_foot
        fps_score = min(fps / 120.0, 1.0)

        bone_cvs = list(stats.get("bone_length_cv", {}).values())
        stability_score = max(0, 1.0 - np.mean(bone_cvs) * 10) if bone_cvs else 0.5

        total = (latency_score * 0.30 +
                 detect_score * 0.25 +
                 stability_score * 0.20 +
                 foot_score * 0.15 +
                 fps_score * 0.10)
        scores[name] = total

    return scores


# ============================================================================
# 차트 생성
# ============================================================================
def fig_to_base64(fig):
    """matplotlib figure를 base64 PNG로 변환"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def generate_latency_breakdown_chart(merged):
    """Latency 분해 stacked bar chart"""
    if not HAS_MPL:
        return None

    names = list(merged.keys())
    grab = [merged[n].get('avg_grab_time_ms', 0) for n in names]
    infer = [merged[n].get('avg_inference_time_ms', merged[n].get('avg_latency_ms', 0)) for n in names]
    post = [merged[n].get('avg_postprocess_time_ms', 0) for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    w = 0.6

    bars1 = ax.bar(x, grab, w, label='Grab', color='#4CAF50')
    bars2 = ax.bar(x, infer, w, bottom=grab, label='Inference', color='#2196F3')
    bottom2 = [g + i for g, i in zip(grab, infer)]
    bars3 = ax.bar(x, post, w, bottom=bottom2, label='Post-process', color='#FF9800')

    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Target: 50ms')
    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('E2E Latency Breakdown (Grab + Inference + Post-processing)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    return fig_to_base64(fig)


def generate_fps_chart(merged):
    """FPS 비교 bar chart"""
    if not HAS_MPL:
        return None

    names = list(merged.keys())
    fps = [merged[n].get('avg_fps', 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4CAF50' if f >= 30 else '#FF5722' for f in fps]
    ax.bar(names, fps, color=colors)
    ax.axhline(y=30, color='orange', linestyle='--', label='Min: 30 Hz')
    ax.axhline(y=60, color='blue', linestyle='--', label='Camera: 60 Hz')
    ax.set_ylabel('FPS')
    ax.set_title('Average FPS per Model')
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    return fig_to_base64(fig)


def generate_detection_chart(merged):
    """인식률 비교 grouped bar chart"""
    if not HAS_MPL:
        return None

    names = list(merged.keys())
    det_rate = [merged[n].get('detection_rate', 0) for n in names]
    ll_rate = [merged[n].get('lower_limb_rate', 0) for n in names]
    conf = [merged[n].get('avg_lower_limb_conf', 0) * 100 for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    w = 0.25

    ax.bar(x - w, det_rate, w, label='Person Detection %', color='#2196F3')
    ax.bar(x, ll_rate, w, label='Lower Limb Detection %', color='#4CAF50')
    ax.bar(x + w, conf, w, label='Lower Limb Confidence %', color='#FF9800')

    ax.set_ylabel('Rate (%)')
    ax.set_title('Detection Rate & Lower Limb Confidence')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    return fig_to_base64(fig)


def generate_score_chart(merged):
    """종합 추천 스코어 chart"""
    if not HAS_MPL:
        return None

    scores = compute_recommendation_scores(merged)
    names = sorted(scores, key=scores.get, reverse=True)
    vals = [scores[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4CAF50' if i == 0 else '#90CAF9' for i in range(len(names))]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel('Score')
    ax.set_title('Model Recommendation Score (higher = better)')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    return fig_to_base64(fig)


def generate_latency_distribution_chart(merged):
    """E2E latency 분포 히스토그램"""
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, stats in merged.items():
        e2e_list = stats.get('e2e_latency_ms_list', [])
        if e2e_list:
            ax.hist(e2e_list, bins=30, alpha=0.5, label=name)

    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Target: 50ms')
    ax.set_xlabel('E2E Latency (ms)')
    ax.set_ylabel('Count')
    ax.set_title('E2E Latency Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    return fig_to_base64(fig)


# ============================================================================
# HTML 리포트 생성
# ============================================================================
def generate_html_report(merged, configs, output_path):
    """HTML 리포트 생성"""
    charts = {}
    if HAS_MPL:
        charts['latency'] = generate_latency_breakdown_chart(merged)
        charts['fps'] = generate_fps_chart(merged)
        charts['detection'] = generate_detection_chart(merged)
        charts['distribution'] = generate_latency_distribution_chart(merged)

    # 최저 E2E 기준으로 best 모델 선정 (스코어 없이)
    valid_models = {n: s for n, s in merged.items() if s.get('avg_e2e_latency_ms', 999) < 999}
    best_model = min(valid_models, key=lambda n: valid_models[n].get('avg_e2e_latency_ms', 999)) if valid_models else "N/A"

    # 모델 비교 테이블 HTML (스코어 제거)
    table_rows = []
    for name, stats in merged.items():
        fps = stats.get('avg_fps', 0)
        infer = stats.get('avg_inference_time_ms', stats.get('avg_latency_ms', 0))
        e2e = stats.get('avg_e2e_latency_ms', 0)
        p95 = stats.get('p95_e2e_latency_ms', 0)
        under50 = stats.get('e2e_under_50ms_rate', 0)
        det = stats.get('detection_rate', 0)
        ll = stats.get('lower_limb_rate', 0)
        conf = stats.get('avg_lower_limb_conf', 0)
        has_foot = stats.get("joint_angle_stats", {}).get("left_ankle_dorsiflexion")

        is_best = ' style="background:#e8f5e9;font-weight:bold"' if name == best_model else ''
        e2e_color = '#4CAF50' if e2e < 50 else '#FF5722'

        table_rows.append(f"""
        <tr{is_best}>
            <td>{name}</td>
            <td>{fps:.1f}</td>
            <td>{infer:.1f}</td>
            <td style="color:{e2e_color}">{e2e:.1f}</td>
            <td>{p95:.1f}</td>
            <td>{under50:.1f}%</td>
            <td>{det:.1f}%</td>
            <td>{ll:.1f}%</td>
            <td>{conf:.3f}</td>
            <td>{'Yes' if has_foot else 'No'}</td>
        </tr>""")

    chart_imgs = ""
    for title, key in [("E2E Latency Breakdown", "latency"),
                        ("FPS Comparison", "fps"),
                        ("Detection Rate", "detection"),
                        ("Latency Distribution", "distribution")]:
        if charts.get(key):
            chart_imgs += f"""
            <div class="chart">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{charts[key]}" alt="{title}">
            </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>H-Walker Pose Estimation Benchmark Report</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #fafafa; }}
    h1 {{ color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 10px; }}
    h2 {{ color: #2196F3; margin-top: 30px; }}
    h3 {{ color: #424242; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 14px; }}
    th {{ background: #1565C0; color: white; padding: 10px 8px; text-align: center; }}
    td {{ padding: 8px; border: 1px solid #ddd; text-align: center; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    .chart {{ margin: 20px 0; text-align: center; }}
    .chart img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .recommendation {{ background: #e8f5e9; border-left: 4px solid #4CAF50; padding: 15px; margin: 20px 0; border-radius: 4px; }}
    .config {{ background: #e3f2fd; padding: 10px 15px; border-radius: 4px; margin: 10px 0; font-size: 13px; }}
    .footer {{ color: #999; font-size: 12px; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 10px; }}
</style>
</head>
<body>
<h1>H-Walker Pose Estimation Benchmark Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="config">
    <strong>Config:</strong>
    {' | '.join(f"{k}: {v}" for c in configs for k, v in c.items())}
</div>

<div class="recommendation">
    <strong>Lowest E2E Latency: {best_model}</strong>
    <br>Raw 메트릭(FPS, Infer, E2E, P95, Detection)으로 직접 비교하세요. 스코어링은 사용하지 않습니다.
</div>

<h2>Model Comparison Table</h2>
<table>
<tr>
    <th>Model</th><th>FPS</th><th>Infer(ms)</th><th>E2E(ms)</th><th>P95(ms)</th>
    <th>&lt;50ms</th><th>Detection</th><th>Lower Limb</th><th>Confidence</th>
    <th>Foot KP</th>
</tr>
{''.join(table_rows)}
</table>

{chart_imgs}

<div class="footer">
    H-Walker Pose Estimation Benchmark System | ZED X Mini + Jetson Orin
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML 리포트 생성: {output_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="벤치마크 결과 분석")
    parser.add_argument("paths", nargs="+", help="JSON 파일 또는 디렉토리 경로")
    parser.add_argument("--output", type=str, default=None,
                        help="HTML 리포트 출력 경로 (기본: results/analysis/report.html)")
    args = parser.parse_args()

    # 데이터 로드
    all_data = load_results(args.paths)
    if not all_data:
        print("결과 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"  {len(all_data)}개 결과 파일 로드됨")
    merged, configs = merge_results(all_data)
    print(f"  {len(merged)}개 모델 결과 병합됨")

    # 터미널 비교표 출력
    print_model_comparison_table(merged)

    # HTML 리포트 생성
    output = args.output or os.path.join(
        os.path.dirname(__file__), "results", "analysis", "report.html")
    generate_html_report(merged, configs, output)


if __name__ == "__main__":
    main()
