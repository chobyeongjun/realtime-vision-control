"""
Jetson Orin 시스템 최적화 유틸리티
==================================
최대 성능을 위한 시스템 레벨 설정:
  - nvpmodel: 전력 모드 (MAXN = 최대 성능)
  - jetson_clocks: CPU/GPU/EMC 클럭 최대 고정
  - fan: 쿨링 팬 최대 속도
  - CPU governor: performance 모드

사용법:
    from jetson_optimizer import optimize_jetson, restore_jetson
    optimize_jetson()   # 벤치마크 전
    restore_jetson()    # 벤치마크 후
"""

import subprocess
import os
import sys


def _run(cmd, check=False):
    """명령 실행 (sudo 필요 시 자동 추가)"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)


def get_current_power_mode():
    """현재 전력 모드 확인"""
    ok, out, _ = _run("nvpmodel -q")
    if ok:
        for line in out.splitlines():
            if "NV Power Mode" in line or "POWER_MODEL" in line:
                return line.strip()
        return out.splitlines()[0] if out else "unknown"
    return "unknown"


def get_gpu_freq():
    """현재 GPU 주파수"""
    paths = [
        "/sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq",
        "/sys/devices/platform/gpu/devfreq/17000000.gpu/cur_freq",
        "/sys/devices/17000000.gpu/devfreq/17000000.gpu/cur_freq",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    freq = int(f.read().strip())
                    return freq // 1000000  # Hz → MHz
            except (ValueError, PermissionError):
                pass
    return None


def set_power_mode_maxn():
    """MAXN 모드 설정 (모든 코어 최대 성능)"""
    ok, out, err = _run("sudo nvpmodel -m 0")
    if ok:
        print("  [POWER] nvpmodel → MAXN (최대 성능)")
        return True
    else:
        # Orin NX에서는 MAXN이 다른 번호일 수 있음
        ok2, out2, _ = _run("sudo nvpmodel -m 2")
        if ok2:
            print("  [POWER] nvpmodel → 모드 2 (고성능)")
            return True
        print(f"  [POWER] nvpmodel 설정 실패: {err}")
        return False


def enable_jetson_clocks():
    """CPU/GPU/EMC 클럭을 최대로 고정"""
    ok, _, err = _run("sudo jetson_clocks")
    if ok:
        print("  [CLOCK] jetson_clocks 활성화 (CPU/GPU/EMC 최대 클럭)")
        return True
    print(f"  [CLOCK] jetson_clocks 실패: {err}")
    return False


def save_jetson_clocks():
    """현재 클럭 설정 저장 (복원용)"""
    ok, _, _ = _run("sudo jetson_clocks --store /tmp/jetson_clocks_backup")
    if ok:
        print("  [CLOCK] 현재 클럭 설정 저장됨 → /tmp/jetson_clocks_backup")
    return ok


def restore_jetson_clocks():
    """저장된 클럭 설정 복원"""
    if os.path.exists("/tmp/jetson_clocks_backup"):
        ok, _, _ = _run("sudo jetson_clocks --restore /tmp/jetson_clocks_backup")
        if ok:
            print("  [CLOCK] 클럭 설정 복원됨")
            return True
    return False


def set_fan_max():
    """쿨링 팬 최대 속도"""
    fan_paths = [
        "/sys/devices/pwm-fan/target_pwm",
        "/sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1",
    ]

    import glob
    for pattern in fan_paths:
        for path in glob.glob(pattern):
            try:
                ok, _, _ = _run(f"sudo sh -c 'echo 255 > {path}'")
                if ok:
                    print(f"  [FAN] 팬 최대 속도 설정 ({path})")
                    return True
            except Exception:
                pass

    # Jetson Orin용 대체 방법
    ok, _, _ = _run("sudo sh -c 'echo 1 > /sys/kernel/debug/tegra_fan/fan_cap_pwm'")
    if not ok:
        print("  [FAN] 팬 제어 경로를 찾지 못함 (수동 확인 필요)")
    return ok


def set_cpu_governor_performance():
    """모든 CPU 코어를 performance governor로 설정"""
    cpu_count = os.cpu_count() or 8
    set_count = 0
    for i in range(cpu_count):
        path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"
        if os.path.exists(path):
            ok, _, _ = _run(f"sudo sh -c 'echo performance > {path}'")
            if ok:
                set_count += 1

    if set_count > 0:
        print(f"  [CPU] {set_count}개 코어 → performance governor")
        return True
    print("  [CPU] governor 설정 실패")
    return False


def disable_gpu_power_management():
    """GPU 전력 관리 비활성화 (일정한 성능 유지)"""
    paths = [
        "/sys/devices/gpu.0/power/control",
        "/sys/devices/17000000.gpu/power/control",
    ]
    for path in paths:
        if os.path.exists(path):
            ok, _, _ = _run(f"sudo sh -c 'echo on > {path}'")
            if ok:
                print(f"  [GPU] 전력 관리 비활성화 ({path})")
                return True
    return False


def _find_ina3221_hwmon():
    """INA3221 센서의 hwmon 디렉토리 찾기"""
    import glob

    # 방법 1: /sys/class/hwmon에서 name=ina3221 찾기 (가장 확실)
    for hwmon_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name_path = os.path.join(hwmon_dir, "name")
        if os.path.exists(name_path):
            ok, name, _ = _run(f"cat {name_path}")
            if ok and "ina3221" in name.lower():
                return hwmon_dir

    # 방법 2: curr1_max가 있는 hwmon 찾기
    for hwmon_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if os.path.exists(os.path.join(hwmon_dir, "curr1_max")):
            return hwmon_dir

    return None


def raise_current_limits():
    """VDD_IN 전류 제한을 높여서 쓰로틀링 방지

    Jetson Orin NX 25W 모드에서 기본 VDD_IN warn=1240mA, crit=1488mA로
    GPU 풀로드 시 전류 스파이크가 이 한계를 초과하면 쓰로틀링 발생.
    warn=2500mA, crit=3000mA로 올려서 headroom 확보.
    (20V × 2.5A = 50W로 하드웨어 한도 내 안전)
    """
    hwmon_dir = _find_ina3221_hwmon()
    if not hwmon_dir:
        print("  [CURRENT] INA3221 센서를 찾지 못함 — "
              "수동으로 확인 필요:")
        print("    sudo find /sys -name 'curr1_max' 2>/dev/null")
        return False

    warn_path = os.path.join(hwmon_dir, "curr1_max")
    crit_path = os.path.join(hwmon_dir, "curr1_crit")

    # 현재 값 출력
    ok_r, cur_warn, _ = _run(f"cat {warn_path}")
    ok_r2, cur_crit, _ = _run(f"cat {crit_path}")
    if ok_r:
        print(f"  [CURRENT] 현재 VDD_IN warn={cur_warn}mA, crit={cur_crit}mA")

    # VDD_IN (ch1): warn → 2500mA, crit → 3000mA
    ok1, _, _ = _run(f"sudo sh -c 'echo 2500 > {warn_path}'")
    ok2, _, _ = _run(f"sudo sh -c 'echo 3000 > {crit_path}'")

    if ok1 and ok2:
        print(f"  [CURRENT] VDD_IN 전류 제한 상향: warn=2500mA, crit=3000mA")

        # VDD_CPU_GPU_CV (ch2), VDD_SOC (ch3)도 충분히 올려줌
        for ch in [2, 3]:
            w = os.path.join(hwmon_dir, f"curr{ch}_max")
            c = os.path.join(hwmon_dir, f"curr{ch}_crit")
            if os.path.exists(w):
                _run(f"sudo sh -c 'echo 32760 > {w}'")
                _run(f"sudo sh -c 'echo 32760 > {c}'")

        return True

    print(f"  [CURRENT] 전류 제한 설정 실패 (권한 문제일 수 있음)")
    return False

    return found


def drop_caches():
    """메모리 캐시 정리"""
    ok, _, _ = _run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    if ok:
        print("  [MEM] 메모리 캐시 정리 완료")
    return ok


def get_system_info():
    """Jetson 시스템 정보 수집"""
    info = {}

    # 모델명
    ok, out, _ = _run("cat /proc/device-tree/model")
    if ok:
        info["model"] = out.replace('\x00', '')

    # CUDA 버전
    ok, out, _ = _run("nvcc --version")
    if ok:
        for line in out.splitlines():
            if "release" in line:
                info["cuda"] = line.strip()

    # JetPack 버전
    ok, out, _ = _run("cat /etc/nv_tegra_release")
    if ok:
        info["jetpack"] = out.splitlines()[0] if out else ""

    # 전력 모드
    info["power_mode"] = get_current_power_mode()

    # GPU 주파수
    gpu_freq = get_gpu_freq()
    if gpu_freq:
        info["gpu_freq_mhz"] = gpu_freq

    # 메모리
    ok, out, _ = _run("free -m")
    if ok:
        for line in out.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                info["ram_total_mb"] = int(parts[1])
                info["ram_available_mb"] = int(parts[6])

    # 온도
    ok, out, _ = _run("cat /sys/devices/virtual/thermal/thermal_zone*/temp")
    if ok:
        temps = []
        for line in out.splitlines():
            try:
                temps.append(int(line) / 1000.0)
            except ValueError:
                pass
        if temps:
            info["temp_max_c"] = max(temps)

    # 전류 제한 확인
    hwmon_dir = _find_ina3221_hwmon()
    if hwmon_dir:
        ok_w, warn_val, _ = _run(f"cat {os.path.join(hwmon_dir, 'curr1_max')}")
        ok_c, crit_val, _ = _run(f"cat {os.path.join(hwmon_dir, 'curr1_crit')}")
        if ok_w:
            try:
                info["vdd_in_warn_ma"] = int(warn_val)
                info["vdd_in_crit_ma"] = int(crit_val) if ok_c else None
            except ValueError:
                pass

    return info


def optimize_jetson(verbose=True):
    """Jetson 시스템을 최대 성능으로 최적화

    Returns:
        dict: 적용된 최적화 목록
    """
    if verbose:
        print("━" * 60)
        print("  Jetson 시스템 최적화 (최대 성능)")
        print("━" * 60)

    results = {}

    # 1. 현재 클럭 저장 (복원용)
    results["clocks_saved"] = save_jetson_clocks()

    # 2. MAXN 전력 모드
    results["power_mode"] = set_power_mode_maxn()

    # 3. jetson_clocks (CPU/GPU/EMC 최대)
    results["jetson_clocks"] = enable_jetson_clocks()

    # 4. CPU governor → performance
    results["cpu_governor"] = set_cpu_governor_performance()

    # 5. 팬 최대 속도
    results["fan_max"] = set_fan_max()

    # 6. GPU 전력관리 비활성화
    results["gpu_power"] = disable_gpu_power_management()

    # 7. VDD_IN 전류 제한 상향 (쓰로틀링 방지)
    results["current_limits"] = raise_current_limits()

    # 8. 메모리 캐시 정리
    results["drop_caches"] = drop_caches()

    if verbose:
        # 결과 출력
        info = get_system_info()
        print()
        print(f"  시스템: {info.get('model', 'unknown')}")
        print(f"  전력 모드: {info.get('power_mode', 'unknown')}")
        gpu_freq = info.get('gpu_freq_mhz')
        if gpu_freq:
            print(f"  GPU 주파수: {gpu_freq} MHz")
        temp = info.get('temp_max_c')
        if temp:
            print(f"  최대 온도: {temp:.1f}°C")
        ram = info.get('ram_available_mb')
        if ram:
            print(f"  가용 RAM: {ram} MB")
        print("━" * 60)

    return results


def restore_jetson(verbose=True):
    """Jetson 시스템을 원래 상태로 복원"""
    if verbose:
        print("\n  [복원] Jetson 시스템 설정 복원 중...")

    restore_jetson_clocks()

    # CPU governor → schedutil (기본값)
    cpu_count = os.cpu_count() or 8
    for i in range(cpu_count):
        path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"
        if os.path.exists(path):
            _run(f"sudo sh -c 'echo schedutil > {path}'")

    if verbose:
        print("  [복원] 완료")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Jetson 시스템 최적화")
    parser.add_argument("--restore", action="store_true", help="원래 설정으로 복원")
    parser.add_argument("--info", action="store_true", help="시스템 정보만 출력")
    args = parser.parse_args()

    if args.info:
        info = get_system_info()
        for k, v in info.items():
            print(f"  {k}: {v}")
    elif args.restore:
        restore_jetson()
    else:
        optimize_jetson()
