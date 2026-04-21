"""
kf_smoother.py
==============
보행 재활 로봇(H-Walker)용 Linear Kalman Filter smoother.

목적:
  1. 카메라 latency(~21ms) 보상 → 현재 시각으로 joint angle 추정
  2. q̇ (각속도) 추정 → 나중에 Kd 항에 사용
  3. 카메라 노이즈(~2~3°) 스무딩

State space (8-state):
  x = [q_knee_L, q_knee_R, q_hip_L, q_hip_R,       # deg    [0:4]
       q̇_knee_L, q̇_knee_R, q̇_hip_L, q̇_hip_R]    # deg/s  [4:8]

Process model (constant velocity):
  x[k+1] = A * x[k] + w[k]

  A = [[I4,  dt*I4],
       [0,   I4   ]]

  즉:  q[k+1] = q[k] + dt * q̇[k]
       q̇[k+1] = q̇[k]

Measurement model:
  z[k] = H * x[k] + v[k]
  H = [I4 | 0]   (position 만 관측)

Process noise Q — Discrete White Noise Acceleration (DWNA) 모델:
  연속 가속도 불확실성 σ_a [deg/s²]를 이산화.
  각 관절 독립적으로:

    Γ = [dt²/2, dt]ᵀ          (가속도 → state 영향 벡터)
    Q_joint = σ_a² * Γ * Γᵀ
            = σ_a² * [[dt⁴/4,   dt³/2 ],
                       [dt³/2,   dt²   ]]

  이를 4 관절 블록 대각으로 조합 → Q (8×8)

  σ_a 선택 기준: 보행 최대 각가속도.
    무릎 f=0.8Hz, amp=40° → max q̈ = amp*(2πf)² ≈ 1005 deg/s²
    기본값 σ_a=300 deg/s²는 이 변화를 따라갈 수 있는 충분한 값.

Measurement noise:
  R = σ_z² * I4,  σ_z ≈ 2~3°  (카메라 keypoint noise)

Kalman 업데이트 방정식:
  Predict:
    x̂⁻ = A * x̂
    P⁻  = A * P * Aᵀ + Q

  Update (measurement z 수신 시):
    K   = P⁻ * Hᵀ * (H * P⁻ * Hᵀ + R)⁻¹
    x̂  = x̂⁻ + K * (z − H * x̂⁻)
    P   = (I − K * H) * P⁻

Latency 보상:
  카메라 프레임은 실제로 latency_s 이전의 포즈를 나타낸다.
  update() 후 predict(latency_s) 로 현재 시각으로 extrapolate.
  구현: q̂_now ≈ q̂ + latency_s * q̂̇  (constant-velocity 근사)
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────
# GaitKalmanFilter
# ─────────────────────────────────────────────────────────────

class GaitKalmanFilter:
    """
    8-state Kalman filter for lower-limb joint angle estimation.

    State order:
        [0] q_knee_L   [deg]
        [1] q_knee_R   [deg]
        [2] q_hip_L    [deg]
        [3] q_hip_R    [deg]
        [4] q̇_knee_L  [deg/s]
        [5] q̇_knee_R  [deg/s]
        [6] q̇_hip_L   [deg/s]
        [7] q̇_hip_R   [deg/s]

    Parameters
    ----------
    dt : float
        Nominal time step [s].  Default 0.02 (50 Hz).
    angle_noise_deg : float
        Camera measurement noise 1-σ [deg].  Default 2.5 deg.
    accel_noise_deg_s2 : float
        Process noise — angular acceleration uncertainty 1-σ [deg/s²].
        DWNA(Discrete White Noise Acceleration) 모델의 σ_a.
        보행 무릎 최대 각가속도 ≈ 1000 deg/s² 수준.
        기본값 1000 은 응답성과 스무딩의 균형점.
        크게(>3000) → 측정값 거의 그대로 통과 / 작게(<100) → 과도한 lag 발생.
    """

    N_STATE = 8     # state dimension
    N_OBS   = 4     # observation dimension

    def __init__(
        self,
        dt: float = 0.02,
        angle_noise_deg: float = 2.5,
        accel_noise_deg_s2: float = 1000.0,
    ) -> None:
        self._dt_nominal: float = float(dt)
        self._angle_noise_deg  = float(angle_noise_deg)
        self._accel_noise      = float(accel_noise_deg_s2)

        n, m = self.N_STATE, self.N_OBS

        # ── 고정 행렬 pre-allocate ────────────────────────────────────
        # Transition matrix A (8×8)  — predict()에서 dt에 맞게 갱신
        self._A: np.ndarray = np.eye(n, dtype=np.float64)
        # 대각 오른쪽 상단 블록 (q → q̇ 결합) 은 predict()에서 채움

        # Measurement matrix H (4×8):  H = [I4 | 0]
        self._H: np.ndarray = np.zeros((m, n), dtype=np.float64)
        self._H[:m, :m] = np.eye(m)

        # Measurement noise covariance R (4×4)  [deg²]
        self._R: np.ndarray = np.eye(m, dtype=np.float64) * (angle_noise_deg ** 2)

        # Process noise Q — 초기엔 nominal dt 기준
        self._Q: np.ndarray = self._build_Q(dt, accel_noise_deg_s2)

        # 작업 버퍼 (루프 내 재사용, 동적 할당 없음)
        self._PHt:   np.ndarray = np.zeros((n, m), dtype=np.float64)
        self._S:     np.ndarray = np.zeros((m, m), dtype=np.float64)
        self._K:     np.ndarray = np.zeros((n, m), dtype=np.float64)
        self._IKH:   np.ndarray = np.zeros((n, n), dtype=np.float64)
        self._innov: np.ndarray = np.zeros(m,       dtype=np.float64)
        self._eye_n: np.ndarray = np.eye(n,         dtype=np.float64)

        # State & covariance
        self._x: np.ndarray = np.zeros(n, dtype=np.float64)
        self._P: np.ndarray = np.eye(n, dtype=np.float64) * 1e4

        self._initialized: bool = False

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────

    @staticmethod
    def _build_Q(dt: float, sigma_a: float) -> np.ndarray:
        """
        DWNA(Discrete White Noise Acceleration) 기반 process noise Q (8×8).

        각 관절 j 에 대해:
          Γ_j = [dt²/2, dt]ᵀ
          Q_j = σ_a² * Γ_j * Γ_jᵀ

        4 관절 블록 대각으로 조합.
        이 방식은 위치-속도 간 noise 상관관계를 물리적으로 정확하게 반영.

        Parameters
        ----------
        dt      : float  시간 간격 [s]
        sigma_a : float  가속도 불확실성 1-σ [deg/s²]

        Returns
        -------
        Q : np.ndarray shape (8, 8)
        """
        n = GaitKalmanFilter.N_STATE
        m = GaitKalmanFilter.N_OBS
        Q = np.zeros((n, n), dtype=np.float64)

        q11 = sigma_a ** 2 * (dt ** 4) / 4.0   # Γ[0]*Γ[0] = (dt²/2)²
        q12 = sigma_a ** 2 * (dt ** 3) / 2.0   # Γ[0]*Γ[1]
        q22 = sigma_a ** 2 * (dt ** 2)          # Γ[1]*Γ[1] = dt²

        for j in range(m):
            Q[j,   j]   = q11    # position-position block
            Q[j,   j+m] = q12    # position-velocity cross term
            Q[j+m, j]   = q12    # velocity-position cross term
            Q[j+m, j+m] = q22    # velocity-velocity block

        return Q

    def _update_A(self, dt: float) -> None:
        """A = [[I4, dt*I4], [0, I4]] 의 상단 오른쪽 블록만 갱신."""
        m = self.N_OBS
        for j in range(m):
            self._A[j, j + m] = dt

    # ── Public API ────────────────────────────────────────────────────

    def predict(self, dt: float | None = None) -> None:
        """
        Kalman predict step.

          x̂⁻  = A(dt) * x̂
          P⁻   = A(dt) * P * A(dt)ᵀ + Q(dt)

        dt가 nominal과 다르면 A와 Q를 그 자리에서 재계산.
        pre-allocate 버퍼를 사용하므로 루프 내 동적 할당 없음.

        Parameters
        ----------
        dt : float, optional
            실제 프레임 간격 [s].  None이면 nominal dt 사용.
        """
        if dt is None:
            dt = self._dt_nominal
            # 동일 dt이면 저장된 Q 재사용
            Q = self._Q
        else:
            # 가변 dt: Q를 즉석 계산 (small stack alloc, 루프 밖 설계 권장)
            Q = self._build_Q(dt, self._accel_noise)

        self._update_A(dt)

        # x̂⁻ = A * x̂  (in-place)
        tmp_x = self._A @ self._x
        self._x[:] = tmp_x

        # P⁻ = A * P * Aᵀ + Q
        self._P[:] = self._A @ self._P @ self._A.T + Q

    def update(self, q_measured: np.ndarray) -> None:
        """
        Kalman update step (measurement 수신 시 호출).

          z   = q_measured                    [deg, shape (4,)]
          ν   = z − H * x̂⁻                  (innovation)
          K   = P⁻ * Hᵀ * (H * P⁻ * Hᵀ + R)⁻¹
          x̂  = x̂⁻ + K * ν
          P   = (I − K*H) * P⁻

        H = [I4 | 0] 구조를 활용해 불필요한 0 곱셈을 제거.

        Parameters
        ----------
        q_measured : np.ndarray, shape (4,)
            [knee_L, knee_R, hip_L, hip_R] 측정값 [deg].
        """
        q_meas = np.asarray(q_measured, dtype=np.float64).ravel()
        m = self.N_OBS

        if not self._initialized:
            # 첫 측정값으로 position state 초기화, velocity = 0
            self._x[:m] = q_meas
            self._x[m:] = 0.0
            # 초기 공분산: position은 측정 노이즈 수준으로, velocity는 크게
            diag = np.concatenate([
                np.full(m, self._angle_noise_deg ** 2),
                np.full(m, 1e4),
            ])
            np.fill_diagonal(self._P, diag)
            self._initialized = True
            return

        # Innovation: ν = z − x̂⁻[:4]  (H=[I4|0] 이므로 H*x̂ = x̂[:4])
        np.subtract(q_meas, self._x[:m], out=self._innov)

        # S = H * P⁻ * Hᵀ + R  = P⁻[:4, :4] + R
        np.add(self._P[:m, :m], self._R, out=self._S)

        # K = P⁻ * Hᵀ * S⁻¹  = P⁻[:, :4] * S⁻¹
        # (H=[I4|0] 이므로 P⁻ * Hᵀ = P⁻의 앞 4열)
        self._PHt[:] = self._P[:, :m]
        S_inv = np.linalg.inv(self._S)
        np.dot(self._PHt, S_inv, out=self._K)

        # x̂ = x̂⁻ + K * ν
        self._x += self._K @ self._innov

        # P = (I − K*H) * P⁻
        np.dot(self._K, self._H, out=self._IKH)
        self._IKH[:] = self._eye_n - self._IKH
        self._P[:] = self._IKH @ self._P

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        현재 추정 state 반환.

        Returns
        -------
        q_hat  : np.ndarray, shape (4,)  [deg]
            [knee_L, knee_R, hip_L, hip_R] 추정 각도.
        qd_hat : np.ndarray, shape (4,)  [deg/s]
            [q̇_knee_L, q̇_knee_R, q̇_hip_L, q̇_hip_R] 추정 각속도.
        """
        m = self.N_OBS
        return self._x[:m].copy(), self._x[m:].copy()

    def get_compensated(self, latency_s: float = 0.021) -> np.ndarray:
        """
        Latency 보상된 현재 시각 joint angle 추정값.

        카메라 영상은 latency_s 이전 시점의 포즈를 담고 있다.
        현재 추정 state에서 constant-velocity 가정으로
        latency_s 만큼 forward-extrapolate:

          q̂_now ≈ q̂ + latency_s * q̂̇

        Parameters
        ----------
        latency_s : float
            카메라 ~ 제어 루프 간 지연 [s].  Default 0.021 (21 ms).

        Returns
        -------
        q_compensated : np.ndarray, shape (4,)  [deg]
            latency 보상된 [knee_L, knee_R, hip_L, hip_R].
        """
        m = self.N_OBS
        return self._x[:m] + latency_s * self._x[m:]

    def reset(self) -> None:
        """State, covariance를 초기 상태로 초기화."""
        self._x[:] = 0.0
        np.fill_diagonal(self._P, 1e4)
        self._P[...] = np.eye(self.N_STATE, dtype=np.float64) * 1e4
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """첫 update() 호출 여부."""
        return self._initialized


# ─────────────────────────────────────────────────────────────
# __main__ : 합성 데이터 테스트
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("GaitKalmanFilter 합성 데이터 테스트")
    print("  50Hz / noisy sinusoidal angle / latency 21ms")
    print("=" * 60)

    # ── 파라미터 ──────────────────────────────────────────────
    FS        = 50.0            # [Hz]
    DT        = 1.0 / FS        # 0.020 s
    DURATION  = 5.0             # [s]
    LATENCY   = 0.021           # [s]  카메라 지연
    N_STEPS   = int(DURATION * FS)

    NOISE_DEG  = 2.5            # 카메라 noise 1-σ [deg]
    FREQ_GAIT  = 0.8            # 보행 주파수 [Hz]
    AMP        = np.array([40.0, 40.0, 25.0, 25.0])    # 진폭 [deg]
    BIAS       = np.array([90.0, 90.0, 170.0, 170.0])  # 중립 [deg]
    PHASE      = np.array([0.0, np.pi, 0.5*np.pi, 1.5*np.pi])

    rng = np.random.default_rng(42)

    # ── True signal ────────────────────────────────────────────
    t = np.arange(N_STEPS) * DT

    # shape (N, 4)
    q_true = BIAS + AMP * np.sin(
        2 * np.pi * FREQ_GAIT * t[:, None] + PHASE[None, :]
    )
    # True angular velocity [deg/s]
    qd_true = AMP * 2 * np.pi * FREQ_GAIT * np.cos(
        2 * np.pi * FREQ_GAIT * t[:, None] + PHASE[None, :]
    )

    # Noisy measurement
    q_meas = q_true + rng.normal(0, NOISE_DEG, (N_STEPS, 4))

    # ── Filter 실행 ────────────────────────────────────────────
    kf = GaitKalmanFilter(
        dt=DT,
        angle_noise_deg=NOISE_DEG,
        accel_noise_deg_s2=1000.0,
    )

    q_filtered    = np.zeros((N_STEPS, 4))
    qd_filtered   = np.zeros((N_STEPS, 4))
    q_compensated = np.zeros((N_STEPS, 4))

    for i in range(N_STEPS):
        kf.predict(DT)
        kf.update(q_meas[i])
        q_f, qd_f          = kf.get_state()
        q_filtered[i]      = q_f
        qd_filtered[i]     = qd_f
        q_compensated[i]   = kf.get_compensated(LATENCY)

    # ── 오차 분석 (후반 2초 steady-state) ─────────────────────
    ss = int(3.0 * FS)   # 3s 이후 steady-state

    rmse_raw  = np.sqrt(np.mean((q_meas[ss:]     - q_true[ss:]) ** 2, axis=0))
    rmse_filt = np.sqrt(np.mean((q_filtered[ss:] - q_true[ss:]) ** 2, axis=0))

    # Latency 보상 효과:
    #   filtered[i]    = estimate at camera-capture time t_i
    #   compensated[i] = estimate at t_i + latency  (current time)
    #   compare against q_true at t_i + latency
    lat_samples = int(round(LATENCY * FS))   # 1 sample @ 50Hz
    if lat_samples > 0:
        # 슬라이싱: filtered[ss:N-lat] ↔ true[ss+lat:N]
        end = N_STEPS - lat_samples
        q_filt_slice = q_filtered[ss:end]
        q_comp_slice = q_compensated[ss:end]
        q_true_ahead = q_true[ss + lat_samples: N_STEPS]

        rmse_no_comp = np.sqrt(np.mean((q_filt_slice - q_true_ahead) ** 2, axis=0))
        rmse_comp    = np.sqrt(np.mean((q_comp_slice - q_true_ahead) ** 2, axis=0))
    else:
        rmse_no_comp = rmse_filt.copy()
        rmse_comp    = rmse_filt.copy()

    # 각속도 RMSE
    rmse_qd = np.sqrt(np.mean((qd_filtered[ss:] - qd_true[ss:]) ** 2, axis=0))

    labels = ["knee_L", "knee_R", "hip_L ", "hip_R "]

    print("\n[1] 노이즈 스무딩 효과 (RMSE vs true, steady-state 3~5s)")
    print(f"  {'Joint':<10} {'Raw [deg]':>10} {'Filtered [deg]':>15}  {'개선율':>8}")
    for j in range(4):
        imp = (1.0 - rmse_filt[j] / rmse_raw[j]) * 100.0
        print(f"  {labels[j]:<10} {rmse_raw[j]:>10.3f} {rmse_filt[j]:>15.3f}  {imp:>7.1f}%")

    print("\n[2] Latency 보상 효과 (RMSE vs 현재 true)")
    print(f"  latency = {LATENCY*1000:.1f} ms  ({lat_samples} sample @ {FS:.0f}Hz)")
    print(f"  {'Joint':<10} {'No-comp [deg]':>13} {'Compensated [deg]':>18}  {'개선율':>8}")
    for j in range(4):
        if rmse_no_comp[j] > 1e-9:
            imp = (1.0 - rmse_comp[j] / rmse_no_comp[j]) * 100.0
        else:
            imp = 0.0
        print(f"  {labels[j]:<10} {rmse_no_comp[j]:>13.3f} {rmse_comp[j]:>18.3f}  {imp:>7.1f}%")

    print("\n[3] 각속도 추정 정확도 (RMSE, steady-state 3~5s)")
    print(f"  {'Joint':<10} {'q̇ RMSE [deg/s]':>16}")
    for j in range(4):
        print(f"  {labels[j]:<10} {rmse_qd[j]:>16.3f}")

    print("\n[4] 최종 state 샘플 (마지막 5 프레임)")
    print(f"  {'t[s]':>6}  {'q_true':>8}  {'q_raw':>7}  {'q_filt':>7}  {'q_comp':>7}  {'q̇_filt':>9}")
    for i in range(N_STEPS - 5, N_STEPS):
        print(f"  {t[i]:>6.3f}  "
              f"{q_true[i,0]:>8.2f}  "
              f"{q_meas[i,0]:>7.2f}  "
              f"{q_filtered[i,0]:>7.2f}  "
              f"{q_compensated[i,0]:>7.2f}  "
              f"{qd_filtered[i,0]:>9.2f}")

    print("\nOK — GaitKalmanFilter 동작 확인 완료")
    sys.exit(0)
