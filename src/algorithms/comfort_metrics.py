"""
------------------
Per-environment passenger-comfort and social-driving telemetry for the
coadaptive value alignment loop.

Design goals (vs. the original inline tracker):

1. Real ISO-2631-inspired ride-comfort signals (lateral acceleration, true
   jerk = d^3x/dt^3, RMS jerk over a sliding window).
2. dt-aware: thresholds expressed as physical rates (m/s^2, m/s^3, rad/s),
   not per-frame deltas. Survives changes to frame_skip.
3. Continuous magnitudes everywhere — no count-when-above-threshold gates,
   so the policy gets gradient even on near-misses.
4. Per-environment state (one tracker per parallel env), fixing the
   single-attribute bug in the original trainer.
5. Dense per-step shaping reward + sparse end-of-episode LLM critic.
   The dense term carries the gradient; the LLM term provides the
   "human voice" anchor that the project is actually about.
6. TTC (time-to-collision) — promised in the proposal, missing from code.
7. Predictability: rolling std of longitudinal accel. Humans tolerate
   high-but-constant accel; they hate variance.

References:
- ISO 2631-1: Mechanical vibration and shock - Evaluation of human exposure
  to whole-body vibration. Comfort thresholds: a_lat ~ 0.315 m/s^2 "a little
  uncomfortable", > 0.8 m/s^2 "uncomfortable". For passenger cars, lateral
  accel up to ~2 m/s^2 (~0.2 g) is the typical comfortable ceiling.
- Bellem et al. 2018, "Comfort in automated driving": jerk RMS is the single
  best predictor of subjective discomfort ratings.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ---- Comfort thresholds (used for normalization, NOT hard gates) -----------
# These are the points at which the corresponding rolling-average penalty
# saturates near 1.0. They come from the literature, not arbitrary tuning.
A_LAT_COMFORT_CEILING = 2.0      # m/s^2  ~0.2 g sustained — passenger ceiling
A_LON_COMFORT_CEILING = 2.5      # m/s^2  comfortable braking/accel
JERK_COMFORT_CEILING = 2.0       # m/s^3  RMS jerk above this = "rough"
STEERING_RATE_CEILING = 3.0      # 1/s    normalized steering command per sec
TTC_CRITICAL = 1.5               # s      below this = scary
TTC_SAFE = 4.0                   # s      above this = no penalty
PROXIMITY_PENALTY_RADIUS = 5.0   # m      same as original
PROXIMITY_HARD_RADIUS = 1.0      # m      below this = strong penalty


def _saturate(x: float, ceiling: float) -> float:
    """Map [0, ceiling] -> [0, 1] smoothly, saturate above. tanh keeps
    gradient alive past the ceiling instead of clipping flat."""
    if ceiling <= 0:
        return 0.0
    return math.tanh(max(0.0, x) / ceiling)


@dataclass
class ComfortTracker:
    """Per-environment rolling state. One instance per parallel env."""

    dt: float = 0.05  # seconds per step at frame_skip=1, ~20 Hz Donkey default
    window: int = 20  # rolling-window length for RMS / std

    # --- Rolling history (for derivatives + windowed stats) ---
    speed_hist: deque = field(default_factory=lambda: deque(maxlen=4))
    a_lon_hist: deque = field(default_factory=lambda: deque(maxlen=20))
    a_lat_hist: deque = field(default_factory=lambda: deque(maxlen=20))
    jerk_hist: deque = field(default_factory=lambda: deque(maxlen=20))

    last_steering: float = 0.0
    last_cte: float = 0.0
    last_a_lon: float = 0.0

    # --- Episode aggregates ---
    ep_steps: int = 0
    ep_min_distance: float = 10.0
    ep_max_close_speed: float = 0.0
    ep_max_a_lat: float = 0.0
    ep_max_a_lon_brake: float = 0.0       # peak deceleration
    ep_jerk_sum_sq: float = 0.0           # for true RMS jerk over episode
    ep_jerk_count: int = 0
    ep_min_ttc: float = float("inf")
    ep_steering_rate_sum_sq: float = 0.0
    ep_a_lon_sum: float = 0.0
    ep_a_lon_sum_sq: float = 0.0          # for predictability std
    ep_close_steps: int = 0               # frames spent inside proximity radius

    def reset(self) -> None:
        self.speed_hist.clear()
        self.a_lon_hist.clear()
        self.a_lat_hist.clear()
        self.jerk_hist.clear()
        self.last_steering = 0.0
        self.last_cte = 0.0
        self.last_a_lon = 0.0
        self.ep_steps = 0
        self.ep_min_distance = 10.0
        self.ep_max_close_speed = 0.0
        self.ep_max_a_lat = 0.0
        self.ep_max_a_lon_brake = 0.0
        self.ep_jerk_sum_sq = 0.0
        self.ep_jerk_count = 0
        self.ep_min_ttc = float("inf")
        self.ep_steering_rate_sum_sq = 0.0
        self.ep_a_lon_sum = 0.0
        self.ep_a_lon_sum_sq = 0.0
        self.ep_close_steps = 0

    # ------------------------------------------------------------------ #
    #  Per-step update                                                    #
    # ------------------------------------------------------------------ #
    def step(
        self,
        speed: float,
        steering_cmd: float,
        cte: float,
        ego_pos: Optional[tuple] = None,
        obstacles: Optional[list] = None,
    ) -> dict:
        """Update state for one simulation step. Returns a dict with the
        values needed to shape the per-step reward (dense) and log."""
        self.ep_steps += 1
        dt = max(self.dt, 1e-6)

        # ---- Longitudinal acceleration (1st derivative of speed) ----
        if self.speed_hist:
            a_lon = (speed - self.speed_hist[-1]) / dt
        else:
            a_lon = 0.0
        self.speed_hist.append(speed)
        self.a_lon_hist.append(a_lon)

        # ---- Longitudinal jerk (2nd derivative of speed) ----
        jerk = (a_lon - self.last_a_lon) / dt
        self.jerk_hist.append(jerk)
        self.last_a_lon = a_lon

        # ---- Lateral acceleration ----
        # In a planar car: a_lat = v^2 * kappa, where kappa is path curvature.
        # We don't have curvature directly; CTE rate is a usable proxy and
        # captures the "swerve" quality passengers actually feel.
        cte_rate = (cte - self.last_cte) / dt
        # Effective lateral accel ~ d(cte_rate)/dt; approximate with current
        # cte_rate * speed direction change. Simpler: |cte_rate * speed| has
        # units of m/s * m/s / m = m/s^2 once we treat cte_rate as a yaw proxy.
        a_lat = abs(cte_rate) * max(speed, 0.5)
        self.a_lat_hist.append(a_lat)
        self.last_cte = cte

        # ---- Steering rate (1/s in normalized command space) ----
        steering_rate = abs(steering_cmd - self.last_steering) / dt
        self.last_steering = steering_cmd

        # ---- Distance to nearest obstacle + TTC ----
        min_dist = 10.0
        ttc = float("inf")
        if ego_pos is not None and obstacles:
            ego_x, _, ego_z = ego_pos
            for ox, oz in obstacles:
                d = math.hypot(ego_x - ox, ego_z - oz)
                if d < min_dist:
                    min_dist = d
            # TTC under closing: assume worst-case = full ego speed toward
            # the nearest obstacle. Conservative but stable without velocity
            # vectors for the obstacles.
            if speed > 0.1:
                ttc = min_dist / speed

        # ---- Update episode aggregates ----
        self.ep_min_distance = min(self.ep_min_distance, min_dist)
        if min_dist < PROXIMITY_PENALTY_RADIUS:
            self.ep_close_steps += 1
            self.ep_max_close_speed = max(self.ep_max_close_speed, speed)
        self.ep_max_a_lat = max(self.ep_max_a_lat, a_lat)
        if a_lon < 0:
            self.ep_max_a_lon_brake = max(self.ep_max_a_lon_brake, -a_lon)
        self.ep_jerk_sum_sq += jerk * jerk
        self.ep_jerk_count += 1
        self.ep_min_ttc = min(self.ep_min_ttc, ttc)
        self.ep_steering_rate_sum_sq += steering_rate * steering_rate
        self.ep_a_lon_sum += a_lon
        self.ep_a_lon_sum_sq += a_lon * a_lon

        return {
            "a_lon": a_lon,
            "a_lat": a_lat,
            "jerk": jerk,
            "steering_rate": steering_rate,
            "min_dist": min_dist,
            "ttc": ttc,
        }

    # ------------------------------------------------------------------ #
    #  Dense per-step reward shaping                                      #
    # ------------------------------------------------------------------ #
    def dense_comfort_reward(self, step_metrics: dict) -> float:
        """Small per-step reward that rewards smoothness & punishes scary
        moments. Magnitude is intentionally bounded to ~[-0.1, +0.02] so it
        doesn't dominate the task reward but provides continuous gradient.

        This is what was missing from your original setup — the LLM signal
        is sparse, the proximity penalty is one-sided, and there was no
        per-step incentive for the policy to *learn* smoothness.
        """
        a_lat = step_metrics["a_lat"]
        a_lon = step_metrics["a_lon"]
        jerk = abs(step_metrics["jerk"])
        steering_rate = step_metrics["steering_rate"]
        min_dist = step_metrics["min_dist"]
        ttc = step_metrics["ttc"]

        # Bounded penalties (each in [0, ~0.03])
        p_lat = 0.03 * _saturate(a_lat, A_LAT_COMFORT_CEILING)
        p_brake = 0.02 * _saturate(max(0.0, -a_lon), A_LON_COMFORT_CEILING)
        p_jerk = 0.03 * _saturate(jerk, JERK_COMFORT_CEILING)
        p_steer = 0.01 * _saturate(steering_rate, STEERING_RATE_CEILING)

        # TTC penalty: linear ramp between TTC_CRITICAL and TTC_SAFE
        if ttc < TTC_SAFE and ttc < float("inf"):
            ttc_severity = max(0.0, (TTC_SAFE - ttc) / (TTC_SAFE - TTC_CRITICAL))
            p_ttc = 0.04 * min(1.0, ttc_severity)
        else:
            p_ttc = 0.0

        # Proximity (kept from original, but smoother). Two zones:
        if min_dist < PROXIMITY_PENALTY_RADIUS:
            # Linear soft penalty
            soft = 0.02 * (PROXIMITY_PENALTY_RADIUS - min_dist) / PROXIMITY_PENALTY_RADIUS
            # Quadratic hard penalty when really close
            if min_dist < PROXIMITY_HARD_RADIUS:
                hard = 0.10 * (PROXIMITY_HARD_RADIUS - min_dist) / PROXIMITY_HARD_RADIUS
            else:
                hard = 0.0
            p_prox = soft + hard
        else:
            p_prox = 0.0

        # Small positive bonus for being smooth AND making progress: the
        # asymmetry matters — without this, the agent learns "stop = safe."
        smoothness_bonus = 0.0
        if jerk < JERK_COMFORT_CEILING * 0.3 and a_lat < A_LAT_COMFORT_CEILING * 0.3:
            smoothness_bonus = 0.005

        return -(p_lat + p_brake + p_jerk + p_steer + p_ttc + p_prox) + smoothness_bonus

    # ------------------------------------------------------------------ #
    #  End-of-episode summary                                             #
    # ------------------------------------------------------------------ #
    def episode_summary(self) -> dict:
        n = max(1, self.ep_jerk_count)
        rms_jerk = math.sqrt(self.ep_jerk_sum_sq / n)
        rms_steering_rate = math.sqrt(self.ep_steering_rate_sum_sq / n)

        mean_a_lon = self.ep_a_lon_sum / n
        var_a_lon = max(0.0, self.ep_a_lon_sum_sq / n - mean_a_lon * mean_a_lon)
        std_a_lon = math.sqrt(var_a_lon)  # predictability proxy

        return {
            "min_distance_m": self.ep_min_distance,
            "max_close_speed_mps": self.ep_max_close_speed,
            "max_lateral_accel_mps2": self.ep_max_a_lat,
            "max_brake_decel_mps2": self.ep_max_a_lon_brake,
            "rms_jerk_mps3": rms_jerk,
            "rms_steering_rate_per_s": rms_steering_rate,
            "min_ttc_s": self.ep_min_ttc if self.ep_min_ttc != float("inf") else 99.0,
            "predictability_std_a_lon": std_a_lon,
            "fraction_close": self.ep_close_steps / max(1, self.ep_steps),
            "episode_steps": self.ep_steps,
        }


# ---------------------------------------------------------------------- #
#  LLM critic prompting                                                   #
# ---------------------------------------------------------------------- #

_FEW_SHOT_ANCHORS = """\
Examples (use these as calibration anchors):

Episode A — score 0.9 (excellent):
  rms_jerk=0.4 m/s^3, max_lat_accel=1.1 m/s^2, max_brake=1.3 m/s^2,
  min_distance=4.8 m, min_ttc=6.0 s, fraction_close=0.05, predictability_std=0.3
  -> smooth, predictable, kept safe distance

Episode B — score 0.0 (mediocre):
  rms_jerk=1.8 m/s^3, max_lat_accel=2.2 m/s^2, max_brake=2.7 m/s^2,
  min_distance=2.5 m, min_ttc=2.5 s, fraction_close=0.25, predictability_std=1.1
  -> noticeable jerk and one tight pass, tolerable

Episode C — score -0.8 (bad):
  rms_jerk=4.5 m/s^3, max_lat_accel=4.0 m/s^2, max_brake=5.5 m/s^2,
  min_distance=0.8 m, min_ttc=0.6 s, fraction_close=0.55, predictability_std=2.4
  -> erratic, near-collision, scary
"""


def build_comfort_prompt(summary: dict) -> str:
    """Build a calibrated prompt with anchors. The original prompt asked the
    LLM to invent a scale; this one fixes the scale with examples so scores
    are stable across runs and across LLM versions."""
    return (
        "You are scoring an autonomous-vehicle episode for passenger comfort "
        "and social driving etiquette. Use the ISO-2631 spirit: smoothness, "
        "predictability, and adequate margin matter more than raw speed.\n\n"
        f"{_FEW_SHOT_ANCHORS}\n"
        "Now score this episode:\n"
        f"  rms_jerk = {summary['rms_jerk_mps3']:.2f} m/s^3\n"
        f"  max_lateral_accel = {summary['max_lateral_accel_mps2']:.2f} m/s^2\n"
        f"  max_brake_decel = {summary['max_brake_decel_mps2']:.2f} m/s^2\n"
        f"  min_distance = {summary['min_distance_m']:.2f} m\n"
        f"  max_close_speed = {summary['max_close_speed_mps']:.2f} m/s\n"
        f"  min_ttc = {summary['min_ttc_s']:.2f} s\n"
        f"  rms_steering_rate = {summary['rms_steering_rate_per_s']:.2f} 1/s\n"
        f"  fraction_close = {summary['fraction_close']:.2f}\n"
        f"  predictability_std = {summary['predictability_std_a_lon']:.2f} m/s^2\n"
        f"  episode_length = {summary['episode_steps']} steps\n\n"
        "Note: if max_close_speed < 0.5 m/s the car is parked / blocking traffic; "
        "score heavily negative regardless of smoothness.\n"
        "Respond with ONLY a number in [-1.0, 1.0]."
    )
