"""
-----------------------
Passenger-comfort and social-driving telemetry for the coadaptive value
alignment loop.

CHANGES FROM v1:
  * Lateral acceleration now computed from REAL yaw rate:
        a_lat = |v * yaw_rate|,   yaw_rate = dyaw/dt (rad/s)
    The v1 formula `|d(cte)/dt| * speed` blew up on respawns and on path-node
    transitions, producing saturated values in the 100+ m/s^2 range.
  * Respawn guard: single-step anomalies in yaw, speed, or cte are detected
    and that step's derivative values are suppressed (zeroed) so they don't
    poison the RMS / max aggregates. Without this, one mid-episode teleport
    would make the entire episode look catastrophic.
  * Obstacle distances now assume SCALED coordinates (both ego and obstacle
    in the same frame /8.0). The old code read obstacle positions via the
    "gyro/accel" smuggler channel in UNSCALED Unity world coords while
    ego_pos was in scaled /8.0 coords — a silent 8x mismatch that made
    every distance look ~8x too large and max_close_speed always 0.
    The PPO script must now pass scaled obstacle positions (the dedicated
    `broken_car_x/z`, `moving_car_x/z` Unity fields already provide these).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ---- Comfort thresholds (used for normalization, NOT hard gates) -----------
A_LAT_COMFORT_CEILING = 2.0
A_LON_COMFORT_CEILING = 2.5
JERK_COMFORT_CEILING = 2.0
STEERING_RATE_CEILING = 3.0
TTC_CRITICAL = 1.5
TTC_SAFE = 4.0
PROXIMITY_PENALTY_RADIUS = 5.0
PROXIMITY_HARD_RADIUS = 1.0

# ---- Respawn / teleport detection ------------------------------------------
RESPAWN_SPEED_JUMP = 3.0         # m/s in one step (at 20Hz = 60 m/s^2 accel — unphysical)
RESPAWN_YAW_JUMP_DEG = 60.0
RESPAWN_CTE_JUMP = 2.0


def _saturate(x: float, ceiling: float) -> float:
    if ceiling <= 0:
        return 0.0
    return math.tanh(max(0.0, x) / ceiling)


def _wrap_angle_deg(delta_deg: float) -> float:
    """Wrap an angle delta to [-180, 180] degrees."""
    d = (delta_deg + 180.0) % 360.0 - 180.0
    return d


@dataclass
class ComfortTracker:
    dt: float = 0.05

    speed_hist: deque = field(default_factory=lambda: deque(maxlen=4))
    a_lon_hist: deque = field(default_factory=lambda: deque(maxlen=20))
    a_lat_hist: deque = field(default_factory=lambda: deque(maxlen=20))
    jerk_hist: deque = field(default_factory=lambda: deque(maxlen=20))

    last_steering: float = 0.0
    last_cte: float = 0.0
    last_yaw: Optional[float] = None
    last_speed: Optional[float] = None
    last_a_lon: float = 0.0

    ep_steps: int = 0
    ep_valid_deriv_steps: int = 0
    ep_min_distance: float = 10.0
    ep_max_close_speed: float = 0.0
    ep_max_a_lat: float = 0.0
    ep_max_a_lon_brake: float = 0.0
    ep_jerk_sum_sq: float = 0.0
    ep_jerk_count: int = 0
    ep_min_ttc: float = float("inf")
    ep_steering_rate_sum_sq: float = 0.0
    ep_a_lon_sum: float = 0.0
    ep_a_lon_sum_sq: float = 0.0
    ep_close_steps: int = 0
    ep_respawn_events: int = 0

    def reset(self) -> None:
        self.speed_hist.clear()
        self.a_lon_hist.clear()
        self.a_lat_hist.clear()
        self.jerk_hist.clear()
        self.last_steering = 0.0
        self.last_cte = 0.0
        self.last_yaw = None
        self.last_speed = None
        self.last_a_lon = 0.0
        self.ep_steps = 0
        self.ep_valid_deriv_steps = 0
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
        self.ep_respawn_events = 0

    def step(
        self,
        speed: float,
        steering_cmd: float,
        cte: float,
        yaw_deg: Optional[float] = None,
        ego_pos: Optional[tuple] = None,
        obstacles: Optional[list] = None,
    ) -> dict:
        """Update state for one simulation step.

        Args:
            speed: m/s (Unity `speed` field — already scaled /8)
            steering_cmd: action in [-1, 1]
            cte: m, cross-track error
            yaw_deg: deg in [0, 360); None falls back to the CTE-rate hack.
            ego_pos: (x, y, z) — MUST be in same frame as obstacles
            obstacles: list of (x, z) — MUST be in same frame as ego_pos
        """
        self.ep_steps += 1
        dt = max(self.dt, 1e-6)

        # Respawn / discontinuity detection
        is_discontinuity = False
        if self.last_speed is not None and abs(speed - self.last_speed) > RESPAWN_SPEED_JUMP:
            is_discontinuity = True
        if abs(cte - self.last_cte) > RESPAWN_CTE_JUMP:
            is_discontinuity = True
        if yaw_deg is not None and self.last_yaw is not None:
            if abs(_wrap_angle_deg(yaw_deg - self.last_yaw)) > RESPAWN_YAW_JUMP_DEG:
                is_discontinuity = True
        if is_discontinuity:
            self.ep_respawn_events += 1

        # Longitudinal acceleration
        if self.last_speed is not None and not is_discontinuity:
            a_lon = (speed - self.last_speed) / dt
            valid_a_lon = True
        else:
            a_lon = 0.0
            valid_a_lon = False
        self.last_speed = speed
        self.speed_hist.append(speed)
        self.a_lon_hist.append(a_lon)

        # Longitudinal jerk
        if valid_a_lon and not is_discontinuity:
            jerk = (a_lon - self.last_a_lon) / dt
            valid_jerk = True
        else:
            jerk = 0.0
            valid_jerk = False
        self.jerk_hist.append(jerk)
        self.last_a_lon = a_lon

        # Lateral acceleration — PROPER yaw-rate version
        a_lat = 0.0
        valid_a_lat = False
        if yaw_deg is not None and self.last_yaw is not None and not is_discontinuity:
            yaw_delta = _wrap_angle_deg(yaw_deg - self.last_yaw)
            yaw_rate = math.radians(yaw_delta) / dt
            a_lat = abs(speed * yaw_rate)
            valid_a_lat = True
        elif yaw_deg is None and not is_discontinuity:
            # Fallback (v1 formula) — noisier, capped so it can't explode
            cte_rate = (cte - self.last_cte) / dt
            a_lat = min(abs(cte_rate) * max(speed, 0.5), 10.0)
            valid_a_lat = True
        self.a_lat_hist.append(a_lat)
        self.last_yaw = yaw_deg
        self.last_cte = cte

        # Steering rate
        steering_rate = abs(steering_cmd - self.last_steering) / dt
        self.last_steering = steering_cmd

        # Distance + TTC (both frames MUST match)
        min_dist = 10.0
        ttc = float("inf")
        if ego_pos is not None and obstacles:
            ego_x, _, ego_z = ego_pos
            for ox, oz in obstacles:
                d = math.hypot(ego_x - ox, ego_z - oz)
                if d < min_dist:
                    min_dist = d
            if speed > 0.1:
                ttc = min_dist / speed

        # Aggregates
        if not is_discontinuity:
            self.ep_valid_deriv_steps += 1
        self.ep_min_distance = min(self.ep_min_distance, min_dist)
        if min_dist < PROXIMITY_PENALTY_RADIUS:
            self.ep_close_steps += 1
            self.ep_max_close_speed = max(self.ep_max_close_speed, speed)
        if valid_a_lat:
            self.ep_max_a_lat = max(self.ep_max_a_lat, a_lat)
        if valid_a_lon and a_lon < 0:
            self.ep_max_a_lon_brake = max(self.ep_max_a_lon_brake, -a_lon)
        if valid_jerk:
            self.ep_jerk_sum_sq += jerk * jerk
            self.ep_jerk_count += 1
        self.ep_min_ttc = min(self.ep_min_ttc, ttc)
        self.ep_steering_rate_sum_sq += steering_rate * steering_rate
        if valid_a_lon:
            self.ep_a_lon_sum += a_lon
            self.ep_a_lon_sum_sq += a_lon * a_lon

        return {
            "a_lon": a_lon if valid_a_lon else 0.0,
            "a_lat": a_lat if valid_a_lat else 0.0,
            "jerk": jerk if valid_jerk else 0.0,
            "steering_rate": steering_rate,
            "min_dist": min_dist,
            "ttc": ttc,
            "respawn": is_discontinuity,
        }

    def dense_comfort_reward(self, step_metrics: dict) -> float:
        # On respawn steps, don't penalize for derivatives we know are garbage.
        if step_metrics.get("respawn", False):
            return 0.0

        a_lat = step_metrics["a_lat"]
        a_lon = step_metrics["a_lon"]
        jerk = abs(step_metrics["jerk"])
        steering_rate = step_metrics["steering_rate"]
        min_dist = step_metrics["min_dist"]
        ttc = step_metrics["ttc"]

        p_lat = 0.03 * _saturate(a_lat, A_LAT_COMFORT_CEILING)
        p_brake = 0.02 * _saturate(max(0.0, -a_lon), A_LON_COMFORT_CEILING)
        p_jerk = 0.03 * _saturate(jerk, JERK_COMFORT_CEILING)
        p_steer = 0.01 * _saturate(steering_rate, STEERING_RATE_CEILING)

        if ttc < TTC_SAFE and ttc < float("inf"):
            ttc_severity = max(0.0, (TTC_SAFE - ttc) / (TTC_SAFE - TTC_CRITICAL))
            p_ttc = 0.04 * min(1.0, ttc_severity)
        else:
            p_ttc = 0.0

        if min_dist < PROXIMITY_PENALTY_RADIUS:
            soft = 0.02 * (PROXIMITY_PENALTY_RADIUS - min_dist) / PROXIMITY_PENALTY_RADIUS
            if min_dist < PROXIMITY_HARD_RADIUS:
                hard = 0.10 * (PROXIMITY_HARD_RADIUS - min_dist) / PROXIMITY_HARD_RADIUS
            else:
                hard = 0.0
            p_prox = soft + hard
        else:
            p_prox = 0.0

        smoothness_bonus = 0.0
        if jerk < JERK_COMFORT_CEILING * 0.3 and a_lat < A_LAT_COMFORT_CEILING * 0.3:
            smoothness_bonus = 0.005

        return -(p_lat + p_brake + p_jerk + p_steer + p_ttc + p_prox) + smoothness_bonus

    def episode_summary(self) -> dict:
        n_jerk = max(1, self.ep_jerk_count)
        rms_jerk = math.sqrt(self.ep_jerk_sum_sq / n_jerk)
        n_steer = max(1, self.ep_steps)
        rms_steering_rate = math.sqrt(self.ep_steering_rate_sum_sq / n_steer)
        n_lon = max(1, self.ep_valid_deriv_steps)
        mean_a_lon = self.ep_a_lon_sum / n_lon
        var_a_lon = max(0.0, self.ep_a_lon_sum_sq / n_lon - mean_a_lon * mean_a_lon)
        std_a_lon = math.sqrt(var_a_lon)

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
            "respawn_events": self.ep_respawn_events,
        }


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
