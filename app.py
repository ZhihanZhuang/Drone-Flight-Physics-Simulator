from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# App configuration + styling
# -----------------------------

st.set_page_config(
    page_title="Drone Flight Physics Simulator",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_APP_CSS = """
<style>
  /* Layout polish */
  .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1400px; }
  [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }

  /* Title */
  .hero {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 18px 18px 14px 18px;
    background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(14,165,233,0.08), rgba(34,197,94,0.06));
  }
  .hero-title {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.2px;
    margin: 0 0 2px 0;
  }
  .hero-subtitle {
    font-size: 0.98rem;
    opacity: 0.85;
    margin: 0;
  }

  /* KPI cards */
  .kpi-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
  @media (max-width: 1200px) { .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  @media (max-width: 700px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  .kpi-card {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 12px 12px 10px 12px;
    background: rgba(255,255,255,0.03);
  }
  .kpi-label { font-size: 0.78rem; opacity: 0.78; margin-bottom: 6px; }
  .kpi-value { font-size: 1.2rem; font-weight: 800; margin: 0; line-height: 1.0; }
  .kpi-foot { font-size: 0.78rem; opacity: 0.68; margin-top: 6px; }

  /* Section cards */
  .panel {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 14px;
    background: rgba(255,255,255,0.02);
  }
</style>
"""
st.markdown(_APP_CSS, unsafe_allow_html=True)


# -----------------------------
# Domain model
# -----------------------------

PropSize = Literal["Small", "Medium", "Large"]


@dataclass(frozen=True)
class DroneParams:
    # Drone Physical Parameters
    mass_kg: float
    gravity: float

    # Battery Parameters
    voltage_v: float
    capacity_mah: int
    battery_efficiency_pct: float

    # Motor Parameters
    motor_k: float
    max_rpm: int
    throttle_pct: float

    # Propeller Parameters
    prop_size: PropSize

    # Simulation Parameters
    dt: float
    max_time_s: float

    # Optional advanced knobs
    air_drag_coeff: float = 0.02  # mild drag for realism (N per (m/s)^2)
    payload_kg: float = 0.0


@dataclass(frozen=True)
class SimResult:
    series: pd.DataFrame  # time-series outputs
    params: DroneParams
    summary: Dict[str, Any]  # computed metrics, stability flags, score, etc.


def _prop_multipliers(prop_size: PropSize) -> Dict[str, float]:
    # Engineering-ish: larger props produce more thrust per RPM but cost more power (higher disc area).
    # We model this with separate multipliers rather than pretending it is "free thrust."
    if prop_size == "Small":
        return {"thrust_mult": 0.92, "power_mult": 0.95, "rpm_eff": 1.05}
    if prop_size == "Medium":
        return {"thrust_mult": 1.00, "power_mult": 1.00, "rpm_eff": 1.00}
    return {"thrust_mult": 1.10, "power_mult": 1.08, "rpm_eff": 0.96}


def _battery_energy_j(capacity_mah: int, voltage_v: float, efficiency_pct: float) -> float:
    # Joules ~= (Ah * V * 3600) * efficiency
    ah = capacity_mah / 1000.0
    return max(0.0, ah * voltage_v * 3600.0 * (efficiency_pct / 100.0))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# Physics simulation
# -----------------------------

def simulate_drone(p: DroneParams) -> SimResult:
    """
    Simplified vertical motion simulator.

    State:
      h: height (m), v: velocity (m/s)

    Dynamics:
      thrust = k * RPM^2
      RPM = throttle * max_rpm (with prop RPM-eff factor)
      F_net = thrust - m*g - drag(v)
      a = F_net / m

    Battery:
      energy -= power_used * dt
      power_used ~ c * thrust^(1.5) * prop_power_mult
    """
    m = max(0.01, p.mass_kg + p.payload_kg)
    g = max(0.0, p.gravity)
    dt = float(max(1e-4, p.dt))
    t_max = float(max(dt, p.max_time_s))

    prop = _prop_multipliers(p.prop_size)
    rpm = (p.throttle_pct / 100.0) * p.max_rpm * prop["rpm_eff"]
    rpm = _clamp(rpm, 0.0, float(p.max_rpm))

    # Thrust model
    thrust_n = p.motor_k * (rpm**2) * prop["thrust_mult"]
    thrust_n = max(0.0, float(thrust_n))

    # Battery initialization
    energy_j = _battery_energy_j(p.capacity_mah, p.voltage_v, p.battery_efficiency_pct)
    initial_energy_j = max(1e-9, energy_j)

    # Mild aero drag (quadratic)
    cd = max(0.0, float(p.air_drag_coeff))

    # Output buffers
    t_list: List[float] = []
    h_list: List[float] = []
    v_list: List[float] = []
    a_list: List[float] = []
    thrust_list: List[float] = []
    twr_list: List[float] = []
    energy_list: List[float] = []

    # State
    h = 0.0
    v = 0.0
    t = 0.0

    # Constants for power draw
    # Tuned so a typical design drains over tens of seconds to minutes, not milliseconds.
    power_c = 0.07

    # Stop reasons
    stop_reason = "max_time"
    crashed = False
    battery_empty = False

    # Sim loop
    while t <= t_max + 1e-12:
        weight_n = m * g
        drag_n = cd * (v**2) * (1.0 if v > 0 else -1.0)  # opposite direction
        f_net = thrust_n - weight_n - drag_n
        a = f_net / m

        # Integrate
        v = v + a * dt
        h = h + v * dt

        # Battery draw (W = J/s)
        power_w = power_c * (max(thrust_n, 0.0) ** 1.5) * prop["power_mult"]
        energy_j -= power_w * dt

        # Log
        t_list.append(t)
        h_list.append(h)
        v_list.append(v)
        a_list.append(a)
        thrust_list.append(thrust_n)
        twr_list.append(thrust_n / max(1e-9, weight_n))
        energy_list.append(max(0.0, energy_j))

        # Stop conditions
        if energy_j <= 0:
            battery_empty = True
            stop_reason = "battery_empty"
            break
        if h < -0.25 and t > 1.5:
            crashed = True
            stop_reason = "crash"
            break
        t += dt

    series = pd.DataFrame(
        {
            "time_s": t_list,
            "height_m": h_list,
            "velocity_mps": v_list,
            "accel_mps2": a_list,
            "thrust_n": thrust_list,
            "twr": twr_list,
            "battery_j": energy_list,
            "battery_pct": (np.array(energy_list) / initial_energy_j) * 100.0,
        }
    )
    series["battery_pct"] = series["battery_pct"].clip(lower=0.0, upper=100.0)

    summary = compute_metrics(series, p, stop_reason=stop_reason, crashed=crashed, battery_empty=battery_empty)
    return SimResult(series=series, params=p, summary=summary)


def compute_metrics(
    series: pd.DataFrame,
    p: DroneParams,
    *,
    stop_reason: str,
    crashed: bool,
    battery_empty: bool,
) -> Dict[str, Any]:
    if series.empty:
        return {
            "stop_reason": "no_data",
            "crashed": True,
            "battery_empty": True,
            "flight_time_s": 0.0,
            "max_altitude_m": 0.0,
            "final_battery_pct": 0.0,
            "peak_velocity_mps": 0.0,
            "peak_twr": 0.0,
            "stable": False,
            "stability_note": "No samples produced.",
            "score": -1e9,
        }

    flight_time_s = float(series["time_s"].iloc[-1])
    max_altitude_m = float(series["height_m"].max())
    final_battery_pct = float(series["battery_pct"].iloc[-1])
    peak_velocity_mps = float(series["velocity_mps"].abs().max())
    peak_twr = float(series["twr"].max())

    # Heuristic stability: did it avoid crash, and did it avoid runaway speed/altitude?
    # We flag "unstable" for (a) crash, (b) persistent upward acceleration late in the sim, (c) extreme oscillation.
    tail = series.tail(max(10, int(1.0 / max(p.dt, 1e-3))))
    tail_mean_accel = float(tail["accel_mps2"].mean()) if not tail.empty else 0.0
    velocity_sign_flips = int((np.sign(series["velocity_mps"]).diff().fillna(0) != 0).sum())
    too_many_flips = velocity_sign_flips > max(25, int(series.shape[0] * 0.12))
    runaway = (max_altitude_m > 1500.0) or (peak_velocity_mps > 120.0)
    stable = (not crashed) and (not runaway) and (not too_many_flips) and (tail_mean_accel < 2.5)

    if crashed:
        stability_note = "Crashed (height dropped below ground threshold)."
    elif runaway:
        stability_note = "Runaway dynamics detected (unrealistic altitude/velocity)."
    elif too_many_flips:
        stability_note = "High oscillation (velocity sign flips exceed threshold)."
    else:
        stability_note = "Stable by heuristic checks."

    # Score for optimization: flight time + altitude bonus - instability penalty - weight penalty
    altitude_bonus = 0.08 * max(0.0, max_altitude_m)
    instability_penalty = 250.0 if not stable else 0.0
    crash_penalty = 400.0 if crashed else 0.0
    weight_penalty = 18.0 * float(p.mass_kg)
    battery_bonus = 0.5 * final_battery_pct
    score = flight_time_s + altitude_bonus + battery_bonus - instability_penalty - crash_penalty - weight_penalty

    return {
        "stop_reason": stop_reason,
        "crashed": bool(crashed),
        "battery_empty": bool(battery_empty),
        "flight_time_s": flight_time_s,
        "max_altitude_m": max_altitude_m,
        "final_battery_pct": final_battery_pct,
        "peak_velocity_mps": peak_velocity_mps,
        "peak_twr": peak_twr,
        "stable": bool(stable),
        "stability_note": stability_note,
        "score": float(score),
    }


# -----------------------------
# Visualization
# -----------------------------

def _chart_template() -> Dict[str, Any]:
    return dict(
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
    )


def make_timeseries_chart(
    designs: List[Tuple[str, SimResult]],
    y_col: str,
    y_label: str,
    *,
    secondary_y: Optional[Tuple[str, str]] = None,
) -> go.Figure:
    fig = go.Figure()
    for name, res in designs:
        df = res.series
        fig.add_trace(
            go.Scatter(
                x=df["time_s"],
                y=df[y_col],
                mode="lines",
                name=f"{name} — {y_label}",
                line=dict(width=2.2),
            )
        )
    if secondary_y:
        sec_col, sec_label = secondary_y
        for name, res in designs:
            df = res.series
            fig.add_trace(
                go.Scatter(
                    x=df["time_s"],
                    y=df[sec_col],
                    mode="lines",
                    name=f"{name} — {sec_label}",
                    line=dict(width=1.6, dash="dot"),
                    yaxis="y2",
                    opacity=0.95,
                )
            )
        fig.update_layout(
            yaxis2=dict(
                title=sec_label,
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )

    fig.update_layout(
        **_chart_template(),
        xaxis_title="Time (s)",
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_overview_chart(designs: List[Tuple[str, SimResult]]) -> go.Figure:
    fig = go.Figure()
    for name, res in designs:
        df = res.series
        fig.add_trace(go.Scatter(x=df["time_s"], y=df["height_m"], name=f"{name} — Height (m)", mode="lines", line=dict(width=2.2)))
        fig.add_trace(go.Scatter(x=df["time_s"], y=df["velocity_mps"], name=f"{name} — Velocity (m/s)", mode="lines", line=dict(width=1.7, dash="dot"), opacity=0.9))
        fig.add_trace(go.Scatter(x=df["time_s"], y=df["battery_pct"], name=f"{name} — Battery (%)", mode="lines", line=dict(width=1.7, dash="dash"), opacity=0.9))
    fig.update_layout(
        **_chart_template(),
        xaxis_title="Time (s)",
        yaxis_title="Mixed Units (see legend)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# -----------------------------
# Optimization + Research
# -----------------------------

def optimize_design(
    base: DroneParams,
    *,
    n_weight: int = 6,
    n_battery: int = 6,
    n_motor: int = 6,
    prop_sizes: Optional[List[PropSize]] = None,
    throttle_policy: Literal["fixed", "sweep"] = "fixed",
    max_evals: int = 350,
    seed: int = 42,
    progress_cb: Optional[Any] = None,
) -> pd.DataFrame:
    rng = random.Random(seed)
    prop_sizes = prop_sizes or ["Small", "Medium", "Large"]

    weights = np.linspace(0.7, 6.0, n_weight).round(2).tolist()
    capacities = np.linspace(1200, 16000, n_battery).round(0).astype(int).tolist()
    motor_ks = np.geomspace(8e-6, 6e-5, n_motor).tolist()

    if throttle_policy == "fixed":
        throttles = [float(base.throttle_pct)]
    else:
        throttles = [45.0, 55.0, 65.0, 75.0, 85.0]

    grid = list(itertools.product(weights, capacities, motor_ks, prop_sizes, throttles))
    rng.shuffle(grid)
    grid = grid[: max_evals]

    rows: List[Dict[str, Any]] = []
    for i, (w, cap, k, prop, thr) in enumerate(grid, start=1):
        p = DroneParams(
            mass_kg=float(w),
            gravity=float(base.gravity),
            voltage_v=float(base.voltage_v),
            capacity_mah=int(cap),
            battery_efficiency_pct=float(base.battery_efficiency_pct),
            motor_k=float(k),
            max_rpm=int(base.max_rpm),
            throttle_pct=float(thr),
            prop_size=prop,  # type: ignore[arg-type]
            dt=float(base.dt),
            max_time_s=float(base.max_time_s),
            air_drag_coeff=float(base.air_drag_coeff),
            payload_kg=float(base.payload_kg),
        )
        res = simulate_drone(p)
        s = res.summary
        rows.append(
            {
                "score": s["score"],
                "stable": s["stable"],
                "stop_reason": s["stop_reason"],
                "flight_time_s": s["flight_time_s"],
                "max_altitude_m": s["max_altitude_m"],
                "final_battery_pct": s["final_battery_pct"],
                "peak_velocity_mps": s["peak_velocity_mps"],
                "peak_twr": s["peak_twr"],
                "weight_kg": p.mass_kg,
                "battery_mah": p.capacity_mah,
                "motor_k": p.motor_k,
                "prop_size": p.prop_size,
                "throttle_pct": p.throttle_pct,
                "max_rpm": p.max_rpm,
                "voltage_v": p.voltage_v,
                "efficiency_pct": p.battery_efficiency_pct,
            }
        )
        if progress_cb:
            progress_cb(i, len(grid))

    df = pd.DataFrame(rows).sort_values(["score", "stable"], ascending=[False, False]).reset_index(drop=True)
    return df


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str, minimize_x: bool = True, maximize_y: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    is_pareto = np.ones(df.shape[0], dtype=bool)
    for i in range(df.shape[0]):
        if not is_pareto[i]:
            continue
        for j in range(df.shape[0]):
            if i == j:
                continue
            better_x = x[j] <= x[i] if minimize_x else x[j] >= x[i]
            better_y = y[j] >= y[i] if maximize_y else y[j] <= y[i]
            strictly_better = (x[j] != x[i]) or (y[j] != y[i])
            if better_x and better_y and strictly_better:
                is_pareto[i] = False
                break
    return df[is_pareto].copy()


# -----------------------------
# UI helpers
# -----------------------------

def _kpi_card(label: str, value: str, foot: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-foot">{foot}</div>
    </div>
    """


def _fmt_s(x: float) -> str:
    return f"{x:,.1f}s"


def _fmt_m(x: float) -> str:
    return f"{x:,.1f}m"


def _fmt_pct(x: float) -> str:
    return f"{x:,.1f}%"


def _fmt_num(x: float) -> str:
    return f"{x:,.2f}"


def _design_label(p: DroneParams) -> str:
    return f"{p.mass_kg:.2f}kg • {p.capacity_mah}mAh @ {p.voltage_v:.1f}V • k={p.motor_k:.2e} • {p.prop_size} • {p.throttle_pct:.0f}%"


def _ensure_state():
    if "saved_designs" not in st.session_state:
        st.session_state.saved_designs = {"A": None, "B": None, "C": None}
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "optimizer_results" not in st.session_state:
        st.session_state.optimizer_results = None
    if "optimizer_best_params" not in st.session_state:
        st.session_state.optimizer_best_params = None


def _random_params(base: DroneParams, seed: Optional[int] = None) -> DroneParams:
    rng = random.Random(seed)
    prop = rng.choice(["Small", "Medium", "Large"])
    return DroneParams(
        mass_kg=float(_clamp(base.mass_kg * rng.uniform(0.6, 1.7), 0.5, 10.0)),
        gravity=float(base.gravity),
        voltage_v=float(_clamp(base.voltage_v * rng.uniform(0.85, 1.15), 7.4, 24.0)),
        capacity_mah=int(_clamp(base.capacity_mah * rng.uniform(0.6, 1.8), 1000, 20000)),
        battery_efficiency_pct=float(_clamp(base.battery_efficiency_pct * rng.uniform(0.85, 1.05), 50.0, 100.0)),
        motor_k=float(_clamp(base.motor_k * rng.uniform(0.65, 1.6), 1e-6, 1.2e-4)),
        max_rpm=int(_clamp(base.max_rpm * rng.uniform(0.8, 1.15), 8000, 40000)),
        throttle_pct=float(_clamp(base.throttle_pct + rng.uniform(-20, 20), 0.0, 100.0)),
        prop_size=prop,  # type: ignore[arg-type]
        dt=float(base.dt),
        max_time_s=float(base.max_time_s),
        air_drag_coeff=float(base.air_drag_coeff),
        payload_kg=float(base.payload_kg),
    )


# -----------------------------
# UI
# -----------------------------

_ensure_state()

st.markdown(
    """
<div class="hero">
  <div class="hero-title">Drone Flight Physics Simulator</div>
  <div class="hero-subtitle">Interactive UAV Design + Research Platform — vertical flight physics, design comparison, and automated optimization.</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

with st.sidebar:
    st.markdown("### Controls")
    st.caption("Tune the design, run the simulator, save variants, and optimize across a design space.")

    st.markdown("#### Drone Physical Parameters")
    mass_kg = st.slider("Weight (kg)", min_value=0.5, max_value=10.0, value=1.20, step=0.05)
    payload_kg = st.slider("Payload (kg)", min_value=0.0, max_value=5.0, value=0.00, step=0.05)
    gravity = st.number_input("Gravity (m/s²)", min_value=0.0, max_value=25.0, value=9.81, step=0.01)

    st.markdown("#### Battery Parameters")
    voltage_v = st.slider("Voltage (V)", min_value=7.4, max_value=24.0, value=11.1, step=0.1)
    capacity_mah = st.slider("Capacity (mAh)", min_value=1000, max_value=20000, value=5000, step=250)
    efficiency_pct = st.slider("Efficiency (%)", min_value=50, max_value=100, value=88, step=1)

    st.markdown("#### Motor Parameters")
    motor_k = st.number_input("Motor Strength Constant k", min_value=1e-6, max_value=1.2e-4, value=2.5e-5, step=1e-6, format="%.6f")
    max_rpm = st.slider("Max RPM", min_value=8000, max_value=40000, value=15000, step=250)
    throttle_pct = st.slider("Throttle (%)", min_value=0, max_value=100, value=72, step=1)

    st.markdown("#### Propeller Parameters")
    prop_size = st.selectbox("Prop size", ["Small", "Medium", "Large"], index=1)

    st.markdown("#### Simulation Parameters")
    dt = st.slider("Timestep dt (s)", min_value=0.01, max_value=0.30, value=0.10, step=0.01)
    max_time_s = st.slider("Max simulation time (s)", min_value=5, max_value=240, value=70, step=5)

    st.markdown("#### Advanced")
    air_drag_coeff = st.slider("Aerodynamic drag coefficient", min_value=0.0, max_value=0.20, value=0.02, step=0.005)
    monte_carlo = st.toggle("Monte Carlo noise (battery + thrust)", value=False)
    mc_runs = st.slider("Monte Carlo runs", min_value=5, max_value=60, value=20, step=5, disabled=not monte_carlo)
    mc_noise_pct = st.slider("Noise level (%)", min_value=0.0, max_value=8.0, value=2.0, step=0.5, disabled=not monte_carlo)

    st.write("")
    run_clicked = st.button("▶ Run Simulation", use_container_width=True)
    random_clicked = st.button("🎲 Random Design", use_container_width=True)
    optimize_clicked = st.button("🧠 Auto Optimize Best Design", use_container_width=True, type="primary")

base_params = DroneParams(
    mass_kg=float(mass_kg),
    payload_kg=float(payload_kg),
    gravity=float(gravity),
    voltage_v=float(voltage_v),
    capacity_mah=int(capacity_mah),
    battery_efficiency_pct=float(efficiency_pct),
    motor_k=float(motor_k),
    max_rpm=int(max_rpm),
    throttle_pct=float(throttle_pct),
    prop_size=prop_size,  # type: ignore[arg-type]
    dt=float(dt),
    max_time_s=float(max_time_s),
    air_drag_coeff=float(air_drag_coeff),
)

if random_clicked:
    rp = _random_params(base_params, seed=int(time.time()) % 10_000_000)
    # Streamlit doesn't let us directly set slider values without rerun; we instead simulate immediately.
    st.session_state.last_result = simulate_drone(rp)
    st.toast("Random design simulated.", icon="🧪")

if run_clicked:
    st.session_state.last_result = simulate_drone(base_params)


def _simulate_with_noise(p: DroneParams, seed: int, noise_pct: float) -> SimResult:
    rng = random.Random(seed)
    # Inject small multiplicative noise into motor_k and battery efficiency to emulate uncertainty.
    nk = _clamp(1.0 + rng.uniform(-noise_pct, noise_pct) / 100.0, 0.75, 1.25)
    ne = _clamp(1.0 + rng.uniform(-noise_pct, noise_pct) / 100.0, 0.75, 1.25)
    pn = DroneParams(
        **{
            **asdict(p),
            "motor_k": float(p.motor_k) * nk,
            "battery_efficiency_pct": float(_clamp(p.battery_efficiency_pct * ne, 50.0, 100.0)),
        }
    )
    return simulate_drone(pn)


left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown("### Live Simulation")
    live_panel = st.container(border=False)

with right:
    st.markdown("### Design Vault + Tools")
    tools_panel = st.container(border=False)


def _render_kpis(res: SimResult):
    s = res.summary
    stable = "Stable" if s["stable"] else "Unstable"
    stable_foot = s["stability_note"]
    twr = s["peak_twr"]

    cards = "".join(
        [
            _kpi_card("Flight Time", _fmt_s(s["flight_time_s"]), f"Stop: {s['stop_reason']}"),
            _kpi_card("Max Altitude", _fmt_m(s["max_altitude_m"]), f"Peak TWR: {_fmt_num(twr)}"),
            _kpi_card("Final Battery", _fmt_pct(s["final_battery_pct"]), f"Voltage: {res.params.voltage_v:.1f}V"),
            _kpi_card("Peak Velocity", f"{s['peak_velocity_mps']:.1f} m/s", f"dt={res.params.dt:.2f}s"),
            _kpi_card("Stability", stable, stable_foot),
            _kpi_card("Design Score", f"{s['score']:.1f}", _design_label(res.params)),
        ]
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def _render_downloads(res: SimResult):
    csv = res.series.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download time-series CSV",
        data=csv,
        file_name="drone_sim_timeseries.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_design_save(res: SimResult):
    c1, c2, c3 = st.columns(3)
    if c1.button("💾 Save as Design A", use_container_width=True):
        st.session_state.saved_designs["A"] = res
        st.toast("Saved to Design A.", icon="💾")
    if c2.button("💾 Save as Design B", use_container_width=True):
        st.session_state.saved_designs["B"] = res
        st.toast("Saved to Design B.", icon="💾")
    if c3.button("💾 Save as Design C", use_container_width=True):
        st.session_state.saved_designs["C"] = res
        st.toast("Saved to Design C.", icon="💾")


def _saved_designs_list() -> List[Tuple[str, SimResult]]:
    out: List[Tuple[str, SimResult]] = []
    for k in ["A", "B", "C"]:
        v = st.session_state.saved_designs.get(k)
        if v is not None:
            out.append((f"Design {k}", v))
    return out


with left:
    res: Optional[SimResult] = st.session_state.last_result
    if res is None:
        st.info("Run a simulation from the sidebar to populate KPIs and charts.")
    else:
        with st.container():
            _render_kpis(res)
        st.write("")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Height vs Time", "Velocity vs Time", "Battery vs Time", "Thrust vs Time", "Multi-axis Overview"]
        )
        designs_single = [("Current", res)]
        with tab1:
            st.plotly_chart(make_timeseries_chart(designs_single, "height_m", "Height (m)"), use_container_width=True)
        with tab2:
            st.plotly_chart(make_timeseries_chart(designs_single, "velocity_mps", "Velocity (m/s)"), use_container_width=True)
        with tab3:
            st.plotly_chart(make_timeseries_chart(designs_single, "battery_pct", "Battery Remaining (%)"), use_container_width=True)
        with tab4:
            st.plotly_chart(make_timeseries_chart(designs_single, "thrust_n", "Thrust (N)", secondary_y=("twr", "Thrust-to-Weight")), use_container_width=True)
        with tab5:
            st.plotly_chart(make_overview_chart(designs_single), use_container_width=True)

        st.write("")
        cols = st.columns([0.62, 0.38], gap="large")
        with cols[0]:
            _render_downloads(res)
        with cols[1]:
            st.metric("Thrust-to-Weight (peak)", f"{res.summary['peak_twr']:.2f}")

        st.write("")

        if monte_carlo:
            st.markdown("### Monte Carlo Robustness")
            with st.container(border=True):
                prog = st.progress(0, text="Running Monte Carlo simulations…")
                rows = []
                for i in range(mc_runs):
                    prog.progress(int((i + 1) / mc_runs * 100), text=f"Running Monte Carlo simulations… {i+1}/{mc_runs}")
                    r = _simulate_with_noise(res.params, seed=1000 + i, noise_pct=float(mc_noise_pct))
                    rows.append({**r.summary, **{"run": i + 1}})
                prog.empty()
                mc = pd.DataFrame(rows)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Flight time (p50)", _fmt_s(float(mc["flight_time_s"].median())))
                c2.metric("Flight time (p10)", _fmt_s(float(mc["flight_time_s"].quantile(0.10))))
                c3.metric("Max altitude (p50)", _fmt_m(float(mc["max_altitude_m"].median())))
                c4.metric("Stable rate", _fmt_pct(float(mc["stable"].mean() * 100.0)))
                fig = px.histogram(mc, x="flight_time_s", nbins=18, title="Flight Time Distribution (Monte Carlo)")
                fig.update_layout(**_chart_template())
                st.plotly_chart(fig, use_container_width=True)


with right:
    if st.session_state.last_result is None:
        st.warning("No live simulation yet. Run one to enable saving and comparisons.")
    else:
        st.markdown("#### Save / Compare")
        _render_design_save(st.session_state.last_result)

    saved = _saved_designs_list()
    if not saved:
        st.caption("Saved designs will appear here (A/B/C).")
    else:
        st.write("")
        with st.container(border=True):
            st.markdown("#### Saved Design Summary")
            rows = []
            for name, r in saved:
                rows.append(
                    {
                        "Design": name,
                        "Stable": r.summary["stable"],
                        "Stop": r.summary["stop_reason"],
                        "Flight Time (s)": round(r.summary["flight_time_s"], 2),
                        "Max Altitude (m)": round(r.summary["max_altitude_m"], 2),
                        "Final Battery (%)": round(r.summary["final_battery_pct"], 2),
                        "Peak Velocity (m/s)": round(r.summary["peak_velocity_mps"], 2),
                        "Peak TWR": round(r.summary["peak_twr"], 3),
                        "Score": round(r.summary["score"], 2),
                        "Label": _design_label(r.params),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.write("")
        with st.container(border=True):
            st.markdown("#### Overlay Charts (Compare)")
            compare_tabs = st.tabs(["Height", "Velocity", "Battery", "Thrust", "Overview"])
            with compare_tabs[0]:
                st.plotly_chart(make_timeseries_chart(saved, "height_m", "Height (m)"), use_container_width=True)
            with compare_tabs[1]:
                st.plotly_chart(make_timeseries_chart(saved, "velocity_mps", "Velocity (m/s)"), use_container_width=True)
            with compare_tabs[2]:
                st.plotly_chart(make_timeseries_chart(saved, "battery_pct", "Battery Remaining (%)"), use_container_width=True)
            with compare_tabs[3]:
                st.plotly_chart(make_timeseries_chart(saved, "thrust_n", "Thrust (N)", secondary_y=("twr", "Thrust-to-Weight")), use_container_width=True)
            with compare_tabs[4]:
                st.plotly_chart(make_overview_chart(saved), use_container_width=True)

    st.write("")
    with st.container(border=True):
        st.markdown("#### Auto Optimization")
        st.caption("Searches a compact design space to maximize a multi-objective score (flight time + altitude + battery, penalizing instability/crash and weight).")

        opt_col1, opt_col2 = st.columns([0.55, 0.45], gap="medium")
        with opt_col1:
            max_evals = st.slider("Max evaluations", 80, 700, 320, 20)
            throttle_policy = st.selectbox("Throttle strategy", ["fixed", "sweep"], index=0)
        with opt_col2:
            include_props = st.multiselect("Prop sizes", ["Small", "Medium", "Large"], default=["Small", "Medium", "Large"])
            stable_only = st.toggle("Show stable-only leaderboard", value=True)

        if optimize_clicked:
            progress = st.progress(0, text="Optimizing designs…")
            status = st.empty()

            def _progress_cb(i: int, n: int):
                progress.progress(int(i / n * 100), text=f"Optimizing designs… {i}/{n}")
                if i % 25 == 0 or i == n:
                    status.caption(f"Evaluated {i}/{n} configurations.")

            df_opt = optimize_design(
                base_params,
                prop_sizes=[p for p in include_props if p in ["Small", "Medium", "Large"]] or ["Small", "Medium", "Large"],  # type: ignore[list-item]
                throttle_policy=throttle_policy,  # type: ignore[arg-type]
                max_evals=int(max_evals),
                progress_cb=_progress_cb,
            )
            progress.empty()
            status.empty()
            st.session_state.optimizer_results = df_opt

            if not df_opt.empty:
                best = df_opt.iloc[0].to_dict()
                st.session_state.optimizer_best_params = best
                st.toast("Optimization complete. Leaderboard updated.", icon="🏁")

        df_opt = st.session_state.optimizer_results
        if df_opt is not None and isinstance(df_opt, pd.DataFrame) and not df_opt.empty:
            view = df_opt.copy()
            if stable_only:
                view = view[view["stable"] == True]  # noqa: E712
            st.markdown("##### Top 10 Configurations")
            st.dataframe(view.head(10), use_container_width=True, hide_index=True)

            st.write("")
            c1, c2 = st.columns([0.55, 0.45], gap="large")
            with c1:
                pareto = pareto_frontier(df_opt, x_col="weight_kg", y_col="flight_time_s", minimize_x=True, maximize_y=True)
                fig = px.scatter(
                    df_opt,
                    x="weight_kg",
                    y="flight_time_s",
                    color="stable",
                    hover_data=["prop_size", "battery_mah", "motor_k", "throttle_pct", "max_altitude_m", "score"],
                    title="Design Space: Flight Time vs Weight",
                    opacity=0.75,
                )
                fig.update_layout(**_chart_template())
                if not pareto.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=pareto["weight_kg"],
                            y=pareto["flight_time_s"],
                            mode="markers",
                            name="Pareto Frontier",
                            marker=dict(size=10, color="#EF4444", symbol="diamond"),
                        )
                    )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                top = df_opt.head(40).copy()
                fig2 = px.parallel_coordinates(
                    top,
                    color="score",
                    dimensions=["weight_kg", "battery_mah", "motor_k", "throttle_pct", "max_altitude_m", "flight_time_s", "score"],
                    title="Top Candidates: Parameter Tradeoffs (Parallel Coordinates)",
                    color_continuous_scale=px.colors.sequential.Viridis,
                )
                fig2.update_layout(**_chart_template(), height=420)
                st.plotly_chart(fig2, use_container_width=True)

            csv = df_opt.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download optimization leaderboard CSV",
                data=csv,
                file_name="drone_optimizer_leaderboard.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Run Auto Optimize to populate the leaderboard.")

    st.write("")
    with st.expander("📊 Research Analytics", expanded=False):
        st.markdown("#### Research Analytics")
        st.caption("Quick analytics to support design research: correlation, sensitivity ranking, and Pareto frontier insights.")

        # Build a dataset from saved designs + optimizer leaderboard (if any).
        rows = []
        for name, r in _saved_designs_list():
            rows.append(
                {
                    "source": name,
                    **{
                        "weight_kg": r.params.mass_kg,
                        "payload_kg": r.params.payload_kg,
                        "voltage_v": r.params.voltage_v,
                        "battery_mah": r.params.capacity_mah,
                        "efficiency_pct": r.params.battery_efficiency_pct,
                        "motor_k": r.params.motor_k,
                        "max_rpm": r.params.max_rpm,
                        "throttle_pct": r.params.throttle_pct,
                        "prop_size": r.params.prop_size,
                    },
                    **{
                        "flight_time_s": r.summary["flight_time_s"],
                        "max_altitude_m": r.summary["max_altitude_m"],
                        "final_battery_pct": r.summary["final_battery_pct"],
                        "peak_velocity_mps": r.summary["peak_velocity_mps"],
                        "peak_twr": r.summary["peak_twr"],
                        "stable": int(r.summary["stable"]),
                        "score": r.summary["score"],
                    },
                }
            )

        df_opt = st.session_state.optimizer_results
        if df_opt is not None and isinstance(df_opt, pd.DataFrame) and not df_opt.empty:
            df_o = df_opt.head(250).copy()
            df_o["source"] = "Optimizer"
            rows.extend(df_o.to_dict(orient="records"))

        research_df = pd.DataFrame(rows)
        if research_df.empty:
            st.info("Save a few designs and/or run optimization to unlock research analytics.")
        else:
            c1, c2 = st.columns([0.55, 0.45], gap="large")
            with c1:
                numeric_cols = [
                    "weight_kg",
                    "payload_kg",
                    "voltage_v",
                    "battery_mah",
                    "efficiency_pct",
                    "motor_k",
                    "max_rpm",
                    "throttle_pct",
                    "flight_time_s",
                    "max_altitude_m",
                    "final_battery_pct",
                    "peak_velocity_mps",
                    "peak_twr",
                    "stable",
                    "score",
                ]
                corr = research_df[numeric_cols].corr(numeric_only=True).round(3)
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Heatmap (Parameters + Outcomes)",
                )
                fig.update_layout(**_chart_template(), height=520)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("##### Sensitivity Ranking (|corr| with Flight Time)")
                sens = (
                    research_df[numeric_cols]
                    .corr(numeric_only=True)["flight_time_s"]
                    .drop(labels=["flight_time_s"])
                    .abs()
                    .sort_values(ascending=False)
                    .rename("abs_corr")
                )
                sens_df = sens.reset_index().rename(columns={"index": "feature"})
                fig = px.bar(sens_df.head(10), x="abs_corr", y="feature", orientation="h", title="Top Drivers of Flight Time (heuristic)")
                fig.update_layout(**_chart_template(), height=420)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### Pareto Frontier (Flight Time vs Weight)")
                pareto = pareto_frontier(research_df, x_col="weight_kg", y_col="flight_time_s", minimize_x=True, maximize_y=True)
                fig2 = px.scatter(
                    research_df,
                    x="weight_kg",
                    y="flight_time_s",
                    color="source",
                    hover_data=["battery_mah", "motor_k", "prop_size", "throttle_pct", "max_altitude_m", "score"],
                    title="Research Set: Flight Time vs Weight",
                    opacity=0.75,
                )
                if not pareto.empty:
                    fig2.add_trace(go.Scatter(x=pareto["weight_kg"], y=pareto["flight_time_s"], mode="markers", name="Pareto", marker=dict(size=10, color="#EF4444")))
                fig2.update_layout(**_chart_template(), height=420)
                st.plotly_chart(fig2, use_container_width=True)

st.write("")
st.caption("Tip: Save A/B/C, then compare overlays. Run Auto Optimize to explore tradeoffs and generate a Pareto frontier.")
