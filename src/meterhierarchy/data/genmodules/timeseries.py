"""Leaf time-series generation with device archetypes + shared exogenous.

Archetypes (distinct shapes so the GNN learns signatures):
- flat           : continuous baseload (always-on, little variance)
- thermostat     : fridge/HVAC, on/off cycling, optionally temperature-driven
- spiky          : stove/kettle/vacuum, mostly zero + rare high bursts
- day_only       : office/lighting, active 07-19 weekdays
- night_only     : EV charger / night storage, active 22-06 some nights
- household_mix  : generic household with seasonal sine + AR + peaks

Shared exogenous per root ("temperature", "activity") drives correlated
behaviour across all meters in the same tree - essential so a GNN sees
real parent-child synchronisation and not just independent noise summed up.
"""
from __future__ import annotations

import numpy as np


STEPS_PER_DAY = 96     # 15-min
STEPS_PER_WEEK = 96 * 7
STEPS_PER_YEAR = 96 * 365


# ---------- low-level primitives ----------


def _harmonic(t, period, phase, amp):
    return amp * np.sin(2 * np.pi * (t / period) + phase)


def ar1_noise(n, phi, sigma, rng):
    eps = rng.normal(0, sigma, size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def garch_noise(n, omega, alpha, beta, rng):
    sigma2 = np.zeros(n)
    x = np.zeros(n)
    sigma2[0] = omega / max(1e-9, 1 - alpha - beta)
    for i in range(1, n):
        sigma2[i] = omega + alpha * x[i - 1] ** 2 + beta * sigma2[i - 1]
        x[i] = rng.normal(0, np.sqrt(sigma2[i]))
    return x


def seasonal_component(n, base, amp_day, amp_week, amp_year, rng):
    t = np.arange(n, dtype=float)
    pd_ = rng.uniform(0, 2 * np.pi)
    pw = rng.uniform(0, 2 * np.pi)
    py = rng.uniform(0, 2 * np.pi)
    return (
        base
        + _harmonic(t, STEPS_PER_DAY, pd_, amp_day * base)
        + _harmonic(t, STEPS_PER_WEEK, pw, amp_week * base)
        + _harmonic(t, STEPS_PER_YEAR, py, amp_year * base)
    )


def _genpareto_rvs(c: float, scale: float, size: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Generalized Pareto Distribution samples (loc=0), numpy-only.

    Replaces ``scipy.stats.genpareto.rvs``. The GPD quantile function is
    closed-form, so we sample uniforms and apply the inverse CDF:
        c != 0 :  x = scale / c * ((1 - u) ** (-c) - 1)
        c == 0 :  x = -scale * log(1 - u)   (exponential limit)
    With shape c in [0.15, 0.35] (as used here) this is heavy-tailed and
    non-negative, matching the peak-magnitude distribution of the original.
    """
    u = rng.random(size)
    if abs(c) < 1e-12:
        return -scale * np.log1p(-u)
    return scale / c * (np.power(1.0 - u, -c) - 1.0)


# ---------- shared exogenous (per root) ----------


def build_shared_exogenous(n: int, rng: np.random.Generator) -> dict:
    """Return a dict of shared signals that multiple leaves consume with
    different sensitivities. All outputs are z-normalised to ~[-1, 1] range.
    """
    t = np.arange(n, dtype=float)

    # Annual temperature-like cycle: max in mid-winter (index 0 = Jan 1)
    annual = np.cos(2 * np.pi * t / STEPS_PER_YEAR)
    # Slow weather random walk
    rw = np.cumsum(rng.normal(0, 1.0, n))
    rw = (rw - rw.mean()) / (rw.std() + 1e-9) * 0.3
    temperature = annual + rw
    temperature = temperature / (np.abs(temperature).max() + 1e-9)

    # Daily human activity (peak noon), plus weekend modulation
    daily = np.sin(2 * np.pi * t / STEPS_PER_DAY - np.pi / 2)
    dow = ((t // STEPS_PER_DAY).astype(int)) % 7
    weekend_mask = np.where(dow >= 5, rng.uniform(0.5, 0.8), 1.0)
    activity = daily * weekend_mask

    # Slow "regime" switch (e.g. season/holiday activity level)
    regime = np.cos(2 * np.pi * t / (STEPS_PER_DAY * rng.uniform(20, 60)))

    return {
        "temperature": temperature.astype(float),
        "activity": activity.astype(float),
        "regime": regime.astype(float),
    }


# ---------- archetypes ----------


ARCHETYPE_NAMES = [
    "flat", "thermostat", "spiky", "day_only", "night_only", "household_mix",
]
# Sampling weights (tuned to look like a small building mix)
ARCHETYPE_WEIGHTS = np.array([0.15, 0.22, 0.15, 0.15, 0.08, 0.25])
ARCHETYPE_WEIGHTS = ARCHETYPE_WEIGHTS / ARCHETYPE_WEIGHTS.sum()


def _sample_base(rng):
    profile = rng.choice(["low", "mid", "high"], p=[0.45, 0.40, 0.15])
    if profile == "low":
        return rng.uniform(0.03, 0.2)
    if profile == "mid":
        return rng.uniform(0.2, 0.9)
    return rng.uniform(0.9, 3.0)


def _gen_flat(n, base, rng, shared):
    x = np.full(n, base)
    x += ar1_noise(n, 0.8, base * 0.05, rng)
    # Very small weekly effect
    t = np.arange(n, dtype=float)
    x += _harmonic(t, STEPS_PER_WEEK, rng.uniform(0, 2 * np.pi), base * 0.03)
    return np.clip(x, 0, None)


def _gen_thermostat(n, base, rng, shared):
    on_power = base * rng.uniform(3.0, 6.0)
    duty = rng.uniform(0.25, 0.55)
    period = int(rng.integers(2, 8))  # 2-8 steps = 30min-2h
    phase = int(rng.integers(0, period))
    cycle = (((np.arange(n) + phase) % period) < max(1, int(period * duty)))
    x = cycle.astype(float) * on_power

    # Temperature coupling (HVAC): scale by temperature signal
    is_hvac = rng.random() < 0.5
    if is_hvac and shared is not None:
        sens = rng.uniform(0.3, 0.8)
        # Higher consumption at temperature extremes
        mod = 1.0 + sens * np.abs(shared["temperature"])
        x = x * mod

    x += rng.normal(0, base * 0.1, n)
    # Random short outages (compressor off)
    n_outages = int(rng.integers(0, 5))
    for _ in range(n_outages):
        s = int(rng.integers(0, n))
        L = int(rng.integers(4, 48))
        x[s : s + L] = 0
    return np.clip(x, 0, None)


def _gen_spiky(n, base, rng, shared):
    x = np.full(n, base * 0.05)
    x += ar1_noise(n, 0.6, base * 0.02, rng)

    # Event bursts concentrated at meal times
    meal_hours = [7.5, 12.5, 18.5]
    days = n // STEPS_PER_DAY
    # Activity scales with shared activity signal
    act = shared["activity"] if shared is not None else np.zeros(n)
    for d in range(days):
        for h in meal_hours:
            if rng.random() > 0.5:
                continue
            jitter = rng.normal(0, 0.75)
            idx = d * STEPS_PER_DAY + int(np.clip((h + jitter) * 4, 0, STEPS_PER_DAY - 1))
            dur = int(rng.integers(1, 6))
            mag = base * rng.uniform(8, 25)
            # Amplify during high activity windows
            mag *= 1.0 + 0.5 * max(0.0, act[idx])
            x[idx : idx + dur] += mag
    return np.clip(x, 0, None)


def _gen_day_only(n, base, rng, shared):
    t = np.arange(n)
    hour = (t % STEPS_PER_DAY) / 4.0
    start = rng.uniform(6.5, 8.5)
    end = rng.uniform(17.5, 20.0)
    on_power = base * rng.uniform(3.0, 8.0)
    base_off = base * rng.uniform(0.02, 0.1)
    profile = np.where((hour >= start) & (hour < end), on_power, base_off)
    dow = (t // STEPS_PER_DAY) % 7
    weekend = dow >= 5
    profile = np.where(weekend, profile * rng.uniform(0.05, 0.3), profile)

    if shared is not None:
        # Follow the activity signal within the active window
        profile = profile * (1.0 + 0.25 * np.clip(shared["activity"], -1, 1))

    profile += rng.normal(0, base * 0.15, n)
    return np.clip(profile, 0, None)


def _gen_night_only(n, base, rng, shared):
    t = np.arange(n)
    hour = (t % STEPS_PER_DAY) / 4.0
    on_power = base * rng.uniform(5.0, 15.0)
    base_off = base * rng.uniform(0.01, 0.05)
    profile = np.where((hour >= 22) | (hour < 6), on_power, base_off)
    # Not every night active
    n_days = n // STEPS_PER_DAY
    active = rng.random(n_days) < 0.6
    for d in range(n_days):
        if not active[d]:
            profile[d * STEPS_PER_DAY : (d + 1) * STEPS_PER_DAY] *= 0.05
    profile += rng.normal(0, base * 0.15, n)
    return np.clip(profile, 0, None)


def _gen_household_mix(n, base, rng, shared):
    s = seasonal_component(
        n, base,
        amp_day=rng.uniform(0.2, 0.5),
        amp_week=rng.uniform(0.05, 0.2),
        amp_year=rng.uniform(0.1, 0.3),
        rng=rng,
    )
    ar = ar1_noise(n, rng.uniform(0.3, 0.8), base * rng.uniform(0.05, 0.15), rng)
    gh = garch_noise(n, (base * 0.02) ** 2,
                     rng.uniform(0.05, 0.15), rng.uniform(0.7, 0.85), rng)

    # Poisson peaks at meal times
    peaks = np.zeros(n)
    base_p = rng.uniform(0.5, 2.0) / STEPS_PER_DAY
    hours = (np.arange(n) % STEPS_PER_DAY) / 4.0
    mod = 1.0 + 1.5 * (np.exp(-((hours - 8) ** 2) / 4) +
                       np.exp(-((hours - 19) ** 2) / 4))
    p = np.clip(base_p * mod, 0, 1)
    events = rng.random(n) < p
    n_ev = int(events.sum())
    if n_ev > 0:
        # Generalized Pareto magnitudes (numpy inverse-CDF, see _genpareto_rvs)
        mags = _genpareto_rvs(
            rng.uniform(0.15, 0.35),
            scale=base * rng.uniform(0.3, 0.8),
            size=n_ev, rng=rng,
        )
        peaks[events] = mags * rng.uniform(0.5, 1.5)

    x = s + ar + gh + peaks

    # Temperature sensitivity (some households)
    if shared is not None and rng.random() < 0.6:
        sens = rng.uniform(0.05, 0.3)
        x = x * (1.0 + sens * shared["temperature"])

    return np.clip(x, 0, None)


_DISPATCH = {
    "flat": _gen_flat,
    "thermostat": _gen_thermostat,
    "spiky": _gen_spiky,
    "day_only": _gen_day_only,
    "night_only": _gen_night_only,
    "household_mix": _gen_household_mix,
}


def sample_archetype(rng: np.random.Generator) -> str:
    return str(rng.choice(ARCHETYPE_NAMES, p=ARCHETYPE_WEIGHTS))


def generate_leaf(n: int, rng: np.random.Generator,
                  shared: dict | None = None,
                  archetype: str | None = None) -> tuple[np.ndarray, str]:
    """Generate a leaf series. Returns (series, archetype_name)."""
    if archetype is None:
        archetype = sample_archetype(rng)
    base = _sample_base(rng)
    x = _DISPATCH[archetype](n, base, rng, shared)
    return x.astype(np.float64), archetype


def build_time_index(n: int, start: str = "2023-01-01",
                     freq: str = "15min") -> "pd.DatetimeIndex":
    import pandas as pd
    return pd.date_range(start=start, periods=n, freq=freq)
