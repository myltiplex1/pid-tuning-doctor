"""PID Tuning Doctor -- dataset generator (Capstone Stage 2).

Simulates closed-loop PID step responses on several plants, injects one
tuning fault on purpose, measures the response, and writes text
input/output pairs for fine-tuning Gemma with LoRA.

Labels are KNOWN INJECTED FAULTS that passed a symptom check, not proven
diagnoses: only the named gain is changed (see `inject`), and the run is kept
only if it breaks the stated requirement(s) tied to that fault and no
requirement that points to a different fault (see `symptom_ok`). Different
gain settings can still produce similar symptoms.

Plants
------
Training family (randomised around realistic values):
  dc_motor      QUBE-Servo 3 disc, position control. Calibrated on QLabs
                (Enoch's Lab 01-04): K = 23.1 rad/s/V, tau = 0.092 s,
                Rm 7.5, kt = km 0.0422, Jeq 2.19e-5, b 5e-6, Tc 4.9e-5,
                2048-count encoder, +/-10 V.
  first_order   K/(tau s + 1), e.g. a speed loop or small thermal process.
  second_order  K wn^2/(s^2 + 2 zeta wn s + wn^2), e.g. mass-spring-damper.
Unseen test plants:
  tank          nonlinear single tank (Torricelli), EEE576 project:
                Ac 0.5 m^2, h_op 1.5 m, q_op 0.0101 m^3/s, tau 148.7 s.
  heater        first order + dead time (FOPDT).
  pendulum_arm  QUBE-Servo 3 rotary arm with the pendulum hanging DOWN,
                nonlinear Furuta model (arm 0.095 kg / 0.085 m,
                pendulum 0.024 kg / 0.129 m, Dr 1e-3, Dp 5e-5).

Usage
-----
    python generate_dataset.py               # default sizes, seed 0
    python generate_dataset.py --seed 1 --out data/

Outputs (in --out):
    pid_doctor_dataset.json   one list of records, all splits
    train.json / val.json / test_A.json / test_B.json
    summary.txt               counts per split x class x plant
"""

import argparse
import json
import math
import os
from collections import Counter

import numpy as np

from rule_baseline import parse as parse_text   # same parser the baselines and checker use


def displayed_violations(text):
    """Requirements broken according to the numbers as printed in `text`."""
    req, obs = parse_text(text)
    v = set()
    if obs["overshoot"] > req["os_max"]:
        v.add("overshoot")
    if obs["osc"] > req["osc_max"]:
        v.add("oscillation")
    if obs["no_settle"] or obs["settle"] > req["ts_max"]:
        v.add("settling")
    if obs["sse"] > req["sse_max"]:
        v.add("end_error")
    if obs["ripple"] > req["rip_max"]:
        v.add("ripple")
    return v

# ----------------------------------------------------------------- classes
CLASSES = [
    "well_tuned",
    "kp_too_high",
    "gain_too_low",
    "ki_too_low",
    "ki_too_high",
    "kd_too_high",
    "actuator_saturation",
]

CLASS_TEXT = {
    "well_tuned": "Well tuned",
    "kp_too_high": "Kp too high (oscillatory)",
    "gain_too_low": "Overall gain too low (sluggish)",
    "ki_too_low": "Ki too low (remaining error)",
    "ki_too_high": "Ki too high (integral overshoot)",
    "kd_too_high": "Kd too high (noise amplification)",
    "actuator_saturation": "Actuator saturation",
}

FIX_TEXT = {
    "well_tuned": "No change needed. Keep the current gains.",
    "kp_too_high": "Reduce Kp by about half, or increase Kd to add damping.",
    "gain_too_low": ("Raise Kp, Ki and Kd together by the same factor (try about 2 times), keeping their ratios, "
                     "and re-test against the settling-time target."),
    "ki_too_low": "Increase Ki so the integral removes the remaining error.",
    "ki_too_high": "Reduce Ki (try a third of the current value); add anti-windup if the actuator saturates.",
    "kd_too_high": "Reduce Kd or strengthen the derivative filtering, then re-test.",
    "actuator_saturation": ("The actuator can hold this target but spends long periods at its limit. Use a smaller "
                            "setpoint step or lower gains so it stays off the limit, and add anti-windup to the "
                            "integrator. Lower gains cannot add actuator capacity: if the target cannot be held "
                            "at steady state, a larger actuator or a smaller target is needed."),
}

SAFETY = "This is a learning aid; check with your instructor if unsure."


# ===================================================================== plants
class Plant:
    """Base class. Subclasses implement reset(), step(u, dt) -> y."""

    name = "plant"
    description = ""
    split_group = "train"

    def __init__(self, rng):
        self.rng = rng


class DCMotor(Plant):
    name = "dc_motor"
    description = "DC motor position control (QUBE-Servo 3 with inertia disc)"
    split_group = "train"

    def __init__(self, rng):
        super().__init__(rng)
        self.Rm = 7.5 * rng.uniform(0.9, 1.1)
        self.k = 0.0422 * rng.uniform(0.95, 1.05)           # kt = km
        self.J = 2.19e-5 * rng.uniform(0.7, 1.4)             # covers manual and QLabs
        self.b = 5e-6 * rng.uniform(0.5, 2.0)
        self.Tc = 4.9e-5 * rng.uniform(0.5, 3.0)
        self.u_max = 10.0
        self.Ts = 0.002
        self.enc = 2 * math.pi / 2048
        self.step_size = rng.uniform(0.5, 2.0)              # rad
        self.dist = rng.uniform(0.15, 0.5) * rng.choice([-1, 1])   # V bias (deadband-like)
        self.K = 1.0 / self.k                                 # rad/s/V
        self.tau = self.Rm * self.J / (self.k * self.k)
        self.delay = self.Ts                                  # one-sample compute delay
        self.units = ("rad", "V")

    def reset(self):
        self.th = 0.0
        self.w = 0.0

    def step(self, u, dt):
        n = 5
        h = dt / n
        v = u + self.dist
        for _ in range(n):
            i = (v - self.k * self.w) / self.Rm
            T = self.k * i - self.b * self.w
            if abs(self.w) > 1e-6:
                T -= self.Tc * np.sign(self.w)
            elif abs(T) <= self.Tc:
                T = 0.0
            self.w += h * T / self.J
            self.th += h * self.w
        return math.floor(self.th / self.enc) * self.enc     # encoder quantised

    def u_required(self):
        # type-1 plant: holding any position only needs to cancel the bias
        return -self.dist

    def nominal_gains(self):
        # PD pole placement on K/(s(tau s+1)), plus a small I for the bias
        wn = self.rng.uniform(2.5, 5.0) / self.tau
        zeta = self.rng.uniform(0.75, 0.95)
        kp = wn ** 2 * self.tau / self.K
        kd = max((2 * zeta * wn * self.tau - 1) / self.K, 0.0)
        ki = kp * wn / self.rng.uniform(6, 10)
        return kp, ki, kd, wn


class FirstOrder(Plant):
    name = "first_order"
    description = "a first-order process (like a motor speed loop or small thermal system)"
    split_group = "train"

    def __init__(self, rng):
        super().__init__(rng)
        self.K = rng.uniform(0.5, 5.0)
        self.tau = 10 ** rng.uniform(-1.3, 0.7)              # 0.05 .. 5 s
        self.Ts = self.tau / rng.uniform(40, 80)
        self.step_size = rng.uniform(0.5, 2.0)
        self.u_ss = self.step_size / self.K
        self.u_max = self.u_ss * rng.uniform(3.0, 5.0)
        self.dist = self.u_ss * rng.uniform(0.1, 0.3) * rng.choice([-1, 1])
        self.noise = 0.003
        self.delay = self.tau * rng.uniform(0.04, 0.10)      # actuator/sensor lag
        self.units = ("units", "units")

    def reset(self):
        self.x = 0.0

    def step(self, u, dt):
        a = math.exp(-dt / self.tau)
        self.x = a * self.x + (1 - a) * self.K * (u + self.dist)
        return self.x

    def u_required(self):
        # steady state: K * (u + dist) = step
        return self.step_size / self.K - self.dist

    def nominal_gains(self):
        wn = self.rng.uniform(2.0, 4.0) / self.tau
        zeta = self.rng.uniform(0.75, 0.95)
        kp = max((2 * zeta * wn * self.tau - 1) / self.K, 0.05 / self.K)
        ki = wn ** 2 * self.tau / self.K
        kd = 0.0
        return kp, ki, kd, wn


class SecondOrder(Plant):
    name = "second_order"
    description = "a second-order system (like a mass-spring-damper)"
    split_group = "train"

    def __init__(self, rng):
        super().__init__(rng)
        self.K = rng.uniform(0.5, 3.0)
        self.wp = 10 ** rng.uniform(0.0, 1.3)                 # 1 .. 20 rad/s
        self.zp = rng.uniform(0.3, 1.5)
        self.Ts = 1.0 / (self.wp * rng.uniform(40, 80))
        self.step_size = rng.uniform(0.5, 2.0)
        self.u_ss = self.step_size / self.K
        self.u_max = self.u_ss * rng.uniform(3.0, 5.0)
        self.dist = self.u_ss * rng.uniform(0.1, 0.3) * rng.choice([-1, 1])
        self.noise = 0.003
        self.delay = rng.uniform(0.04, 0.10) / self.wp       # actuator/sensor lag
        self.units = ("units", "units")

    def reset(self):
        self.x = 0.0
        self.v = 0.0

    def step(self, u, dt):
        n = 5
        h = dt / n
        for _ in range(n):
            acc = self.wp ** 2 * (self.K * (u + self.dist) - self.x) - 2 * self.zp * self.wp * self.v
            self.v += h * acc
            self.x += h * self.v
        return self.x

    def u_required(self):
        # steady state: K * (u + dist) = step
        return self.step_size / self.K - self.dist

    def nominal_gains(self):
        # IMC-PID for a second-order plant; lambda sets closed-loop speed
        lam = self.rng.uniform(0.4, 1.0) / self.wp
        kc = 2 * self.zp / (self.K * lam * self.wp)
        ti = 2 * self.zp / self.wp
        td = 1 / (2 * self.zp * self.wp)
        return kc, kc / ti, kc * td, 1 / lam


class Tank(Plant):
    name = "tank"
    description = "water tank level control (nonlinear, Torricelli outflow)"
    split_group = "test"

    def __init__(self, rng):
        super().__init__(rng)
        self.Ac = 0.5 * rng.uniform(0.8, 1.2)
        self.h_op = 1.5
        self.q_op = 0.0101
        self.c = self.q_op / math.sqrt(self.h_op)            # Cd*Ao*sqrt(2g)
        self.Ts = 1.0
        self.step_size = rng.uniform(0.15, 0.4)              # m above h_op
        self.u_max = self.q_op * rng.uniform(1.0, 1.6)       # extra inflow above q_op
        self.dist = self.q_op * rng.uniform(0.05, 0.15) * rng.choice([-1, 1])
        self.noise = 0.003
        self.K = 2 * self.h_op / self.q_op                    # 297 m/(m^3/s)
        self.tau = 2 * self.Ac * self.h_op / self.q_op        # ~149 s
        self.delay = rng.uniform(5, 15)                       # valve + transport lag [s]
        self.units = ("m", "m^3/s")

    def reset(self):
        self.h = self.h_op

    def step(self, u, dt):
        n = 5
        h = dt / n
        qin = self.q_op + u + self.dist
        qin = max(qin, 0.0)
        for _ in range(n):
            self.h += h * (qin - self.c * math.sqrt(max(self.h, 0.01))) / self.Ac
            self.h = max(self.h, 0.01)
        return self.h - self.h_op

    def u_required(self):
        # extra inflow above q_op needed to hold h_op + step (nonlinear)
        return self.c * math.sqrt(self.h_op + self.step_size) - self.q_op - self.dist

    def nominal_gains(self):
        wn = self.rng.uniform(2.0, 4.0) / self.tau
        zeta = self.rng.uniform(0.75, 0.95)
        kp = (2 * zeta * wn * self.tau - 1) / self.K
        ki = wn ** 2 * self.tau / self.K
        return kp, ki, 0.0, wn


class Heater(Plant):
    name = "heater"
    description = "heater temperature control (slow, with dead time)"
    split_group = "test"

    def __init__(self, rng):
        super().__init__(rng)
        self.K = rng.uniform(0.3, 1.0)                        # degC per % power
        self.tau = rng.uniform(60, 300)
        self.L = self.tau * rng.uniform(0.05, 0.15)
        self.Ts = 1.0
        self.step_size = rng.uniform(5, 15)                   # degC
        self.u_ss = self.step_size / self.K
        self.u_max = min(100.0, self.u_ss * rng.uniform(3.0, 5.0))
        self.dist = self.u_ss * rng.uniform(0.1, 0.3) * rng.choice([-1, 1])
        self.noise = 0.003
        self.units = ("degC", "% power")

    def reset(self):
        self.x = 0.0
        self.buf = [0.0] * max(1, int(round(self.L / self.Ts)))

    def step(self, u, dt):
        self.buf.append(u + self.dist)
        ud = self.buf.pop(0)
        a = math.exp(-dt / self.tau)
        self.x = a * self.x + (1 - a) * self.K * ud
        return self.x

    def u_required(self):
        # steady state: K * (u + dist) = step
        return self.step_size / self.K - self.dist

    def nominal_gains(self):
        lam = self.L * self.rng.uniform(1.5, 2.5)              # SIMC, slightly detuned
        kp = self.tau / (self.K * (lam + self.L))
        ti = min(self.tau, 4 * (lam + self.L))
        return kp, kp / ti, 0.0, 1 / (lam + self.L)


class PendulumArm(Plant):
    name = "pendulum_arm"
    description = "rotary arm position control with a hanging pendulum (QUBE-Servo 3)"
    split_group = "test"

    def __init__(self, rng):
        super().__init__(rng)
        s = lambda: rng.uniform(0.9, 1.1)
        self.mr, self.Lr = 0.095 * s(), 0.085 * s()
        self.mp, self.Lp = 0.024 * s(), 0.129 * s()
        self.Jr = self.mr * self.Lr ** 2 / 12
        self.Jp = self.mp * self.Lp ** 2 / 12
        self.Jm = 1.4e-6 + 0.6e-6
        self.Dr = 1e-3 * rng.uniform(0.5, 1.5)
        self.Dp = 5e-5 * rng.uniform(0.5, 2.0)
        self.Rm, self.k = 7.5, 0.0422
        self.g = 9.81
        self.u_max = 10.0
        self.Ts = 0.002
        self.enc = 2 * math.pi / 2048
        self.step_size = rng.uniform(0.3, 1.0)               # rad (arm)
        self.dist = rng.uniform(0.15, 0.5) * rng.choice([-1, 1])
        Jeq = self.Jm + self.Jr + self.mp * self.Lr ** 2
        self.K = self.k / (self.k ** 2 + self.Rm * self.Dr)
        self.tau = self.Rm * Jeq / (self.k ** 2 + self.Rm * self.Dr)
        self.units = ("rad", "V")

    def reset(self):
        self.th = self.al = self.thd = self.ald = 0.0

    def step(self, u, dt):
        n = 10
        h = dt / n
        mp, Lr, Lp, g = self.mp, self.Lr, self.Lp, self.g
        v = u + self.dist
        for _ in range(n):
            ca, sa = math.cos(self.al), math.sin(self.al)
            tau_m = self.k * (v - self.k * self.thd) / self.Rm
            m11 = self.Jm + mp * Lr ** 2 + 0.25 * mp * Lp ** 2 * sa ** 2 + self.Jr
            m12 = -0.5 * mp * Lp * Lr * ca
            m22 = self.Jp + 0.25 * mp * Lp ** 2
            f1 = (tau_m - self.Dr * self.thd
                  - 0.5 * mp * Lp ** 2 * sa * ca * self.thd * self.ald
                  - 0.5 * mp * Lp * Lr * sa * self.ald ** 2)
            # alpha = 0 hanging down; gravity restores
            f2 = (-self.Dp * self.ald
                  + 0.25 * mp * Lp ** 2 * ca * sa * self.thd ** 2
                  - 0.5 * mp * Lp * g * sa)
            det = m11 * m22 - m12 * m12
            thdd = (m22 * f1 - m12 * f2) / det
            aldd = (m11 * f2 - m12 * f1) / det
            self.thd += h * thdd
            self.ald += h * aldd
            self.th += h * self.thd
            self.al += h * self.ald
        return math.floor(self.th / self.enc) * self.enc

    def u_required(self):
        return -self.dist

    def nominal_gains(self):
        wn = self.rng.uniform(1.5, 3.0) / self.tau
        zeta = self.rng.uniform(0.8, 0.95)
        kp = wn ** 2 * self.tau / self.K
        kd = max((2 * zeta * wn * self.tau - 1) / self.K, 0.0)
        ki = kp * wn / self.rng.uniform(6, 10)
        return kp, ki, kd, wn


PLANTS = {c.name: c for c in [DCMotor, FirstOrder, SecondOrder, Tank, Heater, PendulumArm]}
TRAIN_PLANTS = ["dc_motor", "first_order", "second_order"]
TEST_B_PLANTS = ["tank", "heater", "pendulum_arm"]


# ================================================================ simulation
def simulate(plant, kp, ki, kd, T, u_max, noise_rel, rng):
    """Discrete PID (derivative on measurement, filtered; no anti-windup)."""
    dt = plant.Ts
    n = int(math.ceil(T / dt - 1e-9)) + 1          # last sample at t >= T
    r = plant.step_size
    N = 0.2 / dt                                   # derivative filter, rad/s
    alpha = 1.0 / (1.0 + N * dt)
    plant.reset()
    y = 0.0
    integ = 0.0
    d_state = 0.0
    y_prev = 0.0
    sig = noise_rel * r
    nd = int(round(getattr(plant, "delay", 0.0) / dt))
    buf = [0.0] * nd
    ys, us = np.empty(n), np.empty(n)
    for k in range(n):
        ym = y + rng.normal(0, sig)
        e = r - ym
        integ += ki * e * dt
        d_raw = -(ym - y_prev) / dt
        d_state = alpha * d_state + (1 - alpha) * d_raw
        y_prev = ym
        u = kp * e + integ + kd * d_state
        u_sat = min(max(u, -u_max), u_max)
        ys[k] = ym
        us[k] = u_sat
        buf.append(u_sat)
        y = plant.step(buf.pop(0), dt)
    t = np.arange(n) * dt
    return t, ys, us


def metrics(t, y, u, r, u_max):
    """Step-response metrics a student would read off a plot."""
    n = len(y)
    # light smoothing so noise does not fake crossings
    w = max(1, n // 200)
    # edge padding so the first/last samples are not pulled toward zero
    ys = np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"),
                     np.ones(w) / w, mode="valid")
    tail = ys[int(0.85 * n):]
    y_final = float(np.mean(tail))
    overshoot = max(0.0, (float(np.max(ys)) - r) / r * 100)
    try:
        t10 = t[np.argmax(ys >= 0.1 * r)] if np.any(ys >= 0.1 * r) else None
        t90 = t[np.argmax(ys >= 0.9 * r)] if np.any(ys >= 0.9 * r) else None
        rise = (t90 - t10) if (t10 is not None and t90 is not None) else None
    except Exception:
        rise = None
    band = 0.02 * r
    outside = np.where(np.abs(ys - r) > band)[0]
    if len(outside) == 0:
        settle = 0.0
    elif outside[-1] >= n - 2:
        settle = None                            # did not settle
    else:
        settle = float(t[outside[-1] + 1])
    sse = abs(r - y_final) / r * 100
    # oscillations: sign changes of the error after the first crossing of r
    err = ys - r
    sgn = np.sign(err[np.abs(err) > 0.01 * r])
    crossings = int(np.sum(sgn[1:] != sgn[:-1]))
    oscillations = crossings // 2
    at_lim = np.abs(u) >= 0.995 * u_max
    sat = float(np.mean(at_lim) * 100)
    sat_time = float(np.sum(at_lim) * (t[1] - t[0]))
    u_peak = float(np.max(np.abs(u)))
    # control ripple: std of sample-to-sample change, last 40 %, vs limit
    du = np.diff(u[int(0.6 * n):])
    ripple = float(np.std(du) / u_max * 100)
    return dict(overshoot=overshoot, rise=rise, settle=settle, sse=sse,
                oscillations=oscillations, sat=sat, sat_time=sat_time, ripple=ripple, u_peak=u_peak,
                T=float(t[-1]))


def is_well_tuned(m, ripple_ref=None):
    """Used only to accept a plant's NOMINAL controller as a sensible baseline."""
    ok = (m["overshoot"] < 12 and m["settle"] is not None
          and m["sse"] < 2 and m["sat"] < 15 and m["oscillations"] <= 2)
    if ripple_ref is not None:
        ok = ok and m["ripple"] < 2.5 * max(ripple_ref, 1e-6)
    return ok


# ------------------------------------------------------------ requirements
# Every input states five requirements. A fault is only labelled when the
# response breaks the requirement(s) tied to that fault (its "signature"),
# so a student reading the text has the criteria used to call it faulty.
SSE_MAX = 2.0      # % error left at the end of the recording
OSC_MAX = 2        # oscillation cycles


def make_spec(m0, rng):
    """Design target derived from the nominal controller (which meets it)."""
    ts_max = float(f"{m0['settle'] * rng.uniform(1.5, 2.2):.2g}")
    os_max = 15.0 if m0["overshoot"] >= 8 else float(rng.choice([10.0, 15.0]))
    rip_max = float(f"{max(3 * m0['ripple'], 0.3):.1f}")
    return dict(os_max=os_max, ts_max=ts_max, sse_max=SSE_MAX, osc_max=OSC_MAX, rip_max=rip_max)


def violations(m, spec):
    """Which stated requirements does this response break?"""
    v = set()
    if m["overshoot"] > spec["os_max"]:
        v.add("overshoot")
    if m["oscillations"] > spec["osc_max"]:
        v.add("oscillation")
    if m["settle"] is None or m["settle"] > spec["ts_max"]:
        v.add("settling")
    if m["sse"] > spec["sse_max"]:
        v.add("end_error")
    if m["ripple"] > spec["rip_max"]:
        v.add("ripple")
    return v


def symptom_ok(fault, m, m0, spec, plant=None, u_lim=None):
    """Does the faulty run break the requirement(s) its fault should break,
    without breaking requirements that point to a different fault?"""
    v = violations(m, spec)
    rise0 = m0["rise"] or 1e-9
    # time at the limit judged against this plant's own settling deadline,
    # so fast (motor) and slow (heater) plants are compared fairly
    low_sat = m["sat_time"] < 0.1 * spec["ts_max"]
    if fault == "well_tuned":
        return not v and low_sat
    if fault == "kp_too_high":
        # too much overshoot or too many oscillations; extra noise allowed
        return bool(v & {"overshoot", "oscillation"}) and "end_error" not in v and low_sat
    if fault == "gain_too_low":
        # too slow, but no overshoot problem and no error left at the end
        return (v == {"settling"} and (m["rise"] is None or m["rise"] > 2.0 * rise0)
                and m["settle"] is not None and low_sat)
    if fault == "ki_too_low":
        # error left at the end; not an overshoot, oscillation or noise problem
        return "end_error" in v and not v & {"overshoot", "oscillation", "ripple"} and low_sat
    if fault == "ki_too_high":
        return "overshoot" in v and not v & {"end_error", "ripple"} and low_sat
    if fault == "kd_too_high":
        # ripple broken, optionally also too slow (extra derivative damping); nothing else
        return v in ({"ripple"}, {"ripple", "settling"}) and low_sat
    if fault == "actuator_saturation":
        # long time at the limit, the response is visibly affected, and the
        # actuator CAN hold the target at steady state (checked in inject)
        slowed = m["rise"] is None or m["rise"] > 1.5 * rise0
        return (m["sat_time"] > 0.2 * spec["ts_max"] and bool(v & {"settling", "overshoot"})
                and slowed and "ripple" not in v)
    raise ValueError(fault)


def inject(fault, kp, ki, kd, u_max, plant, rng, u_peak=None):
    """Return faulty (kp, ki, kd, u_lim). Only the NAMED gain is changed."""
    j = lambda: rng.uniform(0.85, 1.15)
    if fault == "well_tuned":
        return kp * j(), ki * j(), kd * j(), u_max
    if fault == "kp_too_high":
        return kp * rng.uniform(2.5, 8), ki, kd, u_max
    if fault == "gain_too_low":
        # overall gain Kc lowered: all three gains scaled by the SAME factor,
        # so the integral and derivative times (and the derivative filter) are unchanged
        f = rng.uniform(0.1, 0.45)
        return kp * f, ki * f, kd * f, u_max
    if fault == "ki_too_low":
        return kp, ki * rng.uniform(0.0, 0.03), kd, u_max
    if fault == "ki_too_high":
        return kp, ki * rng.uniform(4, 15), kd, u_max
    if fault == "kd_too_high":
        base = kd if kd > 0 else kp * plant.Ts * 5
        return kp, ki, base * rng.uniform(2, 10), u_max
    if fault == "actuator_saturation":
        # limit below the nominal peak (so it saturates) but at least 1.3x the
        # input needed to HOLD the target (so the target is reachable)
        need = abs(plant.u_required())
        lo, hi = 1.3 * need, 0.6 * (u_peak if u_peak else u_max)
        if lo >= hi:
            return None
        return kp, ki, kd, min(rng.uniform(lo, hi), u_max)
    raise ValueError(fault)


# ===================================================================== text
def fmt_t(x):
    if x is None:
        return None
    if x < 1:
        return f"{x * 1000:.0f} ms"
    if x < 100:
        return f"{x:.2f} s"
    return f"{x:.0f} s"


def fmt_g(x):
    if x == 0:
        return "0"
    if abs(x) >= 100 or abs(x) < 0.01:
        return f"{x:.3g}"
    return f"{x:.3f}".rstrip("0").rstrip(".")


# The observation part of every template starts with one of these markers;
# everything before it is the plant, requirements and gains. The baseline
# parser splits on them so a requirement is never read as a measurement.
OBS_MARKERS = ["Step response:", "The step response shows", "Observed:", "Measured step response:"]


def spec_text(spec, rng):
    os_, ts, rip = spec["os_max"], fmt_t(spec["ts_max"]), spec["rip_max"]
    return rng.choice([
        # "at most" = the boundary value itself is acceptable, matching the
        # strict ">" used for a violation in violations()
        f"overshoot at most {os_:.0f}%, settling time at most {ts}, at most {spec['osc_max']} oscillation cycles, "
        f"error at most {spec['sse_max']:.0f}% at the end, control ripple at most {rip:.1f}%",
        f"settle within {ts} with at most {os_:.0f}% overshoot, no more than {spec['osc_max']} oscillation cycles, "
        f"at most {spec['sse_max']:.0f}% error at the end and control ripple of at most {rip:.1f}%",
    ])


def make_input(plant, gains, m, rng, spec):
    kp, ki, kd = gains
    osc = m["oscillations"]
    cyc = "cycle" if osc == 1 else "cycles"
    if m["settle"] is not None:
        settle_p = rng.choice([f"settling time {fmt_t(m['settle'])}", f"2% settling time {fmt_t(m['settle'])}"])
    else:
        settle_p = f"did not settle within the {fmt_t(m['T'])} recording"
    if m["rise"] is not None:
        rise_p = rng.choice([f"rise time {fmt_t(m['rise'])}", f"10-90% rise time {fmt_t(m['rise'])}"])
    else:
        rise_p = "did not reach 90% of the target"
    parts = {
        "overshoot": rng.choice([f"overshoot {m['overshoot']:.0f}%", f"{m['overshoot']:.0f}% overshoot",
                                 f"peak overshoot of {m['overshoot']:.0f}%"]),
        "rise": rise_p,
        "settle": settle_p,
        "sse": rng.choice([f"error at the end of the recording {m['sse']:.1f}%",
                           f"{m['sse']:.1f}% error left at the end of the recording"]),
        "osc": ("no oscillation" if osc == 0 else
                rng.choice([f"{osc} oscillation {cyc}",
                            "oscillates once" if osc == 1 else f"oscillates {osc} times"])),
        "sat": ("actuator never at its limit" if m["sat_time"] == 0 else
                rng.choice([f"actuator saturated for {fmt_t(m['sat_time'])} ({m['sat']:.0f}% of the recording)",
                            f"control signal at its limit for {fmt_t(m['sat_time'])} ({m['sat']:.0f}% of the recording)"])),
        "ripple": rng.choice([f"control ripple {m['ripple']:.1f}%",
                              f"control-signal ripple {m['ripple']:.1f}% of full scale"]),
    }
    keys = list(parts)
    order = [keys[0], keys[1]] + list(rng.permutation(keys[2:]))
    rng.shuffle(order[:2])
    obs = ", ".join(parts[k] for k in order)
    gains_txt = f"Kp={fmt_g(kp)}, Ki={fmt_g(ki)}, Kd={fmt_g(kd)}"
    req = spec_text(spec, rng)
    templates = [
        f"Plant: {plant.description}.\nRequirements: {req}.\nGains: {gains_txt}.\nStep response: {obs}.",
        f"I am tuning a PID controller for {plant.description}. Requirements: {req}. Gains are {gains_txt}. "
        f"The step response shows {obs}. What is wrong?",
        f"System: {plant.description}\nRequirements: {req}\nPID gains: {gains_txt}\nObserved: {obs}\nDiagnose the tuning.",
        f"{plant.description[0].upper() + plant.description[1:]} under PID control ({gains_txt}). Requirements: {req}. "
        f"Measured step response: {obs}.",
    ]
    return templates[rng.integers(len(templates))]


def evidence(fault, m, spec):
    """Built ONLY from requirements this example actually breaks (or meets)."""
    v = violations(m, spec)
    ts = fmt_t(spec["ts_max"])
    bits = []
    if "overshoot" in v:
        bits.append(f"overshoot {m['overshoot']:.0f}% against a {spec['os_max']:.0f}% limit")
    if "oscillation" in v:
        bits.append(f"{m['oscillations']} oscillation cycles against a limit of {spec['osc_max']}")
    if "settling" in v:
        if m["settle"] is None:
            bits.append(f"not settled after the {fmt_t(m['T'])} recording against a {ts} target")
        else:
            bits.append(f"settles at {fmt_t(m['settle'])} against a {ts} target")
    if "end_error" in v:
        bits.append(f"{m['sse']:.1f}% error left at the end of the recording against a {spec['sse_max']:.0f}% limit")
    if "ripple" in v:
        bits.append(f"control ripple {m['ripple']:.1f}% against a {spec['rip_max']:.1f}% limit")
    if fault == "well_tuned":
        return (f"all requirements met: overshoot {m['overshoot']:.0f}%, settles at {fmt_t(m['settle'])}, "
                f"{m['sse']:.1f}% error at the end, ripple {m['ripple']:.1f}%")
    if fault == "gain_too_low":
        bits.append("slow rise with little overshoot")
    if fault == "actuator_saturation":
        bits.insert(0, f"actuator at its limit for {fmt_t(m['sat_time'])} ({m['sat']:.0f}% of the recording)")
    return "; ".join(bits)


def make_output(fault, m, spec):
    return (f"Diagnosis: {CLASS_TEXT[fault]}.\nEvidence: {evidence(fault, m, spec)}.\n"
            f"Fix: {FIX_TEXT[fault]}\n{SAFETY}")


# ================================================================== builder
def one_example(plant_name, fault, rng, max_tries=100):
    for _ in range(max_tries):
        plant = PLANTS[plant_name](rng)
        kp, ki, kd, wn = plant.nominal_gains()
        T0 = 30.0 / wn
        noise = getattr(plant, "noise", 0.0)
        t, y, u = simulate(plant, kp, ki, kd, T0, plant.u_max, noise, rng)
        m0 = metrics(t, y, u, plant.step_size, plant.u_max)
        if not is_well_tuned(m0):
            continue
        spec = make_spec(m0, rng)
        inj = inject(fault, kp, ki, kd, plant.u_max, plant, rng, m0["u_peak"])
        if inj is None:
            continue
        fkp, fki, fkd, fu = inj
        # record well past the deadline so "did not settle" / end error is meaningful
        T = max(T0, 3.0 * spec["ts_max"])
        t, y, u = simulate(plant, fkp, fki, fkd, T, fu, noise, rng)
        m = metrics(t, y, u, plant.step_size, fu)
        if not symptom_ok(fault, m, m0, spec):
            continue
        text = make_input(plant, (fkp, fki, fkd), m, rng, spec)
        # v1.1: the requirements broken according to the PRINTED numbers must
        # equal those according to the stored numbers; otherwise rounding has
        # hidden (or invented) a violation, so the example is rejected
        if displayed_violations(text) != violations(m, spec):
            continue
        return dict(
            input=text,
            output=make_output(fault, m, spec),
            spec=spec,
            label=fault,
            plant=plant_name,
            gains=dict(kp=fkp, ki=fki, kd=fkd),
            nominal_gains=dict(kp=kp, ki=ki, kd=kd),
            u_limit=fu,
            u_required=float(plant.u_required()),
            violations=sorted(violations(m, spec)),
            metrics={k: (None if v is None else float(v)) for k, v in m.items()},   # full precision
        )
    return None


def build(split, plants, per_class, rng, log):
    rows = []
    for fault in CLASSES:
        for i in range(per_class):
            p = plants[i % len(plants)]
            ex = one_example(p, fault, rng)
            if ex is None:
                # this plant could not show the fault cleanly: accept fewer
                # examples instead of forcing it (recorded in summary.txt)
                log[(split, fault, p)] += 1
                continue
            ex["split"] = split
            rows.append(ex)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_seed", type=int, default=None,
                    help="v1.1-r5: draw Test A/B from this seed instead of continuing --seed. "
                         "Train/val are unchanged (they are generated first, from --seed).")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=60, help="examples per class")
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--test_a", type=int, default=20)
    ap.add_argument("--test_b", type=int, default=15, help="per class, spread over 3 plants")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    fails = Counter()
    data = []
    data += build("train", TRAIN_PLANTS, args.train, rng, fails)
    data += build("val", TRAIN_PLANTS, args.val, rng, fails)
    if args.test_seed is not None:
        rng = np.random.default_rng(args.test_seed)   # fresh, never-seen test instances
    data += build("test_A", TRAIN_PLANTS, args.test_a, rng, fails)
    data += build("test_B", TEST_B_PLANTS, args.test_b, rng, fails)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pid_doctor_dataset.json"), "w") as f:
        json.dump(data, f, indent=1)
    for s in ["train", "val", "test_A", "test_B"]:
        with open(os.path.join(args.out, f"{s}.json"), "w") as f:
            json.dump([d for d in data if d["split"] == s], f, indent=1)

    counts = Counter((d["split"], d["label"], d["plant"]) for d in data)
    lines = [f"seed={args.seed}  test_seed={args.test_seed}  total={len(data)}", ""]
    for s in ["train", "val", "test_A", "test_B"]:
        lines.append(f"[{s}]  {sum(v for k, v in counts.items() if k[0] == s)} examples")
        for c in CLASSES:
            per = {k[2]: v for k, v in counts.items() if k[0] == s and k[1] == c}
            lines.append(f"  {c:22s} {sum(per.values()):4d}   {per}")
    if fails:
        lines += ["", "Shortfall: plant could not show the fault cleanly after 100 tries (example skipped):"]
        lines += [f"  {k}: {v}" for k, v in sorted(fails.items())]
    txt = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
