"""Baselines for the PID Tuning Doctor.

Both read the SAME text input the model sees:
  * RULES: hand-written if-then rules that compare each measurement with the
    requirement stated in the same input.
  * TREE:  a small decision tree trained on the same parsed numbers
    (train split only).

The text is split at the observation marker first, so a requirement
(e.g. "less than 15% overshoot") is never read as a measurement.

    python rule_baseline.py data/val.json            # development
    python rule_baseline.py data/test_A.json ...     # final, run once (see PROTOCOL.md)
"""

import json
import re
import sys
from collections import Counter, defaultdict

CLASSES = ["well_tuned", "kp_too_high", "gain_too_low", "ki_too_low",
           "ki_too_high", "kd_too_high", "actuator_saturation"]

OBS_MARKERS = ["Step response:", "The step response shows", "Observed:", "Measured step response:"]


def split_text(text):
    """Return (requirements_and_gains_part, observation_part)."""
    for mk in OBS_MARKERS:
        i = text.find(mk)
        if i >= 0:
            return text[:i], text[i + len(mk):]
    raise ValueError("no observation marker in input")


def _num(pattern, t):
    m = re.search(pattern, t)
    return float(m.group(1)) if m else None


def _time(pattern, t):
    """A duration written as '840 ms' or '2.10 s' right after `pattern`."""
    m = re.search(pattern + r"\s*([0-9.]+)\s*(ms|s)\b", t)
    if not m:
        return None
    v = float(m.group(1))
    return v / 1000 if m.group(2) == "ms" else v


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def parse_requirements(t):
    t = t.lower()
    return dict(
        os_max=_first(_num(r"overshoot at most ([0-9.]+)%", t), _num(r"at most ([0-9.]+)% overshoot", t)),
        ts_max=_first(_time(r"settling time at most", t), _time(r"settle within", t)),
        osc_max=_first(_num(r"at most (\d+) oscillation", t), _num(r"no more than (\d+) oscillation", t)),
        sse_max=_first(_num(r"error at most ([0-9.]+)% at the end", t), _num(r"at most ([0-9.]+)% error at the end", t)),
        rip_max=_first(_num(r"control ripple at most ([0-9.]+)%", t), _num(r"control ripple of at most ([0-9.]+)%", t)),
    )


def parse_observation(t):
    t = t.lower()
    if "no oscillation" in t:
        osc = 0
    elif "oscillates once" in t:
        osc = 1
    else:
        osc = _first(_num(r"(\d+) oscillation cycle", t), _num(r"oscillates (\d+) times", t))
    if "never at its limit" in t:
        sat_pct, sat_time = 0.0, 0.0
    else:
        sat_pct = _num(r"\(([0-9.]+)% of the recording\)", t)
        sat_time = _first(_time(r"saturated for", t), _time(r"at its limit for", t))
    return dict(
        overshoot=_first(_num(r"([0-9.]+)% overshoot", t), _num(r"overshoot(?: of)? ([0-9.]+)%", t)),
        sse=_first(_num(r"error at the end of the recording ([0-9.]+)%", t),
                   _num(r"([0-9.]+)% error left at the end", t)),
        ripple=_first(_num(r"control ripple ([0-9.]+)%", t), _num(r"control-signal ripple ([0-9.]+)%", t)),
        osc=osc,
        rise=_time(r"rise time", t),
        settle=_time(r"settling time", t),
        no_settle="did not settle" in t,
        no_rise="did not reach 90%" in t,
        sat_pct=sat_pct,
        sat_time=sat_time,
    )


def parse(text):
    req_part, obs_part = split_text(text)
    return parse_requirements(req_part), parse_observation(obs_part)


def rules(req, obs):
    """Compare each measurement with its stated requirement."""
    over = obs["overshoot"] > req["os_max"]
    many_osc = obs["osc"] > req["osc_max"]
    slow = obs["no_settle"] or obs["settle"] > req["ts_max"]
    end_err = obs["sse"] > req["sse_max"]
    noisy = obs["ripple"] > req["rip_max"]
    long_sat = obs["sat_time"] > 0.2 * req["ts_max"]
    if long_sat and (slow or over) and not noisy:
        return "actuator_saturation"
    if noisy and not over and not many_osc and not end_err:
        return "kd_too_high"
    if end_err and not over and not many_osc:
        return "ki_too_low"
    if over or many_osc:
        # integral-driven overshoot is large and slow to die out; proportional
        # overshoot is moderate and decays fast (thresholds set on train only)
        settle_ratio = 5.0 if obs["no_settle"] else obs["settle"] / req["ts_max"]
        if obs["overshoot"] > 45 or (over and settle_ratio > 0.8):
            return "ki_too_high"
        return "kp_too_high"
    if slow:
        return "gain_too_low"
    return "well_tuned"


FEATS = ["os_ratio", "osc_excess", "settle_ratio", "sse_ratio", "rip_ratio", "sat_ratio", "osc", "no_rise"]


def features(text):
    req, obs = parse(text)
    settle = obs["settle"] if not obs["no_settle"] else 5 * req["ts_max"]
    return [
        obs["overshoot"] / req["os_max"],
        obs["osc"] - req["osc_max"],
        settle / req["ts_max"],
        obs["sse"] / req["sse_max"],
        obs["ripple"] / req["rip_max"],
        obs["sat_time"] / req["ts_max"],
        obs["osc"],
        float(obs["no_rise"]),
    ]


def train_tree(train_path="data/train.json"):
    from sklearn.tree import DecisionTreeClassifier
    data = json.load(open(train_path))
    return DecisionTreeClassifier(max_depth=6, random_state=0).fit(
        [features(d["input"]) for d in data], [d["label"] for d in data])


def report(name, data, preds):
    conf = defaultdict(Counter)
    for d, p in zip(data, preds):
        conf[d["label"]][str(p)] += 1
    present = [c for c in CLASSES if sum(conf[c].values()) > 0]
    correct = sum(conf[c][c] for c in CLASSES)
    rows = []
    for c in present:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in CLASSES if o != c)
        fn = sum(v for k, v in conf[c].items() if k != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((c, sum(conf[c].values()), prec, rec, f1))
    macro = sum(r[4] for r in rows) / len(rows)
    print(f"\n{name}: n={len(data)}  accuracy={correct / len(data):.3f}  macro-F1={macro:.3f}")
    for c, n, p, r, f in rows:
        print(f"  {c:22s} n={n:3d}  P={p:.2f} R={r:.2f} F1={f:.2f}   predicted as: {dict(conf[c])}")
    return correct / len(data), macro


def evaluate(path, tree):
    data = json.load(open(path))
    rp = []
    for d in data:
        req, obs = parse(d["input"])
        rp.append(rules(req, obs))
    a = report(f"RULES {path}", data, rp)
    b = report(f"TREE  {path}", data, list(tree.predict([features(d["input"]) for d in data])))
    return a, b


if __name__ == "__main__":
    args = sys.argv[1:] or ["data/val.json"]
    train = "data/train.json"
    if args[0].startswith("--train="):
        train = args.pop(0).split("=", 1)[1]
    tree = train_tree(train)
    for p in args:
        evaluate(p, tree)
