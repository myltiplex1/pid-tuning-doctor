"""Automatic consistency checks for every record of a generated dataset.

Run after EVERY generation. Any failure means the dataset is not accepted.

    python check_dataset.py data/

Checks
  1. Parser round trip: every requirement and measurement parsed from the
     input text equals the stored value (rounded as printed).
  2. Requirements vs label: well_tuned meets every stated requirement; every
     fault breaks the requirement(s) tied to it and no requirement that points
     to a different fault (same rule as the generator, re-derived here from
     the stored numbers, not from the generator's own flags).
  3. Only the named gain changed (ratio to nominal), gain_too_low scales all
     three by one common factor, saturation keeps the gains nominal.
  4. Evidence line: every "X against a Y limit/target" claim matches the stored
     numbers, and no claim is made for a requirement that was met.
  5. Recording length >= 3 x the settling deadline.
  6. Saturation: the actuator limit is >= 1.3 x the input needed to hold the
     target (reachable), and lower than the nominal peak.
  7. No identical input text appears in two splits.
"""

import json
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rule_baseline import parse  # noqa: E402
from generate_dataset import CLASS_TEXT, FIX_TEXT, SAFETY  # noqa: E402


def _secs(txt):
    v, unit = txt.split()
    return float(v) / 1000 if unit == "ms" else float(v)


def fmt_t(x):
    if x < 1:
        return float(f"{x * 1000:.0f}") / 1000
    if x < 100:
        return float(f"{x:.2f}")
    return float(f"{x:.0f}")


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol + 1e-9


def viol(m, s):
    v = set()
    if m["overshoot"] > s["os_max"]:
        v.add("overshoot")
    if m["oscillations"] > s["osc_max"]:
        v.add("oscillation")
    if m["settle"] is None or m["settle"] > s["ts_max"]:
        v.add("settling")
    if m["sse"] > s["sse_max"]:
        v.add("end_error")
    if m["ripple"] > s["rip_max"]:
        v.add("ripple")
    return v


def check_record(r):
    errs = []
    m, s, lab = r["metrics"], r["spec"], r["label"]
    req, obs = parse(r["input"])

    # 1. parser round trip
    exp_req = dict(os_max=s["os_max"], ts_max=fmt_t(s["ts_max"]), osc_max=s["osc_max"],
                   sse_max=s["sse_max"], rip_max=float(f"{s['rip_max']:.1f}"))
    for k, v in exp_req.items():
        if not close(req[k], v, 1e-6):
            errs.append(f"req {k}: parsed {req[k]} vs stored {v}")
    if not close(obs["overshoot"], float(f"{m['overshoot']:.0f}")):
        errs.append(f"overshoot parsed {obs['overshoot']} vs {m['overshoot']:.0f}")
    if not close(obs["sse"], float(f"{m['sse']:.1f}")):
        errs.append(f"end error parsed {obs['sse']} vs {m['sse']:.1f}")
    if not close(obs["ripple"], float(f"{m['ripple']:.1f}")):
        errs.append(f"ripple parsed {obs['ripple']} vs {m['ripple']:.1f}")
    if obs["osc"] != int(m["oscillations"]):
        errs.append(f"osc parsed {obs['osc']} vs {m['oscillations']}")
    if m["settle"] is None:
        if not obs["no_settle"]:
            errs.append("unsettled run not reported as such")
    elif not close(obs["settle"], fmt_t(m["settle"]), 1e-6):
        errs.append(f"settle parsed {obs['settle']} vs {fmt_t(m['settle'])}")
    if m["sat_time"] == 0:
        if obs["sat_time"] != 0:
            errs.append("sat_time should be 0")
    elif not close(obs["sat_time"], fmt_t(m["sat_time"]), 1e-6):
        errs.append(f"sat_time parsed {obs['sat_time']} vs {fmt_t(m['sat_time'])}")

    # 2. requirements vs label
    v = viol(m, s)
    low_sat = m["sat_time"] < 0.1 * s["ts_max"]
    ok = {
        "well_tuned": not v and low_sat,
        "kp_too_high": bool(v & {"overshoot", "oscillation"}) and "end_error" not in v and low_sat,
        "gain_too_low": v == {"settling"} and m["settle"] is not None and low_sat,
        "ki_too_low": "end_error" in v and not v & {"overshoot", "oscillation", "ripple"} and low_sat,
        "ki_too_high": "overshoot" in v and not v & {"end_error", "ripple"} and low_sat,
        "kd_too_high": v in ({"ripple"}, {"ripple", "settling"}) and low_sat,
        "actuator_saturation": (m["sat_time"] > 0.2 * s["ts_max"] and bool(v & {"settling", "overshoot"})
                                and "ripple" not in v),
    }[lab]
    if not ok:
        errs.append(f"label {lab} inconsistent with violations {sorted(v)} (sat_time {m['sat_time']})")
    if sorted(v) != r["violations"]:
        errs.append("stored violations differ from recomputed")

    # 3. only the named gain changed
    g, g0 = r["gains"], r["nominal_gains"]
    ratio = {k: (g[k] / g0[k] if g0[k] else (None if g[k] else 1.0)) for k in g}
    same = lambda k: ratio[k] is not None and abs(ratio[k] - 1) < 1e-9
    if lab == "kp_too_high" and not (ratio["kp"] > 1 and same("ki") and same("kd")):
        errs.append(f"kp_too_high changed other gains {ratio}")
    if lab == "ki_too_low" and not (ratio["ki"] < 1 and same("kp") and same("kd")):
        errs.append(f"ki_too_low changed other gains {ratio}")
    if lab == "ki_too_high" and not (ratio["ki"] > 1 and same("kp") and same("kd")):
        errs.append(f"ki_too_high changed other gains {ratio}")
    if lab == "kd_too_high" and not (same("kp") and same("ki") and (g0["kd"] == 0 or ratio["kd"] > 1)):
        errs.append(f"kd_too_high changed other gains {ratio}")
    if lab == "gain_too_low":
        rs = [ratio[k] for k in ("kp", "ki", "kd") if g0[k]]
        if not (max(rs) - min(rs) < 1e-9 and rs[0] < 1):
            errs.append(f"gain_too_low not a common factor {ratio}")
    if lab == "actuator_saturation" and not all(same(k) for k in ("kp", "ki", "kd") if g0[k]):
        errs.append(f"saturation changed gains {ratio}")

    # 4. output lines: diagnosis, evidence, fix, safety (v1.1: all checked)
    lines = r["output"].split("\n")
    if lines and lines[0] == "Check:":
        # v1.2 checklist-first answer: the 7 checklist lines must equal the
        # checklist rebuilt from the PRINTED input, then the v1.1 answer follows
        from make_v12 import checklist, broken_set
        want = checklist(r["input"])
        got = lines[:len(want)]
        if got != want:
            errs.append("checklist differs from the printed input")
        if broken_set(got) != v:
            errs.append(f"checklist broken set {sorted(broken_set(got))} differs from {sorted(v)}")
        lines = lines[len(want):]
    if len(lines) != 4:
        errs.append(f"output has {len(lines)} lines, expected 4")
        return errs
    if lines[0] != f"Diagnosis: {CLASS_TEXT[lab]}.":
        errs.append(f"diagnosis line '{lines[0]}' does not match label {lab}")
    if lines[2] != f"Fix: {FIX_TEXT[lab]}":
        errs.append("fix line does not match label")
    if lines[3] != SAFETY:
        errs.append("safety line missing or changed")
    ev = lines[1]
    if not ev.startswith("Evidence: ") or not ev.endswith("."):
        errs.append("evidence line malformed")

    # 4b. requirements broken according to the PRINTED numbers
    dv = set()
    if obs["overshoot"] > req["os_max"]:
        dv.add("overshoot")
    if obs["osc"] > req["osc_max"]:
        dv.add("oscillation")
    if obs["no_settle"] or obs["settle"] > req["ts_max"]:
        dv.add("settling")
    if obs["sse"] > req["sse_max"]:
        dv.add("end_error")
    if obs["ripple"] > req["rip_max"]:
        dv.add("ripple")
    if dv != v:
        errs.append(f"displayed violations {sorted(dv)} differ from stored {sorted(v)} (rounding)")

    # 4c. every evidence claim: present iff broken, and its numbers equal the
    #     printed measurement and requirement
    T = r"([0-9.]+ m?s)"
    pats = {
        "overshoot": r"overshoot (\d+)% against a (\d+)% limit",
        "oscillation": r"(\d+) oscillation cycles against a limit of (\d+)",
        "settling": r"(?:settles at " + T + r"|not settled after the " + T + r" recording) against a " + T + r" target",
        "end_error": r"([0-9.]+)% error left at the end of the recording against a (\d+)% limit",
        "ripple": r"control ripple ([0-9.]+)% against a ([0-9.]+)% limit",
    }
    claims = {k: re.search(p, ev) for k, p in pats.items()}
    if lab == "well_tuned":
        mw = re.search(r"all requirements met: overshoot (\d+)%, settles at " + T +
                       r", ([0-9.]+)% error at the end, ripple ([0-9.]+)%", ev)
        if not mw:
            errs.append("well_tuned evidence malformed")
        else:
            if float(mw.group(1)) != obs["overshoot"] or not close(_secs(mw.group(2)), obs["settle"], 1e-9) \
                    or float(mw.group(3)) != obs["sse"] or float(mw.group(4)) != obs["ripple"]:
                errs.append("well_tuned evidence numbers differ from the input")
        if any(claims.values()):
            errs.append("well_tuned evidence claims a broken requirement")
    else:
        for k, cm in claims.items():
            if cm and k not in dv:
                errs.append(f"evidence claims {k} but it is met")
            if k in dv and not cm:
                errs.append(f"evidence omits broken requirement {k}")
        cm = claims["overshoot"]
        if cm and (float(cm.group(1)) != obs["overshoot"] or float(cm.group(2)) != req["os_max"]):
            errs.append("evidence overshoot numbers differ from the input")
        cm = claims["oscillation"]
        if cm and (int(cm.group(1)) != obs["osc"] or int(cm.group(2)) != req["osc_max"]):
            errs.append("evidence oscillation numbers differ from the input")
        cm = claims["settling"]
        if cm:
            if cm.group(1) and not close(_secs(cm.group(1)), obs["settle"], 1e-9):
                errs.append("evidence settling time differs from the input")
            if cm.group(2) and (not obs["no_settle"] or not close(_secs(cm.group(2)), fmt_t(m["T"]), 1e-9)):
                errs.append("evidence recording length differs from the input")
            if not close(_secs(cm.group(3)), req["ts_max"], 1e-9):
                errs.append("evidence settling target differs from the input")
        cm = claims["end_error"]
        if cm and (float(cm.group(1)) != obs["sse"] or float(cm.group(2)) != req["sse_max"]):
            errs.append("evidence end-error numbers differ from the input")
        cm = claims["ripple"]
        if cm and (float(cm.group(1)) != obs["ripple"] or float(cm.group(2)) != req["rip_max"]):
            errs.append("evidence ripple numbers differ from the input")
        if lab == "actuator_saturation":
            ms = re.search(r"actuator at its limit for " + T + r" \(([0-9.]+)% of the recording\)", ev)
            if not ms or not close(_secs(ms.group(1)), obs["sat_time"], 1e-9) or float(ms.group(2)) != obs["sat_pct"]:
                errs.append("evidence saturation numbers differ from the input")
    for bad in ("drawn-out", "steady-state", "low-pass filter the derivative"):
        if bad in r["output"]:
            errs.append(f"old wording '{bad}' in output")

    # 5. recording length
    if m["T"] < 3 * s["ts_max"] - 1e-9:
        errs.append(f"recording {m['T']} shorter than 3 x deadline {s['ts_max']}")

    # 6. saturation reachable
    if lab == "actuator_saturation":
        if not (r["u_limit"] >= 1.3 * abs(r["u_required"]) - 1e-12):
            errs.append(f"unreachable: limit {r['u_limit']} vs needed {r['u_required']}")
    return errs


def main(folder):
    splits = {}
    for s in ["train", "val", "test_A", "test_B"]:
        p = os.path.join(folder, f"{s}.json")
        if os.path.exists(p):
            splits[s] = json.load(open(p))
    bad = Counter()
    examples = {}
    total = 0
    for s, rows in splits.items():
        for i, r in enumerate(rows):
            total += 1
            for e in check_record(r):
                key = e.split(":")[0][:60]
                bad[key] += 1
                examples.setdefault(key, f"{s}#{i + 1} ({r['label']}): {e}")
    # 7. duplicates across splits
    seen = {}
    dup = 0
    for s, rows in splits.items():
        for r in rows:
            if r["input"] in seen and seen[r["input"]] != s:
                dup += 1
            seen[r["input"]] = s
    print(f"checked {total} records in {list(splits)}")
    if dup:
        print(f"FAIL duplicates across splits: {dup}")
    if bad:
        print("FAIL")
        for k, n in bad.most_common():
            print(f"  {n:4d} x {k}   e.g. {examples[k]}")
        return 1
    if not dup:
        print("PASS: all checks")
    return 0 if not dup else 1


def selftest(folder):
    """Corrupt real records in memory; every corruption must be caught."""
    import copy
    rows = json.load(open(os.path.join(folder, "train.json")))
    assert not check_record(rows[0]), "an untouched record should pass"
    by = {}
    for r in rows:
        by.setdefault(r["label"], r)
    cases = []
    r = copy.deepcopy(by["kd_too_high"])
    r["output"] = re.sub(r"control ripple [0-9.]+%", "control ripple 999.0%", r["output"], count=1)
    cases.append(("evidence ripple number", r))
    r = copy.deepcopy(by["well_tuned"])
    r["output"] = r["output"].replace("Diagnosis: Well tuned.", "Diagnosis: Ki too high (integral overshoot).")
    cases.append(("diagnosis vs label", r))
    r = copy.deepcopy(by["kp_too_high"])
    r["output"] = re.sub(r"overshoot (\d+)% against", lambda mm: f"overshoot {int(mm.group(1)) + 7}% against", r["output"])
    cases.append(("evidence overshoot number", r))
    r = copy.deepcopy(by["gain_too_low"])
    r["output"] = r["output"].replace(r["output"].split("\n")[2], "Fix: Reduce Ki.")
    cases.append(("fix vs label", r))
    r = copy.deepcopy(by["ki_too_low"])
    r["metrics"]["sse"] = 0.0
    cases.append(("stored metric vs printed text", r))
    r = copy.deepcopy(by["well_tuned"])
    from rule_baseline import split_text
    head, obs_part = split_text(r["input"])
    marker = r["input"][len(head):len(r["input"]) - len(obs_part)]
    obs_part = re.sub(r"(\d+)% overshoot|overshoot (?:of )?(\d+)%", "overshoot 99%", obs_part, count=1)
    r["input"] = head + marker + obs_part
    cases.append(("printed measurement changed", r))
    if rows[0]["output"].startswith("Check:"):
        r = copy.deepcopy(by["kp_too_high"])
        r["output"] = r["output"].replace("vs limit 15%: broken", "vs limit 15%: met", 1).replace(
            "vs limit 10%: broken", "vs limit 10%: met", 1)
        cases.append(("checklist broken -> met", r))
        r = copy.deepcopy(by["well_tuned"])
        r["output"] = re.sub(r"- overshoot (\d+)%", lambda mm: f"- overshoot {int(mm.group(1)) + 3}%", r["output"], count=1)
        cases.append(("checklist number changed", r))
    ok = True
    for name, rec in cases:
        caught = bool(check_record(rec))
        print(f"  selftest {name:32s} {'caught' if caught else 'MISSED'}")
        ok &= caught
    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        sys.exit(selftest(args[1] if len(args) > 1 else "data"))
    sys.exit(main(args[0] if args else "data"))
