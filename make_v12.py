"""v1.2 'checklist-first' answers (PROTOCOL §14, run B).

Reads the v1.1 data in data/ and writes data_v12/ with the SAME inputs, labels,
splits and metadata; only the target answer changes. Each answer first checks
the five stated requirements, using the numbers exactly as printed in that
input, then gives the v1.1 Diagnosis / Evidence / Fix / safety lines unchanged:

    Check:
    - overshoot 34% vs limit 15%: broken
    - oscillations 3 vs limit 2: broken
    - settling 328 ms vs limit 1.60 s: met
    - end error 0.0% vs limit 2%: met
    - ripple 0.0% vs limit 0.4%: met
    - time at actuator limit: 74 ms (2% of the recording)
    Diagnosis: Kp too high (oscillatory).
    Evidence: ...
    Fix: ...
    This is a learning aid; check with your instructor if unsure.

The time at the actuator limit is information, not a stated requirement.

    python make_v12.py            # data/ -> data_v12/ (+ gemma_*.json)
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rule_baseline import parse  # noqa: E402

SPLITS = ["train", "val", "test_A", "test_B"]


def fmt_t(x):
    if x < 1:
        return f"{x * 1000:.0f} ms"
    if x < 100:
        return f"{x:.2f} s"
    return f"{x:.0f} s"


def checklist(text):
    """Checklist lines from the PRINTED requirements and measurements."""
    req, obs = parse(text)
    st = lambda broken: "broken" if broken else "met"
    lines = ["Check:"]
    lines.append(f"- overshoot {obs['overshoot']:.0f}% vs limit {req['os_max']:.0f}%: "
                 f"{st(obs['overshoot'] > req['os_max'])}")
    lines.append(f"- oscillations {obs['osc']:.0f} vs limit {req['osc_max']:.0f}: "
                 f"{st(obs['osc'] > req['osc_max'])}")
    if obs["no_settle"]:
        lines.append(f"- settling: not settled vs limit {fmt_t(req['ts_max'])}: broken")
    else:
        lines.append(f"- settling {fmt_t(obs['settle'])} vs limit {fmt_t(req['ts_max'])}: "
                     f"{st(obs['settle'] > req['ts_max'])}")
    lines.append(f"- end error {obs['sse']:.1f}% vs limit {req['sse_max']:.0f}%: "
                 f"{st(obs['sse'] > req['sse_max'])}")
    lines.append(f"- ripple {obs['ripple']:.1f}% vs limit {req['rip_max']:.1f}%: "
                 f"{st(obs['ripple'] > req['rip_max'])}")
    if obs["sat_time"] == 0:
        lines.append("- time at actuator limit: none")
    else:
        lines.append(f"- time at actuator limit: {fmt_t(obs['sat_time'])} ({obs['sat_pct']:.0f}% of the recording)")
    return lines


def broken_set(lines):
    names = {"overshoot": "overshoot", "oscillations": "oscillation", "settling": "settling",
             "end error": "end_error", "ripple": "ripple"}
    out = set()
    for ln in lines[1:]:
        if ln.endswith(": broken"):
            key = ln[2:].split(" ")[0] if not ln.startswith("- end error") else "end error"
            out.add(names[key.rstrip(":")])
    return out


def main(src="data", dst="data_v12"):
    os.makedirs(dst, exist_ok=True)
    for s in SPLITS:
        rows = json.load(open(os.path.join(src, f"{s}.json")))
        for r in rows:
            cl = checklist(r["input"])
            assert broken_set(cl) == set(r["violations"]), (s, r["violations"], cl)
            r["output"] = "\n".join(cl) + "\n" + r["output"]
        json.dump(rows, open(os.path.join(dst, f"{s}.json"), "w"), indent=1)
    shutil.copy(os.path.join(src, "summary.txt"), os.path.join(dst, "summary.txt"))
    import prepare_gemma_data as P
    for s in SPLITS:
        P.convert(s, data_dir=dst)
    print("v1.2 checklist answers written to", dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
