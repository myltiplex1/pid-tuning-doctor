# Data Card — PID Tuning Doctor dataset v1.1

Revised 30 Oct 2026. Author: Enoch Oluwatimilehin Soyinka (OAU). Final data: train/val seed 20261031, fresh Test A/B seed 20261102; 661 examples. Frozen rules: PROTOCOL.md (v1.1-r5).

The dataset is 661 simulated PID step-response descriptions with injected tuning faults, built for an educational classifier. It contains no personal data and no web-scraped text. Its main gap is that no real student wording and no real hardware data are in it yet.

## Input versus output

- **Input (text, English):** the plant, five stated requirements (overshoot, settling time, oscillation cycles, error at the end of the recording, control ripple), the PID gains, and the measured step response: overshoot, rise time, settling time, oscillations, error at the end, time at the actuator limit (seconds and %), and control ripple.
- **Output (generated text):** `Diagnosis:` one of 7 classes; `Evidence:` the requirements the response breaks, with the numbers; `Fix:` one change; and a safety line ("learning aid; check with your instructor").
- Scoring uses only the diagnosis (accuracy, macro F1). The evidence and fix are judged separately.

## Classification problem

| Class | What was changed from a well-tuned controller |
| --- | --- |
| Well tuned | all gains within ±15% |
| Kp too high | Kp only, ×2.5–8 |
| Overall gain too low (sluggish) | Kp, Ki and Kd by one common factor, ×0.1–0.45 |
| Ki too low | Ki only, ×0–0.03 |
| Ki too high | Ki only, ×4–15 |
| Kd too high | Kd only, ×2–10 |
| Actuator saturation | gains unchanged; actuator limit lowered (target still reachable) |

**Why these classes:** each maps to one intervention a student can undo. "Kp too low" was replaced by "Overall gain too low" because lowering only Kp produced no clean sluggish examples within the tested ranges; all of them overshot. Saturation is limited to targets the actuator can still reach; an unreachable setpoint is a different problem and is excluded.

## Labeling

- **Source:** no human or AI labellers. Each label is a **known injected fault that passed a symptom check**: an example is kept only if the response breaks the requirement(s) tied to that fault and none that point to a different fault. Labels are not proven diagnoses, because different gain settings can produce similar symptoms.
- **Consistency checks:** `check_dataset.py` checks every record (parser round trip, label vs broken requirements, only the named gain changed, every evidence number vs the printed input, recording length, saturation reachability, no duplicates across splits). All 661 pass. A self-test deliberately corrupts 6 records, and all 6 are caught.
- **AI-assisted review:** audits by Claude, the AI assistant that built the data (2–3 per class), and four rounds by an external AI-assisted reviewer (30 examples each, all checks reproduced). These found and fixed real problems, including rounding that hid violations and a test-set exposure.
- **Author's manual audit (30 Oct, before training):** 21 training examples (3 per class) reviewed by the author as a control engineer; agreed with all 21. Interpretation notes: record 285 is excessive integral action, not windup (0% saturation); the saturation examples are close calls that could also read as "Ki too high".
- **Not yet done:** an instructor's review of the class definitions and fixes.

## Data creation

- **Synthetic, from physics simulation**, because there was no hardware access and no existing labelled dataset.
- **Training plants (randomised):** QUBE-Servo 3 DC motor, calibrated from the author's own QLabs measurements (K 23.1 rad/s/V, τ 0.092 s); a first-order and a second-order plant with small lags.
- **Test B plants, never trained on:** the author's EEE576 nonlinear water tank, a heater with dead time, and the QUBE rotary arm with a hanging pendulum.
- **Realism added:** encoder quantisation (motor and arm), Gaussian measurement noise (the other plants), a constant input bias, and actuator limits.
- **Text:** 4 sentence templates with shuffled measurement order. Examples are rejected if rounding changes which requirements are broken. The planned Gemini paraphrasing was **not used**.

| Split | Plants | Examples | Per class |
| --- | --- | --- | --- |
| Train | training plants | 377 | 60, 60, 52, 60, 59, 30, 56 |
| Validation | training plants | 60 | 10, 10, 8, 10, 10, 5, 7 |
| Test A | training plants (fresh seed 20261102) | 126 | 20, 20, 17, 20, 19, 13, 17 |
| Test B | unseen plants (fresh seed 20261102) | 98 | 15, 15, 15, 15, 15, 14, 9 |

(Class order: well tuned, Kp too high, overall gain too low, Ki too low, Ki too high, Kd too high, saturation.)

## Web collection ethics

- **No web scraping and no personal data.** The only outside sources are published physical parameters: the Quanser QUBE-Servo 3 values in the author's `params.py`, cross-checked with the Lehigh ME207 pendulum guide.
- **Planned Test C (student descriptions):** written consent, no names stored, right to withdraw. Not yet collected.
- **Licences:** code MIT, data CC BY 4.0 (planned); Gemma used under its terms of use.

## Bias and representation

Who and what is missing:

- **How students really write.** Every input is template text with exact numbers. Real students describe plots in their own words, often without numbers.
- **Other languages.** Students who would describe a response in Yoruba, Hausa, Igbo or Pidgin are not represented.
- **Real hardware effects.** Backlash, cable drag, sensor drift and real noise are missing; QLabs (Test D) only partly covers them.
- **Uneven plant coverage.** No training examples of "Kd too high" come from the motor; motor examples of "Overall gain too low" (12) and saturation (16) are below 20; "Kd too high" has only 30 training examples.
- **Excluded problems.** Mixed faults, multivariable or industrial loops, and unreachable setpoints.
- **Design choices that shape labels.** The thresholds (for example 2 oscillation cycles, ripple limit = 3× nominal) are design choices. 15 of 221 overshoot or ripple violations in train are only one printed unit above their limit.

## Values and community alignment

- **Fairness:** aimed at students with the least supervision; it runs offline on a laptop, which matters where power and data are unreliable.
- **Autonomy:** every answer shows the evidence it used, so a student can check the reasoning instead of trusting a label.
- **Well-being and safety:** every answer says it is a learning aid; the tool is not for real or safety-critical equipment.
- **Accountability:** frozen rules with file fingerprints, a public log of every decision and correction, test sets scored once, and a way to report wrong diagnoses (planned GitHub issues).

## Additional: development disclosures

- **v0.1 (29 Oct):** its Test A/B baseline scores **were seen** and influenced the Kd rule; v0.1 is discarded.
- **v1.0:** withdrawn after an external review; its Test A/B were never scored.
- **v1.1, first test sets (seed 20261031):** a CPU test harness accidentally computed both baselines on them; the scores were not opened and were deleted. Withdrawn and replaced by fresh Test A/B (seed 20261102); train and validation unchanged.
- All six plant families, including the Test B plants, were explored while designing the symptom checks. Test B is unseen by the model, not by the developer. Only Test C (students) and Test D (QLabs) are untouched.

## Additional: scope adjustment

The dataset cannot be rebuilt with human data within the capstone, so any claim is limited to: **English, numeric descriptions of single PID faults on simulated single-loop plants.** Claims about real students or real hardware wait for Test C and Test D, and the report will say so.

## Intended use

Educational aid for control students, with an instructor available. Not for tuning real, industrial or safety-critical equipment.
