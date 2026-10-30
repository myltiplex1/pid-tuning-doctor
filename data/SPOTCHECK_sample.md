# Spot-check sample — v1.1 final train set (seed 20261031)

3 random examples per class (record numbers count from 1 in data/train.json).


## well_tuned (60 in train)


**Record 45** · plant: second_order · change from nominal: kp x0.89, ki x1.05, kd x1.09 · broken: none

INPUT:
```
Plant: a second-order system (like a mass-spring-damper).
Requirements: overshoot at most 15%, settling time at most 1.20 s, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 7.7%.
Gains: Kp=1.008, Ki=5.848, Kd=0.06.
Step response: overshoot 11%, 10-90% rise time 143 ms, settling time 669 ms, 0.0% error left at the end of the recording, control-signal ripple 2.6% of full scale, no oscillation, actuator never at its limit.
```
OUTPUT:
```
Diagnosis: Well tuned.
Evidence: all requirements met: overshoot 11%, settles at 669 ms, 0.0% error at the end, ripple 2.6%.
Fix: No change needed. Keep the current gains.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 37** · plant: dc_motor · change from nominal: kp x1.00, ki x1.14, kd x0.86 · broken: none

INPUT:
```
DC motor position control (QUBE-Servo 3 with inertia disc) under PID control (Kp=3.515, Ki=11.802, Kd=0.131). Requirements: overshoot at most 10%, settling time at most 930 ms, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 0.3%. Measured step response: overshoot 6%, rise time 64 ms, settling time 436 ms, control ripple 0.1%, actuator never at its limit, error at the end of the recording 0.1%, no oscillation.
```
OUTPUT:
```
Diagnosis: Well tuned.
Evidence: all requirements met: overshoot 6%, settles at 436 ms, 0.1% error at the end, ripple 0.1%.
Fix: No change needed. Keep the current gains.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 36** · plant: second_order · change from nominal: kp x0.91, ki x0.99, kd x1.02 · broken: none

INPUT:
```
System: a second-order system (like a mass-spring-damper)
Requirements: overshoot at most 15%, settling time at most 1.20 s, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 4.8%
PID gains: Kp=1.356, Ki=5.196, Kd=0.073
Observed: 4% overshoot, 10-90% rise time 222 ms, control-signal ripple 1.6% of full scale, 2% settling time 838 ms, error at the end of the recording 0.0%, actuator never at its limit, no oscillation
Diagnose the tuning.
```
OUTPUT:
```
Diagnosis: Well tuned.
Evidence: all requirements met: overshoot 4%, settles at 838 ms, 0.0% error at the end, ripple 1.6%.
Fix: No change needed. Keep the current gains.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


## kp_too_high (60 in train)


**Record 63** · plant: second_order · change from nominal: kp x4.17, ki x1.00, kd x1.00 · broken: overshoot

INPUT:
```
Plant: a second-order system (like a mass-spring-damper).
Requirements: settle within 2.50 s with at most 10% overshoot, no more than 2 oscillation cycles, at most 2% error at the end and control ripple of at most 6.7%.
Gains: Kp=19.633, Ki=12.29, Kd=0.413.
Step response: peak overshoot of 18%, 10-90% rise time 124 ms, 2% settling time 748 ms, control-signal ripple 3.5% of full scale, 2 oscillation cycles, 0.0% error left at the end of the recording, control signal at its limit for 157 ms (2% of the recording).
```
OUTPUT:
```
Diagnosis: Kp too high (oscillatory).
Evidence: overshoot 18% against a 10% limit.
Fix: Reduce Kp by about half, or increase Kd to add damping.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 115** · plant: dc_motor · change from nominal: kp x6.29, ki x1.00, kd x1.00 · broken: oscillation, overshoot

INPUT:
```
DC motor position control (QUBE-Servo 3 with inertia disc) under PID control (Kp=17.852, Ki=6.435, Kd=0.191). Requirements: overshoot at most 15%, settling time at most 1.60 s, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 0.4%. Measured step response: peak overshoot of 34%, rise time 34 ms, actuator saturated for 74 ms (2% of the recording), 0.0% error left at the end of the recording, 3 oscillation cycles, control-signal ripple 0.0% of full scale, settling time 328 ms.
```
OUTPUT:
```
Diagnosis: Kp too high (oscillatory).
Evidence: overshoot 34% against a 15% limit; 3 oscillation cycles against a limit of 2.
Fix: Reduce Kp by about half, or increase Kd to add damping.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 81** · plant: second_order · change from nominal: kp x6.07, ki x1.00, kd x1.00 · broken: overshoot, settling

INPUT:
```
System: a second-order system (like a mass-spring-damper)
Requirements: settle within 6.60 s with at most 15% overshoot, no more than 2 oscillation cycles, at most 2% error at the end and control ripple of at most 4.7%
PID gains: Kp=4.62, Ki=1.35, Kd=0.373
Observed: overshoot 21%, rise time 314 ms, control-signal ripple 2.1% of full scale, 0.1% error left at the end of the recording, control signal at its limit for 247 ms (1% of the recording), settling time 7.09 s, 1 oscillation cycle
Diagnose the tuning.
```
OUTPUT:
```
Diagnosis: Kp too high (oscillatory).
Evidence: overshoot 21% against a 15% limit; settles at 7.09 s against a 6.60 s target.
Fix: Reduce Kp by about half, or increase Kd to add damping.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


## gain_too_low (52 in train)


**Record 141** · plant: dc_motor · change from nominal: kp x0.26, ki x0.26, kd x0.26 · broken: settling

INPUT:
```
Plant: DC motor position control (QUBE-Servo 3 with inertia disc).
Requirements: overshoot at most 10%, settling time at most 1.20 s, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 0.3%.
Gains: Kp=0.796, Ki=3.193, Kd=0.045.
Step response: peak overshoot of 4%, 10-90% rise time 262 ms, settling time 1.21 s, control ripple 0.0%, actuator never at its limit, 0.1% error left at the end of the recording, no oscillation.
```
OUTPUT:
```
Diagnosis: Overall gain too low (sluggish).
Evidence: settles at 1.21 s against a 1.20 s target; slow rise with little overshoot.
Fix: Raise Kp, Ki and Kd together by the same factor (try about 2 times), keeping their ratios, and re-test against the settling-time target.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 123** · plant: second_order · change from nominal: kp x0.14, ki x0.14, kd x0.14 · broken: settling

INPUT:
```
A second-order system (like a mass-spring-damper) under PID control (Kp=0.342, Ki=1.011, Kd=0.03). Requirements: overshoot at most 15%, settling time at most 1.90 s, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 5.1%. Measured step response: overshoot 0%, 10-90% rise time 1.39 s, actuator never at its limit, no oscillation, 0.0% error left at the end of the recording, control-signal ripple 0.2% of full scale, 2% settling time 2.56 s.
```
OUTPUT:
```
Diagnosis: Overall gain too low (sluggish).
Evidence: settles at 2.56 s against a 1.90 s target; slow rise with little overshoot.
Fix: Raise Kp, Ki and Kd together by the same factor (try about 2 times), keeping their ratios, and re-test against the settling-time target.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 172** · plant: second_order · change from nominal: kp x0.20, ki x0.20, kd x0.20 · broken: settling

INPUT:
```
Plant: a second-order system (like a mass-spring-damper).
Requirements: overshoot at most 10%, settling time at most 2.80 s, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 5.7%.
Gains: Kp=0.632, Ki=1.504, Kd=0.065.
Step response: peak overshoot of 0%, rise time 1.73 s, actuator never at its limit, no oscillation, settling time 3.15 s, error at the end of the recording 0.0%, control ripple 0.4%.
```
OUTPUT:
```
Diagnosis: Overall gain too low (sluggish).
Evidence: settles at 3.15 s against a 2.80 s target; slow rise with little overshoot.
Fix: Raise Kp, Ki and Kd together by the same factor (try about 2 times), keeping their ratios, and re-test against the settling-time target.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


## ki_too_low (60 in train)


**Record 230** · plant: dc_motor · change from nominal: kp x1.00, ki x0.00, kd x1.00 · broken: end_error, settling

INPUT:
```
DC motor position control (QUBE-Servo 3 with inertia disc) under PID control (Kp=6.521, Ki=0.133, Kd=0.247). Requirements: overshoot at most 15%, settling time at most 750 ms, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 0.7%. Measured step response: 0% overshoot, rise time 100 ms, control-signal ripple 0.0% of full scale, no oscillation, 2.6% error left at the end of the recording, did not settle within the 2.25 s recording, actuator saturated for 8 ms (0% of the recording).
```
OUTPUT:
```
Diagnosis: Ki too low (remaining error).
Evidence: not settled after the 2.25 s recording against a 750 ms target; 2.6% error left at the end of the recording against a 2% limit.
Fix: Increase Ki so the integral removes the remaining error.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 196** · plant: second_order · change from nominal: kp x1.00, ki x0.01, kd x1.00 · broken: end_error, settling

INPUT:
```
I am tuning a PID controller for a second-order system (like a mass-spring-damper). Requirements: overshoot at most 15%, settling time at most 3.20 s, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 7.8%. Gains are Kp=1.282, Ki=0.03, Kd=0.234. The step response shows 0% overshoot, did not reach 90% of the target, no oscillation, 30.0% error left at the end of the recording, control ripple 2.7%, actuator never at its limit, did not settle within the 9.60 s recording. What is wrong?
```
OUTPUT:
```
Diagnosis: Ki too low (remaining error).
Evidence: not settled after the 9.60 s recording against a 3.20 s target; 30.0% error left at the end of the recording against a 2% limit.
Fix: Increase Ki so the integral removes the remaining error.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 213** · plant: first_order · change from nominal: kp x1.00, ki x0.02, kd 0 (nominal 0) · broken: end_error, settling

INPUT:
```
Plant: a first-order process (like a motor speed loop or small thermal system).
Requirements: settle within 3.30 s with at most 15% overshoot, no more than 2 oscillation cycles, at most 2% error at the end and control ripple of at most 1.0%.
Gains: Kp=4.459, Ki=0.163, Kd=0.
Step response: peak overshoot of 0%, did not reach 90% of the target, actuator never at its limit, control-signal ripple 0.3% of full scale, did not settle within the 14.21 s recording, 17.7% error left at the end of the recording, no oscillation.
```
OUTPUT:
```
Diagnosis: Ki too low (remaining error).
Evidence: not settled after the 14.21 s recording against a 3.30 s target; 17.7% error left at the end of the recording against a 2% limit.
Fix: Increase Ki so the integral removes the remaining error.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


## ki_too_high (59 in train)


**Record 285** · plant: second_order · change from nominal: kp x1.00, ki x4.99, kd x1.00 · broken: oscillation, overshoot, settling

INPUT:
```
Plant: a second-order system (like a mass-spring-damper).
Requirements: overshoot at most 15%, settling time at most 980 ms, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 4.1%.
Gains: Kp=2.98, Ki=93.173, Kd=0.14.
Step response: peak overshoot of 69%, rise time 64 ms, actuator never at its limit, 6 oscillation cycles, error at the end of the recording 0.1%, 2% settling time 1.82 s, control ripple 1.4%.
```
OUTPUT:
```
Diagnosis: Ki too high (integral overshoot).
Evidence: overshoot 69% against a 15% limit; 6 oscillation cycles against a limit of 2; settles at 1.82 s against a 980 ms target.
Fix: Reduce Ki (try a third of the current value); add anti-windup if the actuator saturates.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 250** · plant: second_order · change from nominal: kp x1.00, ki x5.35, kd x1.00 · broken: oscillation, overshoot, settling

INPUT:
```
I am tuning a PID controller for a second-order system (like a mass-spring-damper). Requirements: overshoot at most 15%, settling time at most 1.40 s, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 3.5%. Gains are Kp=0.673, Ki=13.795, Kd=0.056. The step response shows overshoot 73%, rise time 110 ms, actuator never at its limit, 0.2% error left at the end of the recording, 2% settling time 3.79 s, control ripple 1.2%, 7 oscillation cycles. What is wrong?
```
OUTPUT:
```
Diagnosis: Ki too high (integral overshoot).
Evidence: overshoot 73% against a 15% limit; 7 oscillation cycles against a limit of 2; settles at 3.79 s against a 1.40 s target.
Fix: Reduce Ki (try a third of the current value); add anti-windup if the actuator saturates.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 262** · plant: second_order · change from nominal: kp x1.00, ki x6.54, kd x1.00 · broken: oscillation, overshoot, settling

INPUT:
```
System: a second-order system (like a mass-spring-damper)
Requirements: settle within 2.50 s with at most 10% overshoot, no more than 2 oscillation cycles, at most 2% error at the end and control ripple of at most 3.5%
PID gains: Kp=1.216, Ki=24.599, Kd=0.128
Observed: peak overshoot of 91%, 10-90% rise time 107 ms, did not settle within the 7.50 s recording, control signal at its limit for 85 ms (1% of the recording), 0.5% error left at the end of the recording, 12 oscillation cycles, control ripple 1.2%
Diagnose the tuning.
```
OUTPUT:
```
Diagnosis: Ki too high (integral overshoot).
Evidence: overshoot 91% against a 10% limit; 12 oscillation cycles against a limit of 2; not settled after the 7.50 s recording against a 2.50 s target.
Fix: Reduce Ki (try a third of the current value); add anti-windup if the actuator saturates.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


## kd_too_high (30 in train)


**Record 317** · plant: second_order · change from nominal: kp x1.00, ki x1.00, kd x5.61 · broken: ripple, settling

INPUT:
```
A second-order system (like a mass-spring-damper) under PID control (Kp=3.573, Ki=3.514, Kd=2.521). Requirements: overshoot at most 15%, settling time at most 3.20 s, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 6.8%. Measured step response: 15% overshoot, rise time 1.16 s, oscillates once, control signal at its limit for 20 ms (0% of the recording), settling time 5.05 s, control ripple 11.2%, 0.2% error left at the end of the recording.
```
OUTPUT:
```
Diagnosis: Kd too high (noise amplification).
Evidence: settles at 5.05 s against a 3.20 s target; control ripple 11.2% against a 6.8% limit.
Fix: Reduce Kd or strengthen the derivative filtering, then re-test.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 304** · plant: second_order · change from nominal: kp x1.00, ki x1.00, kd x4.45 · broken: ripple, settling

INPUT:
```
I am tuning a PID controller for a second-order system (like a mass-spring-damper). Requirements: settle within 4.60 s with at most 15% overshoot, no more than 2 oscillation cycles, at most 2% error at the end and control ripple of at most 4.5%. Gains are Kp=1.047, Ki=1.106, Kd=0.746. The step response shows overshoot 14%, rise time 1.17 s, error at the end of the recording 0.0%, control ripple 5.0%, 1 oscillation cycle, actuator never at its limit, settling time 4.92 s. What is wrong?
```
OUTPUT:
```
Diagnosis: Kd too high (noise amplification).
Evidence: settles at 4.92 s against a 4.60 s target; control ripple 5.0% against a 4.5% limit.
Fix: Reduce Kd or strengthen the derivative filtering, then re-test.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 306** · plant: second_order · change from nominal: kp x1.00, ki x1.00, kd x3.56 · broken: ripple

INPUT:
```
I am tuning a PID controller for a second-order system (like a mass-spring-damper). Requirements: settle within 12.00 s with at most 15% overshoot, no more than 2 oscillation cycles, at most 2% error at the end and control ripple of at most 8.6%. Gains are Kp=7.056, Ki=2.793, Kd=7.743. The step response shows peak overshoot of 12%, rise time 2.03 s, 2% settling time 10.05 s, 0.0% error left at the end of the recording, control-signal ripple 9.1% of full scale, actuator saturated for 151 ms (0% of the recording), no oscillation. What is wrong?
```
OUTPUT:
```
Diagnosis: Kd too high (noise amplification).
Evidence: control ripple 9.1% against a 8.6% limit.
Fix: Reduce Kd or strengthen the derivative filtering, then re-test.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


## actuator_saturation (56 in train)


**Record 342** · plant: first_order · change from nominal: kp x1.00, ki x1.00, kd 0 (nominal 0) · broken: overshoot

INPUT:
```
I am tuning a PID controller for a first-order process (like a motor speed loop or small thermal system). Requirements: settle within 1.10 s with at most 15% overshoot, no more than 2 oscillation cycles, at most 2% error at the end and control ripple of at most 1.5%. Gains are Kp=1.463, Ki=9.173, Kd=0. The step response shows overshoot 27%, rise time 188 ms, control ripple 1.1%, control signal at its limit for 348 ms (10% of the recording), settling time 817 ms, no oscillation, error at the end of the recording 0.0%. What is wrong?
```
OUTPUT:
```
Diagnosis: Actuator saturation.
Evidence: actuator at its limit for 348 ms (10% of the recording); overshoot 27% against a 15% limit.
Fix: The actuator can hold this target but spends long periods at its limit. Use a smaller setpoint step or lower gains so it stays off the limit, and add anti-windup to the integrator. Lower gains cannot add actuator capacity: if the target cannot be held at steady state, a larger actuator or a smaller target is needed.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 325** · plant: first_order · change from nominal: kp x1.00, ki x1.00, kd 0 (nominal 0) · broken: overshoot

INPUT:
```
System: a first-order process (like a motor speed loop or small thermal system)
Requirements: overshoot at most 15%, settling time at most 1.50 s, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 0.8%
PID gains: Kp=0.824, Ki=2.747, Kd=0
Observed: 20% overshoot, rise time 295 ms, 2% settling time 1.33 s, error at the end of the recording 0.0%, actuator saturated for 449 ms (7% of the recording), control-signal ripple 0.7% of full scale, no oscillation
Diagnose the tuning.
```
OUTPUT:
```
Diagnosis: Actuator saturation.
Evidence: actuator at its limit for 449 ms (7% of the recording); overshoot 20% against a 15% limit.
Fix: The actuator can hold this target but spends long periods at its limit. Use a smaller setpoint step or lower gains so it stays off the limit, and add anti-windup to the integrator. Lower gains cannot add actuator capacity: if the target cannot be held at steady state, a larger actuator or a smaller target is needed.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:


**Record 352** · plant: second_order · change from nominal: kp x1.00, ki x1.00, kd x1.00 · broken: overshoot

INPUT:
```
System: a second-order system (like a mass-spring-damper)
Requirements: overshoot at most 15%, settling time at most 950 ms, at most 2 oscillation cycles, error at most 2% at the end, control ripple at most 8.1%
PID gains: Kp=4.384, Ki=21.241, Kd=0.121
Observed: 23% overshoot, rise time 160 ms, 2% settling time 867 ms, control signal at its limit for 295 ms (10% of the recording), error at the end of the recording 0.0%, no oscillation, control-signal ripple 6.5% of full scale
Diagnose the tuning.
```
OUTPUT:
```
Diagnosis: Actuator saturation.
Evidence: actuator at its limit for 295 ms (10% of the recording); overshoot 23% against a 15% limit.
Fix: The actuator can hold this target but spends long periods at its limit. Use a smaller setpoint step or lower gains so it stays off the limit, and add anti-windup to the integrator. Lower gains cannot add actuator capacity: if the target cannot be held at steady state, a larger actuator or a smaller target is needed.
This is a learning aid; check with your instructor if unsure.
```
Verdict: ☐ agree ☐ disagree — why:
