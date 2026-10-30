# PID Tuning Doctor — Capstone log

Owner: Enoch Oluwatimilehin Soyinka (OAU). Course: Google DeepMind AI Research Foundations, capstone.
Started: 29 Oct 2026 (Lagos).
Stage 1 plan (Claude doc): https://claude.ai/artifact/EocvSabmNCWLpJoeMZExL2
Frozen rules: `PROTOCOL.md` **v1.1-r2**; current hashes in `FROZEN_v1.2_r7.sha256` (earlier hash files kept for the record).

**Log rule:** this file is the single source of truth. At the start of every session, Claude reads it first. After every step, Claude adds to the Decision log and the Session log. The master copy lives in the claude.ai project as `claude/CAPSTONE_LOG.md`.

**Cross-check rule (from 30 Oct):** nothing is reported as done until:
1. `check_dataset.py` (or the equivalent check for that step) passes on EVERY record, AND
2. Claude has manually read a sample (at least 2 per class) and compared the text with the numbers, AND
3. any claim about physics or causes is supported by an analysis or worded as "within the tested ranges".

This rule came after an external review of v0.1 found 5 real problems that Claude had missed.

---

## 1. The project in one paragraph

A small Gemma 3 model, fine-tuned with LoRA, reads a text description of a PID step response: the plant, five stated requirements, the gains and the measurements. It returns one of 7 classes, the evidence (the requirements broken), and a fix. Labels are **known injected faults that passed a symptom check**, not proven diagnoses. Only the named gain is changed, and an example is kept only if it breaks the requirement(s) tied to that fault. The aim is a learning aid for control students in Nigerian labs with little supervision. The main question: does it still work on unseen plants, on student-written text, and on QLabs?

**Pathway:** Classification, plus evidence and a fix.
**Classes (v1.0):** well_tuned · kp_too_high · gain_too_low ("Overall gain too low, sluggish") · ki_too_low · ki_too_high · kd_too_high · actuator_saturation (reachable targets only).

## 2. Status

| Stage | Item | Status |
|---|---|---|
| 1 | Plan doc (problem, impact, stakeholders, governance, sustainability, Data Card, evaluation, future work) | ✅ drafted 29 Oct (needs update to v1.0 classes) |
| 1 | Explain problem and impact statements to a friend; record feedback | ⬜ Enoch |
| 1 | Instructor or lab assistant checks the classes and fixes | ⬜ Enoch |
| 2 | Dataset v0.1 (seed 0) | ❌ discarded 30 Oct after external review (archive/) |
| 2 | Generator v1.0 + `check_dataset.py` + `PROTOCOL.md` frozen | ✅ 30 Oct |
| 2 | Dev set (seed 0, `dev/`): v1.0 check PASS on 669 records (historical); current v1.1 dev: PASS on 662; baselines on dev val only | ✅ 30 Oct |
| 2 | v1.0 final dataset (seed 20261030) | ❌ withdrawn 30 Oct after the second review (rounding hid violations); its Test A/B were never scored, and its val scores are kept as historical (archive/) |
| 2 | v1.1: display-consistency rule, "at most" wording, stronger checker + `--selftest`, wording fixes; frozen | ✅ 30 Oct |
| 2 | v1.1 final dataset (seed 20261031): train 377 and val 60 kept; its Test A/B (130/96) **withdrawn** after the baseline exposure. Val macro F1: rules 0.957, tree 0.957 | ✅ / ❌ tests 30 Oct |
| 2 | Fresh Test A/B (seed 20261102): 126 / 98, check PASS, no scores computed | ✅ 30 Oct |
| 2 | **Enoch's own manual audit**: 21 train examples (3/class; records 45, 37, 36, 63, 115, 81, 141, 123, 172, 230, 196, 213, 285, 250, 262, 317, 304, 306, 342, 325, 352): **agreed with all 21**. Claude verified every number in the notes against the records. Interpretation notes: #285 is not windup (0% sat); #317's late settling is likely extra Kd damping, not noise; saturation cases are close calls (7–10% at the limit) | ✅ 30 Oct |
| 2 | Reviewer spot-check of v1.1 (21 sample + 9 extra = 30 examples; all checks reproduced; verdict: agree; proceed to notebook) | ✅ 30 Oct |
| 2 | Data Card revised to v1.1 using the course template + extra rows (development disclosures, scope adjustment, intended use): in the plan doc and `DATA_CARD.md`; all counts verified against the data | ✅ 30 Oct |
| 2 | Plan doc brought up to v1.1 (seventh review, verified): overview example (real v1.1 record), class table ("Overall gain too low", requirement-based symptoms), removed "correct by construction" and "balanced", "7 categories = 6 faults + well tuned", audit status no longer contradictory, evaluation section compares with both baselines and has no pre-set score target | ✅ 30 Oct |
| 2 | Documentation correction v1.1-r1 (version labels on historical scores, dev count, file list) | ✅ 30 Oct |
| 2 | Enoch spot-checks a fresh sample of the final data | ⬜ |
| 2 | Test C: 50–100 classmate descriptions, with consent | ⬜ (now the key test) |
| 3 | Colab notebook `pid_doctor_lora.ipynb` + `pid_lib.py` (tested on CPU by `smoke_test.py` with a tiny random Gemma 3). Training config declared in PROTOCOL §12 (r2) | ✅ 30 Oct |
| 3 | **Run 1** (lr 5e-5) on T4, 29.6 min: best epoch 6, **val macro F1 0.397** (acc 0.45), train 0.480 (acc 0.541); zero-shot 0.000; baselines 0.957. Kd too high never predicted; frequent invalid threshold comparisons (train: 77 overshoot + 12 ripple). Reviewed with an external AI-assisted reviewer: wording corrected (see PROTOCOL §14) | ✅ 30 Oct |
| 3 | **Run A** (lr 1e-4, fresh model and adapter) on T4, 31.0 min: best epoch 6, **val macro F1 0.556** (acc 0.633, 38/60), train 0.687 (acc 0.732); better than run 1 at every epoch 2-6 (paired val: 15 vs 4, McNemar p 0.019) but one seed, epoch-to-epoch swing as big as the gap, and still far below baselines (0.957). Kd too high 0/5 val; false 'all requirements met' 9/16 val. Decision recorded BEFORE run B: run B lr 1e-4 (PROTOCOL r7) | ✅ 30 Oct |
| 3 | Run B data and notebook built while run A trains (v1.2: `make_v12.py`, `data_v12/`, checker + 2 new self-test corruptions, `ANSWER_FORMAT` switch; PROTOCOL r6, hashes FROZEN_v1.2_r6.sha256). Notebook run test passes in v1.2 AND v1.1 | ✅ 30 Oct |
| 3 | **Run B** (v1.2 checklist-first, lr 1e-4) on T4, 35.3 min, one kernel session (execution counts continuous; the 'disconnect' during 6b was a browser reconnect, not a runtime reset): best epoch 6, **val macro F1 0.861** (acc 0.867, 52/60), train 0.959 (acc 0.958); val F1 by epoch 0.282, 0.349, 0.589, 0.625, 0.613, 0.861. All numbers re-verified from the result files (both zips intact). Paired val vs run A: 16 vs 2 (McNemar p 0.0013). Checklist errors only on settling (train 42/170 broken lines called met) and ripple (7/51); never a false 'broken'. **Selected: run B epoch 6** (PROTOCOL r8, recorded before any test scoring) | ✅ 30 Oct |
| 3 | **Final test, once** (run B epoch 6): Test A F1 **0.894** (113/126), Test B (unseen plants) F1 **0.693** (69/98); zero-shot 0.000; rules 0.934 / 0.921, tree 0.978 / 0.918. Paired accuracy (McNemar): Test A vs rules 9/13, p 0.52 (no clear difference; NOT equivalence), vs tree 2/12, p 0.013; Test B vs rules and tree 3/24, p < 0.001. By the plan's own success criterion, the unseen-plant part was not met. Main weakness: judging settling time (and ripple) against the limit, worst on the tank (0/11). All numbers recomputed; files in `final_results/`. PROTOCOL r9 | ✅ 30 Oct |
| 4 | Report + reflection; Test C (students), Test D (QLabs) as future/optional; instructor review | ⬜ |
| 3 | Zero-shot Gemma 3 1B, LoRA fine-tune, final test evaluation (model + baselines, once) | ⬜ |
| 3 | Test D: about 30 QLabs step tests | ⬜ |
| 3 | Reflection and final report | ⬜ |

## 3. Key facts and where they came from

**Qube-Servo 3 DC motor + disc (Enoch's QLabs labs 01–04, `qube-servo3-python/PROGRESS.md`):**
- Ω/V = 23.1/(0.092 s + 1). Position: θ/V = 23.1/(s(0.092 s + 1)).
- Rm 7.5 Ω, kt = km 0.0422, Jeq 2.19e-5 kg·m² (QLabs, 20% above the manual), b 5e-6, Tc 4.9e-5 N·m.
- Encoder 2048 counts/rev. Recommended limit ±10 V.
- QLabs ports: 18922 DC motor, 18923 pendulum.

**Tank (EEE576 report, Part I):** A_c·dh/dt = q_in − C_d·A_o·√(2gh). Ac 0.5 m², h_op 1.5 m, q_op ≈ 0.0101 m³/s, τ 148.7 s, Kp 297.3 m/(m³/s).

**Rotary pendulum (params.py; matches the Lehigh ME207 QUBE guide):** arm 0.095 kg, 0.085 m; pendulum 0.024 kg, 0.129 m; Dr 1e-3, Dp 5e-5. The hanging pendulum is used as a test plant.

**Measurement models (corrected 30 Oct):** encoder quantisation only on dc_motor and pendulum_arm. Gaussian noise (0.3% of the step) on first_order, second_order, tank and heater.

**Course LoRA format (lab 5.6):** Keras `Gemma3CausalLM.from_preset("gemma3_1b")`, `enable_lora(rank=4)`, `{"prompts", "responses"}` with `<start_of_turn>` delimiters, sequence_length 400, lr about 2e-5, batch 1, about 10 epochs.

## 4. Dataset design (v1.0) — see PROTOCOL.md for the frozen details

- Every input states 5 requirements: overshoot, settling time, ≤ 2 oscillation cycles, ≤ 2% error at the end of the recording, and control ripple.
- Only the named gain is changed. gain_too_low scales Kp, Ki and Kd by one common factor (Ti, Td and the filter are unchanged).
- Each class has a "signature" set of broken requirements; anything outside it is rejected.
- Recording length ≥ 3× the settling deadline. The wording is "error at the end of the recording".
- **Saturation:** time at the limit (s and %) compared with the plant's own deadline; only reachable targets (limit ≥ 1.3× the input needed to hold the target, from the plant model).
- Evidence is built per example from the requirements actually broken.
- No forced balancing: plants that can't show a fault supply fewer examples.

**Dev results (seed 0), macro F1 on dev val:** current v1.1: hand rules 0.986, tree 0.914 (*historical v1.0: 0.905 / 0.950*). Current v1.1 final val (seed 20261031): 0.957 / 0.957. On clean simulator text the tree is strong, so the LLM's case rests on Test C (free text), unseen wording, and the generated evidence and fix.

## 5. Decision log

| Date | Decision | Why |
|---|---|---|
| 29 Oct | Drop PPG ideas | Waveforms, not text |
| 29 Oct | Drop the skin-tone auditor | Badly unbalanced; a keyword search nearly matches it |
| 29 Oct | Choose PID Tuning Doctor | Automatic labels; fits control, sysID and Quanser background; teaching need |
| 29 Oct | No hardware now; QLabs = Test D later | No access to the servo |
| 29 Oct | Randomise plants; test on unseen plants | Avoid the NER "learned the template" trap |
| 29 Oct | Inverted pendulum excluded; hanging pendulum = test plant | A single PID loop cannot balance it |
| 29 Oct | LLM auto-tuning (above the loop, never inside it) = future work | Latency; needs a safety check |
| 29 Oct | Small lag added to first-order, second-order and tank plants | Without lag, high Kp cannot oscillate on a first-order PI loop |
| 30 Oct | **Discard v0.1** | External review found: label/intervention mismatch; fixed evidence phrase (23/60 ki_too_high false "slow settling"); parser read the spec as the measurement (186 train records); 36 recordings ended before the deadline; unconditional saturation fix; 8 kp_too_high + 1 kd_too_high met the whole spec. All verified by Claude. |
| 30 Oct | Change only the named gain | Isolate the intervention (reviewer) |
| 30 Oct | Replace "Kp too low" with "Overall gain too low" (common factor on Kp, Ki, Kd) | With only Kp lowered, no qualifying sluggish examples were found within the tested ranges and acceptance rules (all overshot). Linear check on 3 sampled motor instances (no delay, friction or quantisation): lowering all gains reduces closed-loop damping (ζ about 0.8 → 0.35 as the factor goes 1 → 0.15). This fits FEWER clean motor cases (13 in v1.0 train), not none; first-order stays about 0.7–0.8. (Corrected 30 Oct: an earlier chat claim that the motor supplies none was wrong.) |
| 30 Oct | Kd too high accepted for ripple only, or ripple + settling | Larger Kd often breaks other requirements too; a smaller class is accepted and reported with its own P/R/F1 |
| 30 Oct | Saturation: duration and %, judged against the plant's deadline; reachable targets only | Fair across fast and slow plants; unreachable setpoints are a different problem (excluded, documented) |
| 30 Oct | 5 stated requirements in every input; each fault must break its own | A student needs the criteria used to call a response faulty |
| 30 Oct | Hand-rule Kp/Ki split: overshoot > 45%, or overshoot broken with settling > 0.8× deadline → Ki | Set on dev train (median overshoot 69% for Ki vs 18% for Kp); checked on dev val |
| 30 Oct | Freeze protocol v1.0, then final seed 20261030; tests are not evaluated until the model is trained | Avoid adapting to the test sets |
| 30 Oct | **v1.1 revision; v1.0 final data withdrawn** | Second external review (verified by Claude): rounding hid violations in 10 train + 2 val records (+2 test_A, 2 test_B counted only); boundary wording "below" vs code "≤"; checker missed a corrupted ripple number and a corrupted diagnosis; "steady-state" and "low-pass filter" wording; motor statement out of date (13 clean motor gain_too_low examples exist) |
| 30 Oct | Reject examples whose printed numbers change which requirements are broken | Model and baselines only see the printed numbers |
| 30 Oct | Requirement wording "at most" | Matches the code: equality is acceptable |
| 30 Oct | Checker verifies diagnosis, fix, safety line, every evidence number vs printed input, printed vs stored violations; `--selftest` with 6 corruptions (all caught for the right reason) | Close the gaps the reviewer demonstrated |

## 6. Known limits and open questions

- Labels are injected faults with symptom checks; different gain settings can give similar symptoms. Kp-high vs Ki-high remain partly confusable.
- Training has no motor examples of kd_too_high and fewer motor examples of gain_too_low and saturation (see summary.txt). Plant mix per class is uneven.
- `check_dataset.py` checks consistency from stored numbers; it cannot detect a wrong simulator.
- All six plant families were explored during development; Test B is unseen by the model only. Test C and Test D are the only untouched tests.
- Dev (seed 0, v1.1) val macro F1: rules 0.986, tree 0.914.
- Close calls are kept as they are (no threshold changes during model development), e.g. train record 141: settles at 1.21 s against a 1.20 s deadline.
- Some violations are hairline: in v1.1 train, 15 of 221 overshoot or ripple violations are exactly one printed unit above the limit. Correct but close; a future version could require a margin.
- Second-order ripple limits can be large (e.g. 14.2%), because Gaussian measurement noise makes the nominal ripple large and the limit is 3× that.
- Token length: rough estimate for the longest example is about 365 tokens + about 50 for the instruction, so it may exceed 400. Measure with the real tokenizer and raise sequence_length to 512 if needed.
- *Historical, v1.0* final val (seed 20261030): rules 0.937, tree 0.935. **Current v1.1** final val (seed 20261031): rules 0.957, tree 0.957.
- English only. One fault at a time. Saturation from unreachable setpoints excluded.
- The QLabs inertia differs by 20% from the manual (cause not proven).

## 7. Files

`pid_doctor/`: `generate_dataset.py`, `check_dataset.py`, `rule_baseline.py`, `prepare_gemma_data.py`, `PROTOCOL.md` (v1.1-r1), `FROZEN_v1.2_r8.sha256` (current; local record, Colab zip stays r7), `FROZEN_v1.2_r7.sha256`, `FROZEN_v1.2_r6.sha256`, `FROZEN_v1.1_r5.sha256`, `FROZEN_v1.1_r4.sha256`, `FROZEN_v1.1_r3.sha256`, `FROZEN_v1.1_r2.sha256`, `FROZEN_v1.1_r1.sha256`, `FROZEN_v1.1.sha256` and `FROZEN_v1.0.sha256` (historical), `dev/` (v1.1, seed 0, development only, 662 records), `data/` (**current final: train/val seed 20261031, Test A/B seed 20261102**, 661 records, plus gemma_*.json and SPOTCHECK_sample.md), `archive/v0.1_seed0_DISCARDED/`, `archive/v1.0_seed20261030_WITHDRAWN/`, `archive/v1.1_tests_seed20261031_EXPOSED_baselines_computed/`, `make_v12.py`, `data_v12/` (run B answers), `pid_lib.py`, `smoke_test.py`, `test_section2.py`, `test_notebook_run.py`, `DATA_CARD.md`, `make_notebook.py`, `pid_doctor_lora.ipynb`, `CAPSTONE_LOG.md`.

## 8. Session log

- **29 Oct 2026:** Chose the project. Wrote the Stage 1 doc. Read Enoch's QLabs PROGRESS.md, the EEE576 report and the qube-servo3-python repo. Built the generator, baselines and v0.1 dataset (733).
- **30 Oct 2026:** An external reviewer audited v0.1; Claude verified every claim (all true) and found that over half of the saturation class was probably capacity-limited. Rebuilt the generator to v1.0 (named-gain-only, 5 requirements, per-example evidence, longer recordings, reachable saturation, "overall gain too low"). Wrote `check_dataset.py` (first run caught a one-sample recording shortfall and a double-rounding bug; both fixed; then PASS on 669 dev records). Manual audit of 14 dev examples. Rewrote the baseline parser (requirements vs observation split). Froze PROTOCOL v1.0 and generated the final data with seed 20261030. Second external review of v1.0 (verified): rounding hid violations, checker gaps, wording. Built v1.1 (display-consistency rejection, "at most", checker + self-test), dev check PASS (662) + self-test PASS, froze v1.1, withdrew the seed-20261030 data, generated final v1.1 data with seed 20261031. Third review: v1.1 accepted (30 examples, all checks reproduced, self-test errors inspected); documentation references corrected (v1.1-r1, no rule or data change). Built the notebook: verified the keras_hub 0.32 API (LoRA files must end in .lora.h5; generate stops at <end_of_turn>); a CPU smoke test with a tiny random Gemma 3 caught 2 bugs: (1) Keras ignored shuffle and train.json is grouped by class, so it is now shuffled with seed 812 every epoch; (2) fragile prompt stripping, now cut at <start_of_turn>model. Fourth review (verified): notebook `!` checks did not stop on failure -> now subprocess check=True; final baseline output now saved to results/FINAL_baselines.txt with check=True; smoke test only PRINTED losses -> now ASSERTS that LoRA matrices changed (8/8), base weights are frozen (24), and, with SMOKE_EXPECT_LEARNING=1, that the loss falls. Negative test (lr=0) fails as it should. Added Google Drive saving + automatic resume (checkpoint every epoch: LoRA, optimizer state, history; per-epoch seeded shuffle). The resume test (tiny random Gemma 3 on CPU only, not the real model) asserts a difference below 1e-5 between straight and interrupted+resumed runs (measured 0.0 in that test); its negative test (no optimizer restore, difference 0.05) is caught. The final test cell now runs only once (FINAL_RUN_STARTED.txt). Hashes now in FROZEN_v1.1_r3.sha256. Fifth review (verified by reproducing both): section 2 crashed with FileExistsError if a local results folder existed, and silently reused an old zip on Drive. Fixed: local results are copied to Drive before replacing; fresh code extraction; a version check (r3 hash file) stops with a clear message; `UPLOAD_NEW_ZIP` flag (the old zip is kept as a dated backup). `test_section2.py` runs 6 startup scenarios with stubs: all pass. Added course lab 8.3 outputs (same 5 val examples before/after, train+val prediction CSVs, sklearn classification_report with weighted P/R whose macro avg equals protocol macro F1, test CSVs and reports in the final run; PROTOCOL r4, hashes FROZEN_v1.1_r4.sha256). `test_notebook_run.py` EXECUTES the notebook's own cells (section 2 checks through section 7) with a tiny random Gemma 3: all outputs written, final-test lock refuses a second run; it caught a crash risk (watch indexes past the end of a split), fixed. **Sixth review (verified): that harness ran section 7 on the REAL seed-20261031 Test A/B, so rule_baseline.py (a subprocess) computed both baselines on the real test sets.** Scores were not opened; temp folders deleted unopened. Those test sets are withdrawn (archive/v1.1_tests_seed20261031_EXPOSED_baselines_computed/). Fresh Test A/B: `--test_seed 20261102` (126 / 98, no overlap; train/val byte-identical). The harness now uses dummy dev seed-0 test splits and asserts that the model and the subprocess read only those. Data Card corrected: v0.1 test scores WERE seen; reviews are AI-assisted (Claude + external AI-assisted reviewer); Enoch's own manual audit is still to do. PROTOCOL §13 (r5), hashes FROZEN_v1.1_r5.sha256. Training config declared in PROTOCOL §12 (r2, hashes in FROZEN_v1.1_r2.sha256, which now includes pid_lib.py).

- **30 Oct 2026 (evening):** Enoch's manual audit: 21/21 agree. Plan doc corrected (seventh review). Run 1 trained and analysed (eighth review; wording agreed, PROTOCOL §14 pre-registers runs A and B). Run A (lr 1e-4) started by Enoch on a fresh runtime. Meanwhile built run B: `make_v12.py` writes `data_v12/` (same inputs, labels and splits; answer = 5-line requirement checklist + time at limit, then the unchanged v1.1 answer); every checklist's broken set equals the stored violations (661/661); `check_dataset.py` verifies checklists, self-test 8/8 caught. Notebook gained `ANSWER_FORMAT` (v1.1/v1.2) and the `_v12` run-name suffix. Checks: notebook run test PASS in both formats (v1.2 seq_len 448), section-2 scenarios PASS (an r5 zip on Drive gives the clear UPLOAD_NEW_ZIP stop), fresh unzip ALL CHECKS PASSED. Hashes FROZEN_v1.2_r6.sha256.

- **30 Oct 2026 (night):** Run A results received and verified from results.zip (every number recomputed from the CSVs and history). One file in the zip, run 1's `lora_r4_lr5e-05_best.lora.h5`, failed its CRC check; run 1's identical epoch-6 checkpoint is intact and the Drive copy is unaffected. Run B lr fixed at 1e-4 before training (PROTOCOL r7, hashes FROZEN_v1.2_r7.sha256); notebook run test PASS (v1.2), section-2 scenarios PASS, fresh unzip checks PASS.

- **30 Oct 2026 (late):** External AI-assisted review of run B: every claim verified by recomputation and found true (baselines 57/60 on val; 50/60 val answers with all five checks right; 9 status errors in 8 answers; 1 missing ripple line; overshoot copied as 4.8% for 0%; false 'all met' 9 -> 3; wrong-diagnosis rows 15, 16, 22, 26, 42, 44, 52, 53). It corrected one detail of Claude's r8 note: of the '2 miscopied numbers', one was only a format difference (5% vs 5.0%). PROTOCOL r8 text fixed before any test scoring; diagnosis and checklist accuracy will be reported separately.

- **30 Oct 2026 (final review):** External AI-assisted review of the final test: all numbers verified true (Kp too high 5/15 on Test B; gain too low -> well tuned 8/15; 14 of 29 'all requirements met' answers wrong on Test B, 7 of 27 on Test A; Ki too low 100% on both). Accepted: describe zero-shot as 'under this prompt and output format'. Pushed back on one phrase: 'did not achieve the original goal of outperforming the baselines' — beating the baselines was never the stated goal (log §1 asks whether it works on unseen plants, student text and QLabs; log §4 already expected the tree to be strong on clean simulator text). Also nuanced: on Test A the model is statistically level with the hand rules (p 0.52); the tree is stronger there, and both baselines are clearly stronger on Test B.

- **30 Oct 2026 (correction):** The reviewer's reply was checked and is right on both points. (1) Claude's pushback was WRONG: the plan doc (Evaluation plan, current version, rev 30) still says "Success means the model beats both baselines where they are weakest: unseen plants and student-written text." Claude checked only the log and PROTOCOL, not the plan doc. By that stated criterion, the unseen-plant part was NOT met (Test B: both baselines clearly better); the student-text part is untested. The same plan doc also says no score target is fixed in advance, and its release-gate row mentions 'targets': an inconsistency to fix in the report. (2) 'Statistically level' was wrong wording: a large McNemar p does not show equality, and McNemar compares per-example correctness, not macro F1. Reproduced: Test A Gemma vs rules 9/13 p 0.523, vs tree 2/12 p 0.0129; Test B vs rules and tree 3/24 p 0.000049. Agreed wording: rules scored somewhat higher on Test A without a clear difference in the paired accuracy test; the tree was stronger on Test A; both baselines clearly stronger on Test B; student text, QLabs and the educational value of the explanations are open.

- **30 Oct 2026 (report + repo):** Final report drafted as a Claude Doc (https://claude.ai/artifact/GpxT5fRp7L3zbqUDrboW5R), every number checked against the result files; extra finding while writing: every training settling limit is 18 s or less, while 58 of 82 Test B settling lines have limits of 100 s or more. Repo folder `pid-tuning-doctor/` prepared and committed locally (code, data, data_v12, dev, final_results, protocol, log, README, MIT licence; archive/ left out); dataset regenerates byte-identically from seeds 20261031/20261102 (checked). Push waits for a GitHub connection. Deployment choice: Hugging Face Spaces with Gradio.

## 9. Next session

1. Write the final report (results tables, error analysis, limits, reflection). Update the plan doc and Data Card with final results.
3. Choose the best run and epoch on validation; only then unlock section 7 once.
4. Test C collection sheet (plots + consent form); Test D (QLabs); instructor review.
