# PID Tuning Doctor — Frozen protocol v1.1

Status: **v1.1 FROZEN on 30 Oct 2026** before the v1.1 final dataset was generated. (v1.0 was frozen earlier the same day, then revised after a second external review; see §10.)
Any change to anything below requires a new version number, a reason in CAPSTONE_LOG.md, and fresh test sets.

## 1. Task

Input: text describing one PID step response — plant, five stated requirements, gains, measurements.
Output: `Diagnosis: <class>. / Evidence: <...>. / Fix: <...> / safety line`.
The label is a **known injected fault that passed a symptom check**, not a proven diagnosis. Different gain settings can produce similar symptoms.

## 2. Classes and interventions (only the named change is made)

| Class | Intervention (relative to the plant's nominal controller) | Accepted only if the response breaks… |
|---|---|---|
| well_tuned | all gains × U(0.85, 1.15) | no requirement |
| kp_too_high | Kp × U(2.5, 8); Ki, Kd unchanged | overshoot and/or oscillation; NOT end error |
| gain_too_low (Overall gain too low, sluggish) | Kp, Ki, Kd × the same factor U(0.1, 0.45); derivative filter unchanged (Ti, Td unchanged) | settling ONLY; it must settle within the recording; rise > 2× the nominal rise |
| ki_too_low | Ki × U(0, 0.03); Kp, Kd unchanged | end error; NOT overshoot, oscillation or ripple |
| ki_too_high | Ki × U(4, 15); Kp, Kd unchanged | overshoot; NOT end error or ripple |
| kd_too_high | Kd × U(2, 10) (if nominal Kd = 0: base = 5·Kp·Ts); Kp, Ki unchanged | ripple only, or ripple + settling |
| actuator_saturation | gains nominal; actuator limit U(1.3·u_hold, 0.6·u_peak,nominal) | time at the limit > 0.2 × settling deadline AND settling and/or overshoot broken AND rise > 1.5× nominal; NOT ripple |

All classes except saturation require time at the limit < 0.1 × the settling deadline.
**Scope:** saturation is restricted to *reachable* targets (the limit is at least 1.3× the input needed to hold the target, computed from the plant model). Saturation caused by an unreachable setpoint is excluded.
A plant that cannot produce a clean example after 100 tries supplies fewer examples. Nothing is forced, and the shortfall is reported.

## 3. Requirements stated in every input

Derived from the nominal controller, which meets them all. Wording is **"at most"**: a value equal to the limit is acceptable, matching the strict ">" used to decide a violation.
- max overshoot = 15% if the nominal overshoot is ≥ 8%, else 10% or 15%
- max settling time (2%) = nominal settling × U(1.5, 2.2), 2 significant figures
- max oscillation cycles = 2
- max error at the end of the recording = 2%
- max control ripple = max(3 × nominal ripple, 0.3)% of full scale

## 4. Simulation and observation window

- **Training plants:** dc_motor (QUBE-Servo 3, QLabs-calibrated, randomised), first_order (+4–10% lag), second_order (+lag). **Test B plants:** tank (EEE576 nonlinear + 5–15 s valve lag), heater (FOPDT), pendulum_arm (nonlinear Furuta, hanging).
- **Measurement:** encoder quantisation (2048 counts/rev) on dc_motor and pendulum_arm only. Gaussian noise of 0.3% of the step on the other four plants. A constant input bias on all plants.
- **Controller:** discrete PID, derivative on measurement with first-order filter N = 0.2/Ts, no anti-windup, actuator clamp.
- **Recording length** = max(30/ω_design, 3 × settling deadline). "Error at the end of the recording" is the wording used; steady state is not claimed.
- **Displayed-number consistency (v1.1):** an example is rejected unless the requirements broken according to the *printed* (rounded) numbers equal those according to the stored numbers.

## 5. Splits

| Split | Plants | Per class (target) | Use |
|---|---|---|---|
| train | training plants | 60 | fine-tuning; decision-tree baseline fit |
| val | training plants | 10 | checkpoint selection and all development decisions |
| test_A | training plants | 20 | final evaluation, run once |
| test_B | unseen plants | 15 (5 per plant) | final evaluation, run once |
| test_C | student-written descriptions | ~50–100 | final evaluation, run once (collected separately) |
| test_D | QLabs virtual Qube-Servo 3 | ~30 | final evaluation, run once |

The v1.1 final dataset uses **seed 20261031 for train and val**, and **test seed 20261102 for Test A and Test B** (r5, see §13). Seed 20261030 (v1.0 final) is withdrawn: its Test A and Test B were never scored (its validation scores were calculated and are recorded as historical v1.0 results). Seed 0 is development-only (folder `dev/`).

## 6. Metrics

Primary: **macro F1** over the classes present. Also reported: accuracy, per-class precision, recall, F1 with class sizes, and the confusion matrix. Diagnosis is extracted from the generated `Diagnosis:` line. Unparseable output counts as wrong.
Token-level training accuracy is NOT a task metric.

## 7. Model selection

- **Checkpoint:** chosen by **validation macro F1** (val prompts generated without reference answers), with **validation loss** as the tie-breaker. Fixed in advance.
- The same 3 validation examples are printed after every epoch for inspection.
- Before training, the tokenised length of every full training sequence (prompt + response) is measured with the real Gemma tokenizer and must fit in `sequence_length`.

## 8. Baselines (frozen)

`rule_baseline.py` at freeze time: hand rules (comparing each measurement with its requirement) and a depth-6 decision tree trained on train only. The Kp/Ki split rule (overshoot > 45%, or overshoot broken with settling > 0.8 × deadline → Ki too high) was set on dev train and checked on dev val (seed 0) before freezing. They are evaluated on the test sets once, alongside the model.

**Dev results (seed 0, dev val only, macro F1):** v1.1 (current): rules 0.986, tree 0.914. *Historical, v1.0:* rules 0.905, tree 0.950. **v1.1 final val (seed 20261031):** rules 0.957, tree 0.957. *Historical, v1.0 final val (seed 20261030):* rules 0.937, tree 0.935. On clean simulator text the tree is already strong, so the main case for the LLM rests on Test C (free-text student descriptions), unseen phrasing, and the generated evidence and fix. This was stated before any final test result was seen.

## 9. Development disclosures

- **Dataset v0.1 (29 Oct, seed 0):** Test A/B baseline results were seen, and the Kd rule was changed afterwards. v0.1 is discarded.
- **All six plant families, including the Test B plants,** were explored while designing the symptom checks (delays, saturation rule). Test B is unseen by the *model*, not by the *developer*. Only Test C and Test D are untouched by development.
- **Development checks** (`dev/`, seed 0): `check_dataset.py` (v1.1: PASS on all 662 records in the current `dev/`; v1.0 dev had 669) plus baselines on **dev train/val only**. Dev test_A/test_B were never evaluated.
- `check_dataset.py` recomputes everything from stored numbers. It does not re-run the simulation, so a wrong simulator would not be caught by it. Manual audits (2 per class) cover that partly.
- An external reviewer audited v0.1 and found: label/intervention mismatch, fixed evidence phrases, a baseline parser bug (186 train records), recordings shorter than the deadline (36), and an unconditional saturation fix. All are addressed in v1.0.

## 10. Revision v1.0 → v1.1 (30 Oct 2026)

A second external review of v1.0 (30 records read, hashes verified, no test performance computed) found the following. Claude verified each one.
1. **Rounding hid violations:** in 10 train + 2 val records (and 2 test_A, 2 test_B, counted only), the requirements broken according to the printed numbers differed from the stored ones (e.g. overshoot 15.4% printed as "15% against a 15% limit"). Fix: displayed-number consistency rule (§4).
2. **Boundary wording:** "below"/"less than" implied strict inequality while the code accepted equality. Fix: "at most".
3. **Checker gaps:** a corrupted evidence number (ripple 999%) and a corrupted diagnosis both passed. Fix: the checker now verifies the Diagnosis line against the label, the Fix line, the safety line, every evidence number against the printed input, and printed-vs-stored violations. `check_dataset.py --selftest` corrupts real records and must catch every corruption.
4. **Wording:** "Ki too low (steady-state error)" → "(remaining error)". The Kd fix now reads "Reduce Kd or strengthen the derivative filtering, then re-test" (the derivative is already filtered).
5. **Motor statement corrected:** the v1.0 final train set contains 13 clean motor gain_too_low examples. The earlier claim that the motor supplies none was wrong. The damping analysis (3 sampled motor instances, linear model, no delay, friction or quantisation) shows damping falls as the common factor falls, which fits *fewer* clean motor cases, not none.

Classes, interventions, thresholds and the evaluation plan are unchanged.

## 11. Documentation-only correction v1.1-r1 (30 Oct 2026)

After a third review, these text references in this file and the log were brought up to date: the §8 dev scores are now labelled by version, the §9 dev record count, and "never evaluated" is now "Test A and Test B never scored". **No rule, threshold, code or data changed.** `FROZEN_v1.1.sha256` is kept unchanged as the record of the freeze. `FROZEN_v1.1_r1.sha256` holds the same code hashes plus the hash of this corrected file.

## 12. Training configuration, declared before any training (v1.1-r2, 30 Oct 2026)

Added before the first fine-tuning run. Documentation addition only: no rule, threshold, code or data changed.
- Model: Keras `Gemma3CausalLM.from_preset("gemma3_1b")` (as in course lab 5.6), sampler greedy.
- LoRA rank 4 (lab 5.6). Optimizer Adam, **learning rate 5e-5**, batch size 1, **up to 6 epochs**. The training set is shuffled with seed 812 before every epoch (Keras ignores `shuffle` for this input, and train.json is grouped by class).
- `sequence_length` = smallest multiple of 32 ≥ (longest real tokenised prompt + response + 1 + 16), measured with the Gemma tokenizer on train/val.
- **Zero-shot baseline:** the same pre-trained model, before LoRA, scored on val (and on the tests in the final run).
- **Optional sensitivity runs:** learning rate 2e-5 or 1e-4, judged on **validation only**. The test sets are scored once, for the single checkpoint chosen by §7.
- Code: `pid_lib.py` (tested on CPU by `smoke_test.py` with a tiny random Gemma 3: fit, evaluate, generate, parsing, checkpoint selection, LoRA save/load).
- (r3, 30 Oct 2026, before any training) **Checkpoint and resume:** after every epoch, the latest LoRA weights, the optimizer state and the history are saved to Google Drive. A disconnected run resumes after the last finished epoch. The shuffle order depends only on (seed 812, epoch), so a resumed run is identical to an uninterrupted one: `smoke_test.py` asserts zero difference, and a negative test (optimizer not restored) is detected. The final test cell writes `FINAL_RUN_STARTED.txt` and refuses to run a second time. No change to the training settings.
- (r4, 30 Oct 2026, before any training) **Course lab 8.3 outputs added** (evaluation only, no change to training or selection): the same 5 validation examples shown before and after fine-tuning; predictions on train (seen) and val (unseen) saved as CSV; sklearn `classification_report` with weighted precision and recall, using the classes present in the true labels so its macro average equals §6 macro F1 (checked on 200 random cases); test-set CSVs and reports in the final run. Training-set scores are for the seen-vs-unseen comparison only and play no part in checkpoint selection.

## 13. Test exposure and fresh test sets (v1.1-r5, 30 Oct 2026, before any training)

- **What happened:** Claude's CPU harness `test_notebook_run.py` (r4) unlocked section 7 in a throwaway copy that contained the REAL seed-20261031 Test A/B files. The notebook's `rule_baseline.py` subprocess read those files, so **both baselines (rules and tree) were computed on the real Test A and Test B**. The scores went to a temporary file that Claude did not open (its output filter hid them), and the temporary folders were then deleted unopened. No model (tiny or real) was scored on the real test sets: the harness's tiny model saw only an 8-record subset of each split.
- **Classification:** computed but not viewed still counts as exposure. The seed-20261031 Test A/B are **withdrawn** (archive/v1.1_tests_seed20261031_EXPOSED_baselines_computed/).
- **Fresh test sets:** `generate_dataset.py --seed 20261031 --test_seed 20261102`. Train and val are byte-for-byte identical to before (verified by comparing the files). Test A = 126, Test B = 98, no overlap with the withdrawn sets. Only `check_dataset.py` (PASS) and class counts were computed on them.
- **Harness fixed:** it now replaces Test A/B in its throwaway copy with the development-only seed-0 test splits (never scored, not part of the final data), and asserts that both the model and the baseline subprocess read only those dummy files.
- **Earlier exposure, restated:** v0.1 Test A/B baseline scores WERE seen (29 Oct) and influenced the Kd rule. v1.0 Test A/B were never scored.

## 14. Run 1 result and pre-registered follow-up runs (30 Oct 2026, before running them)

**Run 1** (lr 5e-5, rank 4, 6 epochs, seq_len 384 measured; T4, 29.6 min). Best epoch 6: **val macro F1 0.397, accuracy 0.45 (27/60)**; train macro F1 0.480, accuracy 0.541. Zero-shot val macro F1 0.000 (60/60 unparseable). Baselines on val: 0.957 / 0.957. Run 1 val unparseable: 1/60. Kd too high was never predicted (train or val). False threshold claims in generated evidence (pattern "X against a Y limit" with X within the limit): train 77 overshoot (65 below, 12 equal) + 12 ripple; val 9 overshoot + 1 ripple. ki_too_high predicted 123/377 times on train. Run 1 shows checkpoints are saved; it did not test recovery from a real disconnect. Test sets not scored.

Agreed wording: *In this run, LoRA substantially improved output-format compliance, but generated diagnosis accuracy remained far below both baselines, with frequent invalid threshold comparisons and no predictions of the Kd-too-high class.* Not claimed: absence of overfitting (train–val gap; 60 val examples), the cause of the comparison errors, or that diagnosis-first ordering causes errors.

**Run A** (allowed by §12): learning rate **1e-4**, everything else as run 1. Starts from a **fresh pretrained Gemma and a fresh LoRA adapter** (new Colab runtime), separate RUN_NAME and checkpoint directory (`lora_r4_lr0.0001`). Uses the current zip unchanged.

**Run B** (v1.2, separate experiment, declared now, built after run A starts): **checklist-first answers**. Each answer first states each of the 5 requirements as broken or met with the comparison (e.g. `overshoot 34% > 15% limit: broken`), then Diagnosis, Evidence and Fix. Inputs, labels and splits stay fixed; only the target answer text changes. The output checks (`check_dataset.py`, `pid_lib` parsing) are updated and tested, and token lengths are remeasured because answers get longer. lr 5e-5 unless run A shows 1e-4 is clearly better on validation (decided before run B starts).

**Selection across runs:** the run and epoch with the best **validation** diagnosis macro F1 (val loss as the tie-breaker) is the one scored once on the tests. Choosing among several runs on 60 validation examples can flatter the validation score; the report will say so, and the test score is the unbiased one.
- (r6, 30 Oct 2026, built while run A trains, before any run B training) **Run B data (v1.2):** `make_v12.py` builds `data_v12/` from `data/` by putting a checklist before the v1.1 answer: one line per requirement with the printed value, the limit and broken/met, plus the time at the actuator limit (information only). Inputs, labels, splits and the v1.1 Diagnosis/Evidence/Fix lines are unchanged (verified for all 661 records); every checklist's broken set equals the stored violations (asserted while building). `check_dataset.py` checks the checklist against the printed input; the self-test gains 2 checklist corruptions (8/8 caught). The notebook has `ANSWER_FORMAT` ("v1.1" or "v1.2"); run B's RUN_NAME ends in `_v12`. Hashes: FROZEN_v1.2_r6.sha256 (adds make_v12.py).
- (r7, 30 Oct 2026, after run A, BEFORE any run B training) **Run A result and run B learning rate.** Run A (lr 1e-4, fresh model and adapter, T4, 31.0 min): best epoch 6, **val macro F1 0.556, accuracy 0.633 (38/60)**, 0 unparseable; train macro F1 0.687, accuracy 0.732. Per epoch, val F1 1e-4 vs 5e-5: 0.097/0.064, 0.273/0.181, 0.373/0.291, 0.413/0.275, 0.322/0.280, 0.556/0.397; val loss lower at every epoch. Paired on the 60 val examples (epoch 6 vs epoch 6): 15 right only in run A, 4 right only in run 1 (exact McNemar p = 0.019). Caveats: one seed per run; run A's val F1 moved 0.413 -> 0.322 -> 0.556 between epochs 4-6, a swing as large as the gap between runs. Still far below the baselines (0.957). Kd too high: 0/5 val, 5/30 train. False "all requirements met" claims: val 9 of 16, train 29 of 71 (mostly Kd too high and saturation). **Decision: run B uses lr 1e-4**, judged clearly better on validation by the consistency above, and it keeps runs A and B different in only the answer format. Test sets not scored.
- (r8, 30 Oct 2026, after run B, BEFORE any test scoring) **Run B result and final selection.** Run B (v1.2 checklist-first, lr 1e-4, seq_len 480 measured, T4, 35.3 min, one kernel session): best epoch 6, **val macro F1 0.861, accuracy 0.867 (52/60)**, 0 unparseable; train macro F1 0.959, accuracy 0.958. Val F1 by epoch 0.282, 0.349, 0.589, 0.625, 0.613, 0.861 (the last-epoch jump is as large as earlier swings). Paired on val vs run A epoch 6: 16 right only in run B, 2 right only in run A (exact McNemar p = 0.0013). Recomputed from the prediction CSVs; best weights byte-identical to the epoch-6 checkpoint. **Checklist errors** (generated vs the correct checklist from the printed input): every requirement line is correct except settling and ripple, and every status error is "met" where it should be "broken" — settling broken: train 42/170 wrong, val 6/26; ripple broken: train 7/51, val 3/8; no false "broken" (val: 1 ripple line missing; 1 miscopied number, overshoot 4.8% for 0%; one more line differs only in format, 5% for 5.0%). Counting numerically equivalent formats as correct, 50/60 val answers have all five checks right; the 9 status errors fall in 8 answers. False "all requirements met" on val: run A 9, run B 3. Checklist broken-set correct: train 328/377, val 52/60. Still below the baselines on val (0.957). **Selection (§14): run B, epoch 6** (`lora_r4_lr0.0001_v12_best.lora.h5`) is the single checkpoint scored once on Test A/B. Zero-shot on val was generated in run 1 (seq_len 384; inputs identical; 60/60 unparseable); the zero-shot tests are generated in the final run at seq_len 480. The report gives diagnosis accuracy and checklist (explanation) accuracy separately, because some answers have the right diagnosis with a wrong checklist and vice versa. No further training runs before the test.
- (r9, 30 Oct 2026) **Final test, scored once** (FINAL_RUN_STARTED 2026-10-30 19:53 UTC, run lora_r4_lr0.0001_v12, epoch 6). Fine-tuned: **Test A macro F1 0.894, accuracy 0.897 (113/126); Test B macro F1 0.693, accuracy 0.704 (69/98)**; 0 unparseable. Zero-shot: 0.000 on both (all unparseable, seq_len 480). Baselines (reproduced exactly offline): rules A 0.934 / B 0.921, tree A 0.978 / B 0.918. Paired accuracy (McNemar, per-example correctness, not macro F1): Test A vs rules 9 right only for the model, 13 only for rules (p = 0.52: no clear difference detected, which does not show equivalence); vs tree 2 vs 12 (p = 0.013); Test B vs rules and vs tree 3 vs 24 (p < 0.001). Test B by plant (model / rules): heater 24/35 vs 32/35, pendulum arm 22/29 vs 29/29, tank 23/34 vs 29/34. Checklist: all five checks right A 102/126, B 57/98; errors are mainly settling "met" where it should be "broken" (A 20, B 29; tank 0/11 settling-broken lines right) and ripple (A 6, B 12). False "all requirements met": A 7, B 14. Main confusions: overall gain too low -> well tuned (A 3, B 8), Kd too high -> well tuned (A 4, B 4), Kp/Ki too high mixed up (A 5, B 8). Recomputed from FINAL_predictions_*.csv; matches the notebook print-out.
