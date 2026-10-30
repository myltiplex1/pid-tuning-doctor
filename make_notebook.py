"""Builds pid_doctor_lora.ipynb (run once: python make_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# PID Tuning Doctor — fine-tune Gemma 3 1B with LoRA

Capstone, Stage 3. Follows course lab 5.6 (Keras + Gemma 3 + LoRA) and the frozen rules in `PROTOCOL.md` v1.1.

**What this notebook does**
1. Checks the data and the frozen file hashes.
2. Measures the **real token length** of every training example with the Gemma tokenizer and sets `sequence_length`.
3. Scores the **pre-trained Gemma (zero-shot)** on validation.
4. Fine-tunes with **LoRA (rank 4, lr 5e-5, up to 6 epochs)**. After each epoch: validation loss, validation **macro F1 from generated answers**, and the same 3 validation examples printed. The best epoch (val macro F1, val loss as the tie-breaker) is saved.
5. **Final test section: locked.** Run it **once**, only after you are finished with every choice above.

**Runtime:** choose **T4 GPU** (Runtime → Change runtime type). Expect roughly 5–15 min per epoch, depending on Colab. You need the **Kaggle secrets** set up exactly as in lab 5.6.

**Google Drive + automatic resume**
- Everything important is saved to **My Drive / pid_doctor / results**: zero-shot scores, a checkpoint after **every** epoch (LoRA weights + optimizer state + history), the best LoRA weights, and the final results.
- **If Colab disconnects:** reconnect (choose T4 again if asked), then **Runtime → Run all**. It finds the zip on Drive (no re-upload), skips the zero-shot step if already done, and **resumes training after the last finished epoch**. Only the epoch that was running is lost. Resuming was tested to give exactly the same weights as an uninterrupted run.
- Section 7 stays locked, so **Run all** is safe.""")

md("## 1. Install and set up (as in lab 5.6)")
code("""%%capture
!pip install "git+https://github.com/google-deepmind/ai-foundations.git@main"
!pip install -q sentencepiece""")
code("""import os, json, time, zipfile
from google.colab import userdata, files
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import keras, keras_hub
keras.utils.set_random_seed(812)
print("keras", keras.__version__, "| keras_hub", keras_hub.__version__)
assert hasattr(keras_hub.models.Backbone, "save_lora_weights"), (
    "keras_hub is too old for save_lora_weights: run  !pip install -q -U keras-hub  then Runtime > Restart session")
!nvidia-smi --query-gpu=name,memory.total --format=csv""")

md("## 2. Connect Google Drive, get the project zip, verify it\nThe first time, a **Choose Files** button appears: upload **`pid_doctor_v1.1.zip`**. It is copied to your Drive, so after a disconnect you will not be asked again. Allow Drive access when Colab asks.\n\n**When Claude sends you a NEW zip:** set `UPLOAD_NEW_ZIP = True` in the cell below, run it, upload the new zip, then set it back to `False`. If you forget, the cell stops with a clear message.")
code("""# Set to True ONLY when Claude sends you a NEW pid_doctor_v1.1.zip: the old
# copy on Drive is kept as a dated backup and the new one is used from then on.
UPLOAD_NEW_ZIP = False
EXPECTED_HASH_FILE = "FROZEN_v1.2_r7.sha256"   # this notebook needs the zip that contains this file

import shutil
from google.colab import drive
drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/pid_doctor"
os.makedirs(f"{DRIVE}/results", exist_ok=True)
ZIP_ON_DRIVE = f"{DRIVE}/pid_doctor_v1.1.zip"
stamp = time.strftime("%Y%m%d_%H%M%S")

# 1. the project zip: upload the first time (or when replacing), otherwise reuse the Drive copy
if UPLOAD_NEW_ZIP or not os.path.exists(ZIP_ON_DRIVE):
    up = files.upload()
    zname = [n for n in up if n.endswith(".zip")][0]
    if os.path.exists(ZIP_ON_DRIVE):
        shutil.move(ZIP_ON_DRIVE, f"{DRIVE}/pid_doctor_v1.1_old_{stamp}.zip")
        print("old zip kept as backup:", f"pid_doctor_v1.1_old_{stamp}.zip")
    shutil.copy(zname, ZIP_ON_DRIVE)
    print("zip saved to Drive:", ZIP_ON_DRIVE)
else:
    print("using the zip already on Drive:", ZIP_ON_DRIVE)

# 2. a 'results' folder left from an earlier run in THIS runtime: never lose its files
local = "/content/pid_doctor/results"
if os.path.islink(local):
    os.unlink(local)
elif os.path.isdir(local):
    if os.listdir(local):
        backup = f"{DRIVE}/results/local_backup_{stamp}"
        shutil.copytree(local, backup)
        print("found local results from an earlier run -> copied to Drive:", backup)
    shutil.rmtree(local)

# 3. fresh copy of the code (no stale files from an older version), then check its version
if os.path.isdir("/content/pid_doctor"):
    shutil.rmtree("/content/pid_doctor")
with zipfile.ZipFile(ZIP_ON_DRIVE) as z:
    z.extractall("/content")
if not os.path.exists(f"/content/pid_doctor/{EXPECTED_HASH_FILE}"):
    raise RuntimeError("The zip on your Drive is an OLDER version than this notebook. "
                       "Set UPLOAD_NEW_ZIP = True at the top of this cell, run it again, "
                       "and upload the latest pid_doctor_v1.1.zip.")
os.chdir("/content/pid_doctor")

# 4. every 'results/...' path in this notebook is really on Google Drive
os.symlink(f"{DRIVE}/results", "results")
print("working in", os.getcwd(), "| results ->", os.path.realpath("results"))""")
code("""# frozen code + protocol must be unchanged; data must pass every check.
# check=True stops the notebook with an error if ANY check fails.
import subprocess, sys
for command in [
    ["sha256sum", "-c", "FROZEN_v1.2_r7.sha256"],
    [sys.executable, "check_dataset.py", "data"],
    [sys.executable, "check_dataset.py", "--selftest", "data"],
    [sys.executable, "check_dataset.py", "data_v12"],
    [sys.executable, "check_dataset.py", "--selftest", "data_v12"],
]:
    subprocess.run(command, check=True)
print("ALL CHECKS PASSED")""")
code("""# Answer format: "v1.1" = run 1 and run A; "v1.2" = run B (checklist first, PROTOCOL §14).
# Inputs, labels and splits are identical in both; only the target answers differ.
ANSWER_FORMAT = "v1.1"

import sys; sys.path.insert(0, "/content/pid_doctor")
import pid_lib as L
DATA = {"v1.1": "data", "v1.2": "data_v12"}[ANSWER_FORMAT]
print("answer format", ANSWER_FORMAT, "| data folder:", DATA)
train, val = L.load_split(DATA, "train"), L.load_split(DATA, "val")
from collections import Counter
print("train", len(train["prompts"]), dict(Counter(train["labels"])))
print("val  ", len(val["prompts"]), dict(Counter(val["labels"])))
print(train["prompts"][0] + train["responses"][0])""")

md("## 3. Load Gemma 3 1B and measure real token lengths")
code("""model = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_1b")
tok = model.preprocessor.tokenizer
lens = L.token_lengths(tok, train["prompts"] + val["prompts"], train["responses"] + val["responses"])
SEQ_LEN = L.choose_sequence_length(lens)
print(f"token length: min {min(lens)}, median {sorted(lens)[len(lens)//2]}, max {max(lens)}")
print(f"course default was 400 -> fits: {max(lens) + 1 <= 400}.  Using sequence_length = {SEQ_LEN}")
model.preprocessor.sequence_length = SEQ_LEN
# prompts alone must leave room for an answer of ~100 tokens
p_lens = [len(tok(p)) for p in val["prompts"]]
print("longest val prompt:", max(p_lens), "tokens; room for answer:", SEQ_LEN - max(p_lens))""")

md("## 4. Zero-shot baseline: pre-trained Gemma, before LoRA (validation only)")
code("""model.compile(sampler="greedy")
if os.path.exists("results/zeroshot_val.json"):          # already done before a disconnect
    _z = json.load(open("results/zeroshot_val.json"))
    zs, zs_answers, zs_preds = _z["scores"], _z["answers"], _z["preds"]
    print("zero-shot already done (loaded from Drive)")
    L.print_scores("ZERO-SHOT val", zs)
else:
    t0 = time.time()
    zs, zs_answers, zs_preds = L.evaluate_split(model, val, SEQ_LEN, batch_size=8)
    L.print_scores("ZERO-SHOT val", zs)
    print(f"({time.time() - t0:.0f} s)")
    json.dump(dict(scores=zs, answers=zs_answers, preds=zs_preds, seq_len=SEQ_LEN),
              open("results/zeroshot_val.json", "w"), indent=1)
    L.save_predictions_csv("results/predictions_val_zeroshot.csv", val, zs_answers, zs_preds)

# lab 8.3: the SAME 5 validation examples are shown again after fine-tuning (section 6b)
SPOT = L.spot_indices(len(val["prompts"]))
print("spot-check examples (val indices):", SPOT)
for i in SPOT:
    print(f"\\n--- val {i} | expected: {val['labels'][i]} | zero-shot predicted: {zs_preds[i]}\\n{zs_answers[i][:400]}")""")

md("## 5. LoRA fine-tuning with validation-based checkpoint selection\nSettings are fixed in `PROTOCOL.md` §12. Change them only for an optional sensitivity run (lr 2e-5 or 1e-4), judged on **validation only**.")
code("""LR, RANK, EPOCHS = 5e-5, 4, 6          # PROTOCOL §12
RUN_NAME = f"lora_r{RANK}_lr{LR:g}" + ("_v12" if ANSWER_FORMAT == "v1.2" else "")
LORA_PATH = f"results/{RUN_NAME}_best.lora.h5"

model.backbone.enable_lora(rank=RANK)
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=LR),
              weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
              sampler="greedy")
model.summary()""")
code("""# A checkpoint is saved to Drive after every epoch; re-running this cell
# after a disconnect RESUMES after the last finished epoch.
t0 = time.time()
history, best = L.train_with_selection(model, train, val, epochs=EPOCHS, lora_path=LORA_PATH,
                                       max_length=SEQ_LEN, watch_idx=(0, 20, 40), gen_batch=8,
                                       ckpt_dir=f"results/{RUN_NAME}_checkpoints")
print(f"\\ntotal {(time.time() - t0) / 60:.1f} min\\nBEST:", best)
json.dump(dict(config=dict(lr=LR, rank=RANK, epochs=EPOCHS, seq_len=SEQ_LEN), history=history, best=best),
          open(f"results/{RUN_NAME}_history.json", "w"), indent=1)""")
code("""import matplotlib.pyplot as plt
ep = [h["epoch"] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train loss")
ax[0].plot(ep, [h["val_loss"] for h in history], "o-", label="val loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(ep, [h["macro_f1"] for h in history], "o-"); ax[1].set_ylim(0, 1)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val macro F1 (generated diagnosis)")
ax[1].axhline(0.957, ls="--", c="gray"); ax[1].text(ep[0], 0.93, "baselines on val (0.957)", fontsize=8)
plt.tight_layout(); plt.savefig(f"results/{RUN_NAME}_curves.png", dpi=150); plt.show()""")

md("## 6. Reload the best checkpoint and inspect it on validation")
code("""model.backbone.load_lora_weights(LORA_PATH)
vs, v_answers, v_preds = L.evaluate_split(model, val, SEQ_LEN, batch_size=8)
L.print_scores(f"BEST CHECKPOINT (epoch {best['epoch']}) val", vs)
for i, (t, p) in enumerate(zip(val["labels"], v_preds)):
    if t != p:
        print(f"\\n--- val {i}: true {t}, predicted {p}\\n{v_answers[i]}")""")

md("""## 6b. Course lab 8.3 evaluation: spot checks, seen vs unseen, prediction files

Uses the best checkpoint loaded in section 6. Writes `predictions_train.csv` and `predictions_val.csv` (input, true label, model output, predicted class) to Drive, and prints sklearn's `classification_report` with weighted precision and recall, as lab 8.3 asks. Its macro average equals the protocol's macro F1. The training-set predictions take roughly 15–30 minutes and are saved, so after a disconnect they are not redone.""")
code("""# 1. the same 5 validation examples: before (zero-shot) vs after (fine-tuned)
for i in SPOT:
    print(f"=== val {i} | expected: {val['labels'][i]}")
    print(f"BEFORE (zero-shot) -> {zs_preds[i]}:\\n{zs_answers[i][:300]}")
    print(f"AFTER (fine-tuned, epoch {best['epoch']}) -> {v_preds[i]}:\\n{v_answers[i]}\\n")""")
code("""# 2. predictions on SEEN (train) and UNSEEN (val) data, saved as CSV on Drive
tag = f"results/{RUN_NAME}_epoch{best['epoch']}"
L.save_predictions_csv(f"{tag}_predictions_val.csv", val, v_answers, v_preds)
if os.path.exists(f"{tag}_predictions_train.json"):        # already done before a disconnect
    _t = json.load(open(f"{tag}_predictions_train.json")); t_answers, t_preds = _t["answers"], _t["preds"]
    print("train predictions loaded from Drive")
else:
    t0 = time.time()
    _, t_answers, t_preds = L.evaluate_split(model, train, SEQ_LEN, batch_size=8)
    json.dump(dict(answers=t_answers, preds=t_preds), open(f"{tag}_predictions_train.json", "w"))
    print(f"train predictions: {(time.time() - t0) / 60:.1f} min")
L.save_predictions_csv(f"{tag}_predictions_train.csv", train, t_answers, t_preds)

# 3. classification reports and the seen-vs-unseen summary
reports = {"train (seen)": L.course_report(train["labels"], t_preds),
           "val (unseen)": L.course_report(val["labels"], v_preds)}
for name, r in reports.items():
    print(f"\\n===== {name} =====\\n{r['report']}")
print(f"{'':14s}{'accuracy':>10s}{'precision(w)':>14s}{'recall(w)':>11s}{'macro F1':>10s}")
for name, r in reports.items():
    print(f"{name:14s}{r['accuracy']:10.3f}{r['precision_weighted']:14.3f}{r['recall_weighted']:11.3f}{r['macro_f1']:10.3f}")
json.dump({k: {m: v for m, v in r.items()} for k, r in reports.items()},
          open(f"{tag}_course_reports.json", "w"), indent=1)""")

md("""## 7. FINAL TEST — run ONCE

Only run this after **every** choice above is final. PROTOCOL §5–§7: the test sets are scored once, for the single checkpoint chosen on validation, alongside the zero-shot model and the two baselines. **Do not change anything and re-run after seeing these numbers.** If you must (for example after a crash), say so in the report.

Set `RUN_FINAL_TESTS = True` to unlock.""")
code("""RUN_FINAL_TESTS = False

if RUN_FINAL_TESTS and os.path.exists("results/FINAL_RUN_STARTED.txt"):
    print("A final test run was already started on", open("results/FINAL_RUN_STARTED.txt").read(),
          "\\nNot running again (PROTOCOL: test sets are scored once). If it crashed part-way, tell Claude;"
          " any re-run must be reported.")
elif RUN_FINAL_TESTS:
    import subprocess, sys
    with open("results/FINAL_RUN_STARTED.txt", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" | run {RUN_NAME} | best epoch {best['epoch']}")
    out = {"checkpoint": best, "run": RUN_NAME}
    for split in ["test_A", "test_B"]:
        t = L.load_split(DATA, split)
        s, ans, pr = L.evaluate_split(model, t, SEQ_LEN, batch_size=8)
        L.print_scores(f"FINE-TUNED {split}", s)
        out[split] = dict(scores=s, answers=ans, preds=pr, labels=t["labels"], plants=t["plants"])
        L.save_predictions_csv(f"results/FINAL_predictions_{split}.csv", t, ans, pr)
        cr = L.course_report(t["labels"], pr)
        out[split]["course_report"] = cr
        print(cr["report"])
    json.dump(out, open("results/FINAL_test_results.json", "w"), indent=1)   # save before freeing memory
    # zero-shot on the tests: a fresh copy of the base model (free the fine-tuned one first: T4 has 16 GB)
    import gc; del model; gc.collect()
    zs_model = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_1b")
    zs_model.preprocessor.sequence_length = SEQ_LEN
    zs_model.compile(sampler="greedy")
    for split in ["test_A", "test_B"]:
        t = L.load_split(DATA, split)
        s, ans, pr = L.evaluate_split(zs_model, t, SEQ_LEN, batch_size=8)
        L.print_scores(f"ZERO-SHOT {split}", s)
        out[f"zeroshot_{split}"] = dict(scores=s, preds=pr)
    result = subprocess.run(
        [sys.executable, "rule_baseline.py", "data/test_A.json", "data/test_B.json"],
        capture_output=True, text=True, check=True)
    print(result.stdout)
    with open("results/FINAL_baselines.txt", "w") as f:
        f.write(result.stdout)
    json.dump(out, open("results/FINAL_test_results.json", "w"), indent=1)
    print("Saved results/FINAL_test_results.json")
    # seen vs unseen summary, as in lab 8.3 (train/val from section 6b)
    rows = dict(reports, **{f"{sp} (test)": out[sp]["course_report"] for sp in ["test_A", "test_B"]})
    print(f"\\n{'':16s}{'accuracy':>10s}{'precision(w)':>14s}{'recall(w)':>11s}{'macro F1':>10s}")
    for name, r in rows.items():
        print(f"{name:16s}{r['accuracy']:10.3f}{r['precision_weighted']:14.3f}{r['recall_weighted']:11.3f}{r['macro_f1']:10.3f}")
else:
    print("Final tests are locked. Set RUN_FINAL_TESTS = True only when model selection is finished.")""")

md("## 8. Download the results\nEverything is already on your Drive (My Drive / pid_doctor / results). This also makes `results.zip` (without the bulky optimizer files) to send to Claude.")
code("""!zip -qr results.zip results -x "*_optimizer.npz"
files.download("results.zip")""")

nb["cells"] = C
nb["metadata"] = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                  "kernelspec": {"display_name": "Python 3", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "pid_doctor_lora.ipynb")
print("wrote pid_doctor_lora.ipynb with", len(C), "cells")
