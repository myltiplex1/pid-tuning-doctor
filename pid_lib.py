"""Shared training and evaluation code for the PID Tuning Doctor notebook.

The Colab notebook calls these functions, and so does a CPU smoke test
(`smoke_test.py`) with a tiny random Gemma 3, so the logic that runs in
Colab is the logic that was tested. Rules come from PROTOCOL.md §6–§7:
  * task metric = macro F1 of the diagnosis extracted from GENERATED text
    (prompts only, no reference answers); unparseable output counts as wrong
  * checkpoint = best validation macro F1, validation loss as the tie-breaker
  * the same 3 validation examples are printed after every epoch
"""

import json
import os
import re
from collections import Counter, defaultdict

CLASSES = ["well_tuned", "kp_too_high", "gain_too_low", "ki_too_low",
           "ki_too_high", "kd_too_high", "actuator_saturation"]
CLASS_TEXT = {
    "well_tuned": "Well tuned",
    "kp_too_high": "Kp too high (oscillatory)",
    "gain_too_low": "Overall gain too low (sluggish)",
    "ki_too_low": "Ki too low (remaining error)",
    "ki_too_high": "Ki too high (integral overshoot)",
    "kd_too_high": "Kd too high (noise amplification)",
    "actuator_saturation": "Actuator saturation",
}
TEXT_TO_CLASS = {v.lower(): k for k, v in CLASS_TEXT.items()}
UNPARSEABLE = "unparseable"


# ------------------------------------------------------------------ data
def load_split(data_dir, split):
    """Gemma-formatted split (prompts, responses, labels, plants)."""
    with open(os.path.join(data_dir, f"gemma_{split}.json")) as f:
        return json.load(f)


# ------------------------------------------------------- output parsing
def strip_prompt(generated, prompt):
    """model.generate returns prompt + continuation; keep only the model turn.

    Cut at the model-turn marker rather than at len(prompt): the decoded
    prompt can differ slightly from the original text (e.g. whitespace).
    """
    marker = "<start_of_turn>model"
    if marker in generated:
        out = generated.split(marker, 1)[1]
    elif generated.startswith(prompt):
        out = generated[len(prompt):]
    elif "<end_of_turn>" in generated:
        out = generated.split("<end_of_turn>", 1)[1]   # after the user turn
    else:
        out = ""
    return out.split("<end_of_turn>")[0].strip()


def extract_label(answer):
    """Class from the first 'Diagnosis: ...' line, or UNPARSEABLE."""
    m = re.search(r"Diagnosis:\s*(.+?)\.?\s*(?:\n|$)", answer)
    if not m:
        return UNPARSEABLE
    return TEXT_TO_CLASS.get(m.group(1).strip().rstrip(".").lower(), UNPARSEABLE)


# ---------------------------------------------------------------- metrics
def scores(true, pred):
    """Accuracy, macro F1 over the classes present in `true`, per-class P/R/F1,
    and the confusion counts. Unparseable predictions count as wrong."""
    conf = defaultdict(Counter)
    for t, p in zip(true, pred):
        conf[t][p] += 1
    present = [c for c in CLASSES if sum(conf[c].values()) > 0]
    per = {}
    for c in present:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in CLASSES if o != c)
        fn = sum(v for k, v in conf[c].items() if k != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[c] = dict(n=sum(conf[c].values()), precision=p, recall=r,
                      f1=2 * p * r / (p + r) if p + r else 0.0)
    acc = sum(conf[c][c] for c in CLASSES) / max(len(true), 1)
    macro = sum(v["f1"] for v in per.values()) / max(len(per), 1)
    return dict(accuracy=acc, macro_f1=macro, per_class=per,
                unparseable=sum(1 for p in pred if p == UNPARSEABLE),
                confusion={k: dict(v) for k, v in conf.items()})


def print_scores(name, s):
    print(f"{name}: accuracy={s['accuracy']:.3f}  macro-F1={s['macro_f1']:.3f}  unparseable={s['unparseable']}")
    for c, v in s["per_class"].items():
        print(f"   {c:22s} n={v['n']:3d}  P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f}")


def is_better(new, best):
    """PROTOCOL §7: higher val macro F1 wins; on a tie, lower val loss wins."""
    if best is None:
        return True
    if new["macro_f1"] > best["macro_f1"] + 1e-12:
        return True
    return abs(new["macro_f1"] - best["macro_f1"]) <= 1e-12 and new["val_loss"] < best["val_loss"]


# ------------------------------------------------------- model utilities
def token_lengths(tokenizer, prompts, responses):
    """Real tokenised length of each full training sequence (prompt + response)."""
    return [len(tokenizer(p + r)) for p, r in zip(prompts, responses)]


def choose_sequence_length(lengths, margin=16, step=32):
    """Smallest multiple of `step` that fits the longest sequence + margin
    (+1 for the start token the preprocessor adds)."""
    need = max(lengths) + 1 + margin
    return int(-(-need // step) * step)


def generate_answers(model, prompts, max_length, batch_size=8):
    answers = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        outs = model.generate(chunk, max_length=max_length)
        if isinstance(outs, str):
            outs = [outs]
        answers += [strip_prompt(o, p) for o, p in zip(outs, chunk)]
    return answers


def evaluate_split(model, split, max_length, batch_size=8):
    answers = generate_answers(model, split["prompts"], max_length, batch_size)
    preds = [extract_label(a) for a in answers]
    return scores(split["labels"], preds), answers, preds


def _save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)                       # atomic: never a half-written file


def epoch_order(n, seed, ep):
    """Shuffled order for epoch `ep`. Depends only on (seed, ep), so a resumed
    run uses exactly the same order as an uninterrupted one."""
    import random
    order = list(range(n))
    random.Random(seed * 1000 + ep).shuffle(order)
    return order


def save_checkpoint(model, ckpt_dir, history, best):
    """Everything needed to resume after a disconnect: latest LoRA weights,
    optimizer state (Adam moments + step count), history and best record.
    The state file is written LAST, so a crash mid-save leaves the previous
    complete checkpoint in place."""
    import numpy as np
    os.makedirs(ckpt_dir, exist_ok=True)
    ep = history[-1]["epoch"]
    model.backbone.save_lora_weights(os.path.join(ckpt_dir, f"epoch{ep}.lora.h5"))
    opt = [np.array(v.value) for v in model.optimizer.variables]
    np.savez(os.path.join(ckpt_dir, f"epoch{ep}_optimizer.npz"), *opt)
    _save_json(dict(last_epoch=ep, history=history, best=best), os.path.join(ckpt_dir, "state.json"))


def load_checkpoint(model, ckpt_dir):
    """Returns (history, best, last_epoch); (empty, None, 0) if no checkpoint."""
    import numpy as np
    sp = os.path.join(ckpt_dir, "state.json")
    if not os.path.exists(sp):
        return [], None, 0
    with open(sp) as f:
        st = json.load(f)
    ep = st["last_epoch"]
    model.backbone.load_lora_weights(os.path.join(ckpt_dir, f"epoch{ep}.lora.h5"))
    model.optimizer.build(model.trainable_variables)
    arrs = np.load(os.path.join(ckpt_dir, f"epoch{ep}_optimizer.npz"))
    vals = [arrs[f"arr_{i}"] for i in range(len(arrs.files))]
    ov = model.optimizer.variables
    if len(vals) != len(ov):
        raise RuntimeError(f"optimizer state mismatch: saved {len(vals)} vs model {len(ov)}")
    for v, a in zip(ov, vals):
        v.assign(a)
    return st["history"], st["best"], ep


def train_with_selection(model, train, val, epochs, lora_path, max_length,
                         watch_idx=(0, 1, 2), batch_size=1, gen_batch=8, log=print, seed=812,
                         ckpt_dir=None, stop_after_epoch=None):
    """One epoch at a time. After each epoch: val loss, val generation ->
    macro F1, print the same examples, save LoRA weights if best (§7).

    With `ckpt_dir` (put it on Google Drive), a checkpoint is saved after
    every epoch and the run RESUMES automatically from the last finished
    epoch when called again. `stop_after_epoch` simulates a disconnect (tests).
    """
    history, best, done = ([], None, 0)
    if ckpt_dir:
        history, best, done = load_checkpoint(model, ckpt_dir)
        if done:
            log(f"RESUMING after epoch {done} (best so far: epoch {best['epoch']}, val macro F1 {best['macro_f1']:.3f})")
    vdata = {"prompts": val["prompts"], "responses": val["responses"]}
    for ep in range(done + 1, epochs + 1):
        # Keras ignores shuffle=True for this input, and train.json is grouped
        # by class, so shuffle explicitly (seeded per epoch) before every epoch
        order = epoch_order(len(train["prompts"]), seed, ep)
        data = {"prompts": [train["prompts"][i] for i in order],
                "responses": [train["responses"][i] for i in order]}
        h = model.fit(data, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        val_loss = float(model.evaluate(vdata, batch_size=batch_size, verbose=0, return_dict=True)["loss"])
        s, answers, preds = evaluate_split(model, val, max_length, gen_batch)
        rec = dict(epoch=ep, train_loss=float(h.history["loss"][-1]), val_loss=val_loss,
                   macro_f1=s["macro_f1"], accuracy=s["accuracy"], unparseable=s["unparseable"])
        better = is_better(rec, best)
        if better:
            best = dict(rec)
            model.backbone.save_lora_weights(lora_path)
        history.append(rec)
        if ckpt_dir:
            save_checkpoint(model, ckpt_dir, history, best)
        log(f"\n=== epoch {ep}: train_loss={rec['train_loss']:.4f} val_loss={val_loss:.4f} "
            f"val_macroF1={s['macro_f1']:.3f} val_acc={s['accuracy']:.3f} unparseable={s['unparseable']}"
            f"{'  <- best so far, LoRA weights saved' if better else ''}"
            f"{'  [checkpoint saved]' if ckpt_dir else ''}")
        for i in (i for i in watch_idx if i < len(val["labels"])):   # skip indexes past the end
            log(f"--- val example {i} (true: {val['labels'][i]}, predicted: {preds[i]})\n{answers[i]}")
        if stop_after_epoch is not None and ep >= stop_after_epoch:
            log(f"(stopping after epoch {ep} to simulate a disconnect)")
            break
    return history, best


# ---------------------------------------------- course lab 8.3 deliverables
def save_predictions_csv(path, split, answers, preds):
    """predictions_<split>.csv as in lab 8.3: input, output (true label),
    model_output (generated answer), predicted (parsed class), plant."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["input", "output", "model_output", "predicted", "correct", "plant"])
        for p, t, a, pr, pl in zip(split["prompts"], split["labels"], answers, preds, split["plants"]):
            w.writerow([p, t, a, pr, int(t == pr), pl])


def course_report(true, pred):
    """sklearn classification_report + weighted precision/recall (lab 8.3) next
    to our macro F1. Unparseable answers stay in as their own (wrong) label."""
    from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
    # classes present in the TRUE labels, as PROTOCOL §6 macro F1: predictions
    # of any other label (or unparseable) still count as errors for the class
    labels = [c for c in CLASSES if c in set(true)]
    return dict(
        accuracy=accuracy_score(true, pred),
        precision_weighted=precision_score(true, pred, labels=labels, average="weighted", zero_division=0),
        recall_weighted=recall_score(true, pred, labels=labels, average="weighted", zero_division=0),
        macro_f1=scores(true, pred)["macro_f1"],
        report=classification_report(true, pred, labels=labels, zero_division=0),
    )


def spot_indices(n, k=5, seed=812):
    """The same k validation examples before and after fine-tuning (lab 8.3)."""
    import random
    return sorted(random.Random(seed).sample(range(n), k))
