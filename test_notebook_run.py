"""Execute the notebook's own cells (sections 2-check to 7) on CPU with a tiny
random Gemma 3 in place of the real preset, in a throwaway copy of the project.

Proves every cell runs in order and writes the files the report needs. Scores
are meaningless (random model). Section 1 (pip/Kaggle) and the Drive cell
(tested by test_section2.py) are replaced by stubs; section 8 (download) is skipped.

    python test_notebook_run.py pid_doctor_lora.ipynb
"""
import os
import shutil
import sys
import tempfile

os.environ.setdefault("KERAS_BACKEND", "jax")
HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pid_doctor_lora.ipynb"))

import json, time, zipfile  # noqa: E402,E401
import keras  # noqa: E402
import keras_hub  # noqa: E402
import nbformat  # noqa: E402

work = tempfile.mkdtemp()
proj = os.path.join(work, "pid_doctor")
shutil.copytree(HERE, proj, ignore=shutil.ignore_patterns("__pycache__", "archive", "dev", "*.log", "results"))
drive_results = os.path.join(work, "drive_results")
os.makedirs(drive_results)
os.chdir(proj)
os.symlink(drive_results, "results")                 # what section 2 sets up

# NEVER touch the real test sets: replace Test A/B in this throwaway copy with
# DUMMY records (the development-only seed-0 test splits in dev/, never scored and
# not part of the final data). Subprocesses (rule_baseline.py) read these files too.
import json as _json  # noqa: E402
REAL_TEST_INPUTS = set()
for _s in ["test_A", "test_B"]:
    REAL_TEST_INPUTS |= {r["input"] for r in _json.load(open(os.path.join(HERE, "data", f"{_s}.json")))}
    shutil.copy(os.path.join(HERE, "dev", f"{_s}.json"), os.path.join(proj, "data", f"{_s}.json"))
sys.path.insert(0, proj)
import prepare_gemma_data as _P  # noqa: E402
for _s in ["test_A", "test_B"]:
    _P.convert(_s, data_dir="data")
DUMMY_N = {_s: len(_json.load(open(f"data/{_s}.json"))) for _s in ["test_A", "test_B"]}
# v1.2 folder rebuilt in the copy FROM the copy's data/ (so its tests are the dummies too)
import make_v12 as _M  # noqa: E402
shutil.rmtree("data_v12", ignore_errors=True)
_M.main("data", "data_v12")
for _s in ["test_A", "test_B"]:
    assert not ({r["input"] for r in _json.load(open(f"data_v12/{_s}.json"))} & REAL_TEST_INPUTS)
FORMAT = os.environ.get("PIDDOC_FORMAT", "v1.1")
print("answer format under test:", FORMAT)
for _s in ["test_A", "test_B"]:
    _d = _json.load(open(f"data/{_s}.json"))
    assert not ({r["input"] for r in _d} & REAL_TEST_INPUTS), "dummy test overlaps the real test set"
    assert len(_json.load(open(f"data/gemma_{_s}.json"))["prompts"]) == DUMMY_N[_s]
print("Test A/B in the throwaway copy are DUMMY (dev seed 0):", DUMMY_N)
import pid_lib as L  # noqa: E402
import smoke_test as S  # noqa: E402

# small subsets so CPU is quick; the notebook's own code runs unchanged on them
_orig_load = L.load_split
L.load_split = lambda d, s: {k: v[:{"train": 8, "val": 7}.get(s, 6)] for k, v in _orig_load(d, s).items()}

tok_dir = tempfile.mkdtemp()
tr = _orig_load("data", "train")
_, PROTO = S.tiny_tokenizer(tr["prompts"][:20] + tr["responses"][:20], tok_dir)


def tiny_from_preset(name, **kw):
    assert name == "gemma3_1b", name
    keras.utils.set_random_seed(812)
    tok = keras_hub.tokenizers.Gemma3Tokenizer(proto=PROTO)
    bb = keras_hub.models.Gemma3Backbone(
        vocabulary_size=tok.vocabulary_size(), image_size=896, num_layers=2, num_query_heads=2,
        num_key_value_heads=1, hidden_dim=32, intermediate_dim=64, head_dim=16, vision_encoder=None)
    pre = keras_hub.models.Gemma3CausalLMPreprocessor(tokenizer=tok, sequence_length=512)
    return keras_hub.models.Gemma3CausalLM(backbone=bb, preprocessor=pre)


keras_hub.models.Gemma3CausalLM.from_preset = staticmethod(tiny_from_preset)

cells = [c.source for c in nbformat.read(NB, 4).cells if c.cell_type == "code"]
ns = dict(os=os, json=json, time=time, zipfile=zipfile, keras=keras, keras_hub=keras_hub,
          files=None, DRIVE=work, display=print)
ran = 0
for src in cells:
    if "pip install" in src or "userdata" in src or "drive.mount" in src or "files.download" in src:
        continue                                     # section 1, Drive cell, section 8
    src = src.replace('sys.path.insert(0, "/content/pid_doctor")', f'sys.path.insert(0, "{proj}")')
    src = src.replace("EPOCHS = 5e-5, 4, 6", "EPOCHS = 5e-5, 4, 2")          # 2 epochs on CPU
    src = src.replace("RUN_FINAL_TESTS = False", "RUN_FINAL_TESTS = True")   # exercise section 7 here only
    src = src.replace("plt.show()", "plt.close('all')")
    src = src.replace('ANSWER_FORMAT = "v1.1"', f'ANSWER_FORMAT = "{FORMAT}"')
    exec(compile(src, "<notebook cell>", "exec"), ns)
    ran += 1

out = sorted(os.listdir(drive_results))
print("\ncells executed:", ran)
print("files written to (fake) Drive results:", out)
need = ["zeroshot_val.json", "predictions_val_zeroshot.csv", "FINAL_test_results.json", "FINAL_baselines.txt",
        "FINAL_RUN_STARTED.txt", "FINAL_predictions_test_A.csv", "FINAL_predictions_test_B.csv"]
missing = [f for f in need if f not in out]
missing += [s for s in ["_predictions_train.csv", "_predictions_val.csv", "_course_reports.json",
                        "_best.lora.h5", "_history.json", "_curves.png"] if not any(f.endswith(s) for f in out)]
assert not missing, f"missing outputs: {missing}"
assert any(f.endswith("_checkpoints") for f in out)
fr = json.load(open(os.path.join(drive_results, "FINAL_test_results.json")))
assert "course_report" in fr["test_A"] and "zeroshot_test_B" in fr
# the final-test lock: running section 7 again must refuse
lock_cell = [c for c in cells if "RUN_FINAL_TESTS = False" in c][0].replace("RUN_FINAL_TESTS = False", "RUN_FINAL_TESTS = True")
import io, contextlib  # noqa: E402,E401
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(lock_cell, ns)
assert "already started" in buf.getvalue(), buf.getvalue()
print("final-test lock refuses a second run: OK")
# the baseline SUBPROCESS must have read the dummy files, not the real ones
_bl = open(os.path.join(drive_results, "FINAL_baselines.txt")).read()
import re as _re  # noqa: E402
_ns = [int(x) for x in _re.findall(r"data/test_[AB]\.json: n=(\d+)", _bl)]
assert _ns and set(_ns) <= set(DUMMY_N.values()), (_ns, DUMMY_N)
_fr = json.load(open(os.path.join(drive_results, "FINAL_test_results.json")))
for _s in ["test_A", "test_B"]:
    assert len(_fr[_s]["labels"]) <= DUMMY_N[_s]
print("baseline subprocess and model read only DUMMY test files: OK")
print("NOTEBOOK RUN TEST PASS")
