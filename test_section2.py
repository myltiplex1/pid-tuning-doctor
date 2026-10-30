"""Run the notebook's section-2 Drive cell outside Colab with stubs, under scenarios."""
import os, sys, types, shutil, zipfile, tempfile, json, nbformat
NB, GOOD_ZIP = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])

def make_old_zip(path):           # an earlier project zip (no r3 hash file)
    with zipfile.ZipFile(GOOD_ZIP) as zin, zipfile.ZipFile(path, "w") as zout:
        for i in zin.infolist():
            if not i.filename.endswith("FROZEN_v1.2_r7.sha256"):
                zout.writestr(i, zin.read(i.filename))

def run(name, setup, upload_new=None, expect_error=None):
    root = tempfile.mkdtemp()
    content = os.path.join(root, "content")
    os.makedirs(content)
    cell = [c.source for c in nbformat.read(NB, 4).cells if c.cell_type == "code" and "drive.mount" in c.source][0]
    cell = cell.replace("/content", content)
    if upload_new is not None:
        cell = cell.replace("UPLOAD_NEW_ZIP = False", f"UPLOAD_NEW_ZIP = {upload_new}")
    uploads = []
    g = types.ModuleType("google"); colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda p: os.makedirs(os.path.join(p, "MyDrive"), exist_ok=True))
    def upload():
        dst = os.path.join(os.getcwd(), "pid_doctor_v1.1.zip"); shutil.copy(GOOD_ZIP, dst); uploads.append(1)
        return {"pid_doctor_v1.1.zip": b""}
    colab.files = types.SimpleNamespace(upload=upload)
    sys.modules["google"], sys.modules["google.colab"] = g, colab
    setup(content)
    cwd = os.getcwd(); os.chdir(root)
    ns = {"os": os, "zipfile": zipfile, "files": colab.files, "time": __import__("time")}
    try:
        exec(cell, ns)
        err = None
    except Exception as e:
        err = e
    finally:
        os.chdir(cwd)
    drive_res = os.path.join(content, "drive/MyDrive/pid_doctor/results")
    link = os.path.join(content, "pid_doctor/results")
    ok_link = os.path.islink(link) and os.path.realpath(link) == os.path.realpath(drive_res)
    print(f"[{name}] error={type(err).__name__ if err else None}: {str(err)[:110] if err else ''}")
    print(f"    results is link to Drive: {ok_link}; uploads: {len(uploads)}; drive results: {sorted(os.listdir(drive_res)) if os.path.isdir(drive_res) else None}")
    has_r3 = os.path.exists(os.path.join(content, "pid_doctor/FROZEN_v1.2_r7.sha256"))
    print(f"    extracted code has r3 hash file: {has_r3}")
    if expect_error:
        assert err is not None and expect_error in str(err), f"expected error containing {expect_error!r}"
    return err, ok_link, uploads, drive_res, has_r3

def fresh(c): pass
def local_results_with_file(c):
    os.makedirs(os.path.join(c, "pid_doctor/results")); open(os.path.join(c, "pid_doctor/results/zeroshot_val.json"), "w").write("{}")
def local_results_empty(c): os.makedirs(os.path.join(c, "pid_doctor/results"))
def old_zip_on_drive(c):
    d = os.path.join(c, "drive/MyDrive/pid_doctor"); os.makedirs(d); make_old_zip(os.path.join(d, "pid_doctor_v1.1.zip"))
mode = sys.argv[3]
if mode == "before":
    run("existing local results dir (earlier notebook)", local_results_with_file)
    run("old zip already on Drive", old_zip_on_drive)
else:
    e, ok, up, dr, r3 = run("fresh start", fresh); assert not e and ok and len(up) == 1 and r3
    e, ok, up, dr, r3 = run("local results dir WITH files", local_results_with_file)
    assert not e and ok and any(x.startswith("local_backup_") for x in os.listdir(dr))
    e, ok, up, dr, r3 = run("local results dir EMPTY", local_results_empty); assert not e and ok
    run("old zip on Drive, flag off -> clear stop", old_zip_on_drive, expect_error="UPLOAD_NEW_ZIP")
    e, ok, up, dr, r3 = run("old zip on Drive, UPLOAD_NEW_ZIP=True", old_zip_on_drive, upload_new=True)
    assert not e and ok and len(up) == 1 and r3
    def already_linked(c):
        d = os.path.join(c, "drive/MyDrive/pid_doctor"); os.makedirs(os.path.join(d, "results"))
        shutil.copy(GOOD_ZIP, os.path.join(d, "pid_doctor_v1.1.zip"))
        os.makedirs(os.path.join(c, "pid_doctor")); os.symlink(os.path.join(d, "results"), os.path.join(c, "pid_doctor/results"))
        open(os.path.join(d, "results/zeroshot_val.json"), "w").write("{}")
    e, ok, up, dr, r3 = run("cell run twice (results already a Drive link)", already_linked)
    assert not e and ok and len(up) == 0 and "zeroshot_val.json" in os.listdir(dr)
    print("SECTION-2 SCENARIOS PASS")
