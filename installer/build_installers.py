"""Builds both Lite Classify editions and their installers. Run from the repository root:

    python installer/build_installers.py              # both editions
    python installer/build_installers.py cpu          # one edition
    python installer/build_installers.py --skip-app   # installers only, from existing dist/

Each edition's app is built by build_exe.py with its own environment (.venv = CUDA
PyTorch, .venv-cpu = CPU-only PyTorch), then packed by Inno Setup's ISCC into
releases/installers/LiteClassify-<version>-Setup-<CPU|GPU>.exe (plus -N.bin slices when
the data exceeds ~2 GB). See installer/README.md.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import app_info  # noqa: E402

ENVS = {"gpu": os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
        "cpu": os.path.join(ROOT, ".venv-cpu", "Scripts", "python.exe")}


def find_iscc():
    for p in (shutil.which("ISCC"),
              os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"),
              r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
              r"C:\Program Files\Inno Setup 6\ISCC.exe"):
        if p and os.path.exists(p):
            return p
    sys.exit("Inno Setup 6 not found. Install it: winget install JRSoftware.InnoSetup --scope user")


def run(cmd, log):
    t = time.time()
    print(" ".join(cmd), "->", log)
    with open(log, "w", encoding="utf8") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"failed ({r.returncode}); see {log}")
    print(f"  done in {(time.time() - t) / 60:.1f} min")


def size_gb(path):
    total = 0
    for dp, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(dp, f)) for f in files)
    return total / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("editions", nargs="*", choices=["cpu", "gpu"], help="default: both")
    ap.add_argument("--skip-app", action="store_true", help="reuse dist/<edition>/LiteClassify")
    o = ap.parse_args()
    o.editions = o.editions or ["cpu", "gpu"]
    logs = os.path.join(ROOT, "releases", "logs")
    os.makedirs(logs, exist_ok=True)
    iscc = find_iscc()
    for ed in o.editions:
        if not o.skip_app:
            if not os.path.exists(ENVS[ed]):
                sys.exit(f"{ENVS[ed]} missing: create it with requirements{'-cpu' if ed == 'cpu' else ''}.txt")
            run([ENVS[ed], "build_exe.py", "--edition", ed], os.path.join(logs, f"build_{ed}.log"))
        app_dir = os.path.join(ROOT, "dist", ed, "LiteClassify")
        print(f"  {ed} app: {size_gb(app_dir):.2f} GB")
        run([iscc, f"/DEdition={ed.upper()}", f"/DAppVersion={app_info.APP_VERSION}",
             os.path.join("installer", "liteclassify.iss")], os.path.join(logs, f"installer_{ed}.log"))
    out = os.path.join(ROOT, "releases", "installers")
    for f in sorted(os.listdir(out)):
        print(f"  {f}: {os.path.getsize(os.path.join(out, f)) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
