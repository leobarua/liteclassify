import os
import sys
import shutil
import argparse
import PyInstaller.__main__

# Output name and icon (defaults: LiteClassify, logo_liteclassify.ico). The installers
# use --edition cpu|gpu; see installer/README.md.
_args = argparse.ArgumentParser(description="Build the app with PyInstaller.")
_args.add_argument("--name", default="LiteClassify", help="executable and dist folder name")
_args.add_argument("--icon", default="logo_liteclassify.ico", help="exe and window icon (.ico); the app loads app_info.ICON_FILE")
_args.add_argument("--edition", choices=["gpu", "cpu", "none"], default="none",
                   help="output to dist/<edition>/ and build/<edition>/ so the two installer builds "
                        "don't overwrite each other. Run each with its own environment: gpu from .venv "
                        "(CUDA PyTorch), cpu from .venv-cpu (requirements-cpu.txt). See installer/README.md")
_opts = _args.parse_args()
APP_NAME = _opts.name
ICON_FILE = _opts.icon
EDITION = _opts.edition
DIST_DIR = "dist" if EDITION == "none" else os.path.join("dist", EDITION)
WORK_DIR = "build" if EDITION == "none" else os.path.join("build", EDITION)


def prune_bundle(app_dir):
    """ Drops what only compiling against PyTorch needs (static .lib files, C++ headers):
    ~850 MB that the app never loads. """
    torch_dir = os.path.join(app_dir, "_internal", "torch")
    freed = 0
    for dp, dirs, files in os.walk(torch_dir):
        for f in files:
            if f.lower().endswith(".lib"):
                fp = os.path.join(dp, f)
                freed += os.path.getsize(fp)
                os.remove(fp)
    inc = os.path.join(torch_dir, "include")
    if os.path.isdir(inc):
        for dp, _, files in os.walk(inc):
            freed += sum(os.path.getsize(os.path.join(dp, f)) for f in files)
        shutil.rmtree(inc)
    print(f" -> Pruned {freed / 1e6:.0f} MB of build-only PyTorch files (.lib, include/).")


def build_application():
    print("==============================================================")
    print("              LITE CLASSIFY EXECUTABLE BUILDER                ")
    print("==============================================================")
    
    # Define primary entry script
    entry_script = "liteclassify.py"   # launcher: splash screen, then obia_app (see liteclassify.py)
    
    if not os.path.exists(entry_script):
        print(f"[ERROR] Could not find entry script '{entry_script}' in the current directory.")
        sys.exit(1)

    print(" -> Configuring build directories and cleaning old caches...")
    # Clean up previous build outputs to prevent compile mismatches
    if EDITION != "none":
        import torch
        is_cuda = torch.version.cuda is not None
        if (EDITION == "gpu") != is_cuda:
            print(f"[ERROR] --edition {EDITION} needs {'CUDA' if EDITION == 'gpu' else 'CPU-only'} PyTorch, but this "
                  f"environment has torch {torch.__version__}. Run it with {'.venv' if EDITION == 'gpu' else '.venv-cpu'}.")
            sys.exit(1)
    for folder in [os.path.join(WORK_DIR, APP_NAME), os.path.join(DIST_DIR, APP_NAME)]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f" -> Notice: Could not completely delete '{folder}' directory ({e}). Continuing...")

    # Build standard compilation arguments
    pyinstaller_args = [
        entry_script,                  # Entry point
        f"--name={APP_NAME}",           # Name of output executable
        "--onedir",                    # Package as a single directory (highly stable for geospatial DLLs)
        "--windowed",                  # Hide the redundant command terminal (since our UI has its own Console)
        "--clean",                     # Clear PyInstaller caches before compiling
        "--noconfirm",                 # Overwrite existing directories automatically
        f"--distpath={DIST_DIR}",
        f"--workpath={WORK_DIR}",
        "--exclude-module=torchvision",   # not used by the app

        # UPX compression is a well-documented source of corrupted/AV-flagged DLLs for
        # large native-heavy stacks like PyTorch + GDAL -- it "works on the dev machine"
        # (where the DLLs are already cached/trusted) and then silently fails to launch
        # or crashes on a clean target machine. The size savings aren't worth that risk
        # for a distributable build, so UPX is disabled outright rather than trying to
        # cherry-pick which DLLs to exclude from it.
        "--noupx",

        # Ensure deep-learning, GIS, and scientific-computing engines are explicitly
        # discovered -- these are imports PyInstaller's static analysis is prone to miss
        # because they're pulled in indirectly (a compiled/Cython submodule, or a
        # backend selected at runtime rather than imported by name at the top of a file).
        "--hidden-import=rasterio",
        "--hidden-import=fiona",
        "--hidden-import=geopandas",
        "--hidden-import=shapely",
        "--hidden-import=rasterstats",
        "--hidden-import=sklearn.ensemble",
        "--hidden-import=sklearn.tree",
        "--hidden-import=torch",
        "--hidden-import=pyogrio._geometry",
        "--hidden-import=pyogrio._io",
        "--hidden-import=pyogrio._err",
        "--hidden-import=pyogrio._ogr",
        "--hidden-import=pyogrio._env",
        "--hidden-import=skimage.segmentation._slic",
        "--hidden-import=scipy.ndimage",

        # Gather all sidecar data files, schemas, and dynamic DLL libraries. torch,
        # skimage, scipy, and sklearn were added here (previously only referenced via
        # --hidden-import above, or not referenced at all for skimage/scipy) because
        # obia_engine.py depends on all four for real compute (GPU SLIC via torch,
        # skimage.segmentation/color/feature/exposure for CPU SLIC + GLCM texture +
        # color-space conversion, scipy.ndimage.gaussian_filter for pre-segmentation
        # smoothing, sklearn.ensemble.RandomForestClassifier for classification) --
        # --hidden-import alone only tells PyInstaller a module exists, it does not
        # pull in a package's own data files or every compiled extension it loads
        # internally, which torch/skimage/scipy all lean on heavily.
        "--collect-all=rasterio",
        "--collect-all=fiona",
        "--collect-all=geopandas",
        "--collect-all=shapely",
        "--collect-all=rasterstats",
        # Not --collect-all=PySide6: that bundles all of Qt (WebEngine, QML, 3D... ~550 MB)
        # while the app imports only QtCore/QtGui/QtWidgets; PyInstaller's PySide6 hook
        # collects what is imported plus the platform/image plugins.
        "--collect-all=pyogrio",
        "--collect-all=torch",
        "--collect-all=skimage",
        "--collect-all=scipy",
        "--collect-all=sklearn",
    ]

    # Explicitly pack the icon as a dynamic asset to copy into the distribution folder
    # -- this is also the icon obia_app.py loads at runtime (via get_resource_path) for
    # the window's title-bar icon, so keep this destination ('.', i.e. the bundle
    # root) in sync with get_resource_path's expectations if either ever changes.
    icon_path = ICON_FILE
    if os.path.exists(icon_path):
        pyinstaller_args.append(f"--icon={icon_path}")
        pyinstaller_args.append(f"--add-data={icon_path};.")
    else:
        print(f"[WARNING] '{icon_path}' not found in the current directory -- the built "
              f"exe will fall back to PyInstaller's default icon, and the app window "
              f"itself will fall back to Qt's default icon at runtime.")

    # Discover Classes: the vendored DINOv2 backbone code (imported at run time from
    # <bundle>/third_party, see obia_discover.load_backbone) and its weights in
    # <bundle>/models, so deep features work offline. The weights are not in git
    # (*.pth is ignored): they are copied from the torch hub cache when missing.
    # Splash background (branding/make_splash_bg.py), drawn by liteclassify.py at start-up.
    if os.path.exists(os.path.join("branding", "splash_bg.png")):
        pyinstaller_args.append("--add-data=branding/splash_bg.png;.")
    pyinstaller_args.append("--add-data=third_party;third_party")
    weights = os.path.join("models", "dinov2_vits14_pretrain.pth")
    if not os.path.exists(weights):
        try:
            import torch, shutil
            cached = os.path.join(torch.hub.get_dir(), "checkpoints", "dinov2_vits14_pretrain.pth")
            if os.path.exists(cached):
                os.makedirs("models", exist_ok=True)
                shutil.copy2(cached, weights)
        except Exception:
            pass
    if os.path.exists(weights):
        pyinstaller_args.append(f"--add-data={weights};models")
    else:
        print("[WARNING] DINOv2 weights not found (models/ or the torch hub cache) -- the built app "
              "will download them the first time Discover computes deep features.")

    print("\n -> Triggering PyInstaller compilation engine...")
    print(" -> This process may take several minutes to package the PyTorch/GDAL/Qt binaries.")
    print(" -> Note: UPX compression is disabled (--noupx) for build reliability, so the")
    print("    output folder will be larger than a UPX-compressed build would be -- that's")
    print("    expected, not a regression.")
    print("==============================================================\n")

    try:
        PyInstaller.__main__.run(pyinstaller_args)
        prune_bundle(os.path.join(DIST_DIR, APP_NAME))

        exe_path = os.path.join(DIST_DIR, APP_NAME, f"{APP_NAME}.exe")
        if not os.path.exists(exe_path):
            # PyInstaller.__main__.run() doesn't raise on every failure mode (e.g. it
            # can exit cleanly having skipped the final EXE stage) -- verify the actual
            # artifact exists rather than trusting a lack of exception alone.
            print("\n[BUILD ERROR] PyInstaller reported no exception, but the expected "
                  f"output '{exe_path}' was not found. Check the log above for the real "
                  "failure -- this usually means an import/collection step failed partway "
                  "through rather than at the very end.")
            sys.exit(1)

        print("\n==============================================================")
        print("                 BUILD COMPLETION DIAGNOSTIC                  ")
        print("==============================================================")
        print("Success! The application has been cleanly packaged.")
        print(f" -> Output Path: ./{DIST_DIR}/{APP_NAME}/")
        print(f" -> Launch File: ./{DIST_DIR}/{APP_NAME}/{APP_NAME}.exe")
        print("\nBefore distributing, smoke-test the built exe yourself first (a clean")
        print("build is not the same as a working one) -- launch it, load an orthomosaic,")
        print("run a segmentation on both CPU and GPU if available, and confirm the")
        print("window/taskbar icon shows up correctly.")
        print("\nDistribution Instructions:")
        print(" 1. Navigate inside your project's 'dist' folder.")
        print(f" 2. Right-click the '{APP_NAME}' folder and select 'Compress to ZIP file'.")
        print(" 3. Send this ZIP file to other users. They can run it by unzipping it")
        print(f"    and double-clicking '{APP_NAME}.exe' -- no Python install required.")
        print(" 4. If a recipient's machine has never run a Python/PyTorch application")
        print("    before, they may also need the Microsoft Visual C++ Redistributable")
        print("    (x64) installed -- a common, one-time, machine-wide prerequisite for")
        print("    PyTorch/GDAL's compiled DLLs that PyInstaller does not bundle itself.")
        print("==============================================================")

    except Exception as e:
        print(f"\n[BUILD ERROR] PyInstaller run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_application()