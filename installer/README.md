# Building the Lite Classify installers

Two Windows installers per release, both offline (everything bundled):

| Edition | Environment | Installer | For |
| --- | --- | --- | --- |
| **CPU** | `.venv-cpu` (`requirements-cpu.txt`, PyTorch CPU wheels) | `LiteClassify-<ver>-Setup-CPU.exe` (v1.0: 317 MB; 1.0 GB installed) | Everyone; no NVIDIA GPU needed |
| **GPU** | `.venv` (`requirements.txt`, PyTorch CUDA 12.4) | `LiteClassify-<ver>-Setup-GPU.exe` (v1.0: 1.81 GB; 4.8 GB installed) | NVIDIA GPU users: faster segmentation; DINOv2 ~16x faster |

## One-time setup

    python -m venv .venv-cpu
    .venv-cpu\Scripts\pip install -r requirements-cpu.txt
    winget install JRSoftware.InnoSetup --scope user

`.venv` is the normal development environment (`requirements.txt`). The DINOv2 weights are
bundled from `models/` or the torch hub cache (`%USERPROFILE%\.cache\torch\hub\checkpoints`).

## Build

    python installer/build_installers.py            # both (GPU app build and compression take longest)
    python installer/build_installers.py cpu
    python installer/build_installers.py --skip-app # repack existing dist/<edition>/LiteClassify

Outputs go to `releases/installers/`, logs to `releases/logs/`. The version comes from
`app_info.APP_VERSION`.

## What the installer does

- Installs per user into `%LOCALAPPDATA%\Programs\Lite Classify` without admin rights (an admin
  can choose all users); Start-menu shortcut, optional desktop shortcut, uninstaller.
- Shows the GPL-3.0 licence.
- Both editions share one AppId: installing one over the other replaces it, and the old
  `_internal` libraries are removed first.
- The GPU installer warns when no NVIDIA driver (`nvcuda.dll`) is present.
- Each installer is one file. If the compressed data ever exceeds Inno Setup's ~2 GB
  single-file limit, compile with `ISCC /DSpan ...` to split it into `.bin` slices beside the
  `.exe` (keep them together when distributing).

## Testing

Silent install, self-test from the installed copy, uninstall (done for both v1.0 installers):

    LiteClassify-1.0-Setup-CPU.exe /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER /LOG=setup.log
    "%LOCALAPPDATA%\Programs\Lite Classify\LiteClassify.exe" --selftest ortho.tif dsm.tif
    type %TEMP%\liteclassify_selftest.json
    "%LOCALAPPDATA%\Programs\Lite Classify\unins000.exe" /VERYSILENT

Install folders must keep paths under Windows' 260-character limit: the deepest file is about
103 characters below the install folder.

## Before distributing

Install each on a clean Windows machine (or a fresh user account), launch from the Start
menu, load an orthomosaic, segment on CPU (and GPU for the GPU edition), run Find Groups
(DINOv2), train/classify, export, and uninstall. Some PCs also need the Microsoft Visual
C++ Redistributable (x64).

The source release on GitHub ships code only (no `dist/`, no libraries): advanced users
install the requirements and run `python liteclassify.py`, or rebuild with these scripts.
