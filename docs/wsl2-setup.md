# Environment Setup — WSL2 + CUDA + COLMAP

Hardware used: Windows 11 laptop, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM, Blackwell / `sm_120`), 32GB RAM.

## 1. Install WSL2

From an **Administrator** PowerShell:
```powershell
wsl --install -d Ubuntu-24.04
```
Restart when prompted, then create a Linux username/password on first launch of the Ubuntu app.

## 2. GPU passthrough

**Do not install an NVIDIA driver inside WSL2** — it uses your existing Windows driver via passthrough; installing a second one breaks it.

Confirm from Windows PowerShell:
```powershell
nvidia-smi
```
Then confirm passthrough from inside WSL2 Ubuntu:
```bash
nvidia-smi
```
Both should show your GPU. If the WSL2 one fails, the Windows driver is usually out of date.

## 3. CUDA toolkit (WSL-specific repo, not the standard Linux one)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8
```
CUDA 12.8 specifically — required for Blackwell (`sm_120`) GPUs.

## 4. Base tools

```bash
sudo apt install -y python3-pip python3-venv git build-essential ffmpeg sqlite3
```

## 5. Python + PyTorch (verify Blackwell support explicitly)

Keep the project inside the WSL2 filesystem (e.g. `~/repositories/...`), not under `/mnt/c/...` — cross-boundary file I/O is noticeably slower and COLMAP does a lot of small file access.

```bash
mkdir -p ~/repositories/gaussian-splatting-poc && cd ~/repositories/gaussian-splatting-poc
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
```
Expect `True`, your GPU's name, and `sm_120` present in the arch list.

## 6. COLMAP — use conda-forge's CUDA build, not apt

Ubuntu's `apt install colmap` is CPU-only. Building COLMAP from source against CUDA is fiddly (Ceres solver dependency chain); conda-forge ships a working prebuilt CUDA binary instead.

```bash
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate

conda create -n colmap_gpu -c conda-forge colmap openimageio -y
conda activate colmap_gpu

colmap -h | grep -i cuda
```
Expect: `COLMAP 4.2.0 (... with CUDA)`.

**Known gotcha:** installing `colmap` alone via conda-forge can resolve to a build missing `libOpenImageIO.so.3.1` at runtime (`error while loading shared libraries`), even though the COLMAP package itself is fine. Installing `colmap` and `openimageio` **together in one `create` command** (as above) avoids this — installing them separately/incrementally can let conda's solver land on an inconsistent combination.

This env is fully separate from the PyTorch `.venv` — `conda activate colmap_gpu` for COLMAP steps, `source .venv/bin/activate` for anything Python/PyTorch, and both can be active simultaneously without conflict.

## 7. COLMAP 4.2.0 API notes (differs from 3.x docs found online)

Generic controls (GPU usage, thread count, image size) moved out of the SIFT-specific namespace into a general one, since 4.x supports multiple feature extractor/matcher types (SIFT, ALIKED, LoMa):

| 3.x (outdated) | 4.2.0 (current) |
|---|---|
| `--SiftExtraction.use_gpu` | `--FeatureExtraction.use_gpu` |
| `--SiftMatching.use_gpu` | `--FeatureMatching.use_gpu` |

`SiftExtraction.*` / `SiftMatching.*` still exist in 4.2.0, but now only contain SIFT-specific tuning (peak threshold, octaves, ratio test, etc.), not GPU control.

**When a flag errors as unrecognized, don't guess a fix — run `colmap <command> -h` and check the current flag list directly.** Several other renames likely exist beyond the two above.

Other 4.2.0 quirks encountered:
- `model_converter` and `bundle_adjuster` require their `--output_path` directory to **already exist** — unlike most other COLMAP commands, they don't create it for you. `mkdir -p` first.
- `two_view_geometries.config` integer codes weren't confirmed against an authoritative current source for this version — don't rely on the classic 3.x enum mapping (`0=undefined, 1=degenerate, 2=calibrated...`) without verifying against this version's source.
