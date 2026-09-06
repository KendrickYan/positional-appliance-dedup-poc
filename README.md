# Gaussian Splatting POC — Room Reconstruction for Appliance Detection

## What this is

A standalone proof-of-concept, developed alongside [SoapAI](https://github.com/) (a university group project generating sustainability reports for Australian SMEs by detecting electrical appliances in scanned office rooms).

SoapAI's current production approach captures a single 360° panorama per standing position and runs object detection on it. This POC tests an alternative: reconstructing a full room as a 3D Gaussian Splat from a walked capture, then rendering synthetic views from the splat to run detection on — with the goal of solving appliance deduplication geometrically (via known 3D position) rather than by visual appearance matching alone.

**This repo is intentionally separate from the SoapAI monorepo.** It has no stable interface, no tests, and no guarantee anything here ships. If the results are promising, the relevant pieces get rebuilt properly as real SoapAI PRs — nothing from this repo is copied in as-is.

## The question this POC answers

1. Can COLMAP reliably reconstruct camera poses from a walked room capture (not a tripod dataset)?
2. Does `gsplat` training produce a geometrically clean splat (straight walls, no floating artifacts) from that reconstruction?
3. How long does each stage actually take on consumer hardware?
4. Is the capture UX (walking orbit, varied height) realistic for a non-technical SME user?

This is explicitly **not** trying to prove Gaussian Splatting is the right call — it's trying to find out, honestly, including if the answer is "no, this doesn't work well enough."

## Non-goals

- Not production code
- Not integrated into the SoapAI Turborepo
- Not optimized, not tested, not error-handled beyond what's needed to get a result
- Not a claim that this approach is superior to SoapAI's current panorama pipeline — see the comparison discussion this POC came out of

## Hardware used

- Windows 11 laptop, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM, Blackwell / `sm_120`)
- Reconstruction and training run inside WSL2 (Ubuntu) with CUDA passthrough — native Windows CUDA/COLMAP builds were avoided due to toolchain complexity

## Pipeline

```
captured photos (walked orbit, varied height)
        │
        ▼
   COLMAP (SfM) ── sparse point cloud + estimated camera poses
        │
        ▼
   gsplat (training) ── trained 3D Gaussian Splat
        │
        ▼
   render synthetic views from known camera poses
        │
        ▼
   run existing object detector on renders
        │
        ▼
   project 2D detections into 3D using render camera poses → cluster by position (dedup)
```

## Status

🚧 In progress — environment setup phase.

## Setup

See `docs/wsl2-setup.md` for the step-by-step WSL2 + CUDA + Python environment guide.

## License

TBD — currently private. If made public as a portfolio piece, will add a license at that point.
