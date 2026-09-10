# Positional Appliance Deduplication — Geometric 3D Room Reconstruction

**A proof-of-concept validating that 3D geometry, not appearance, is the right signal for deduplicating appliances detected across a room scan.**

## The problem

A university project generating sustainability reports for Australian SMEs by detecting electrical appliances from a scanned office needs to count physical appliances accurately across a room scan. The hard case: two visually identical appliances in different locations — say, two matching desk monitors — must **not** be merged into one.

Appearance-based matching (comparing detection crops via CLIP/OSNet-style embeddings) is structurally incapable of solving this: two identical monitors produce near-identical embeddings regardless of where they physically are. No amount of tuning fixes that, because the signal the approach relies on — what something looks like — doesn't encode where it is.

## The approach

Reconstruct the room in 3D, localize each 2D detection in that 3D space, and cluster by **position** instead of appearance:

```
phone video walkthrough
        │
        ▼
COLMAP sparse reconstruction (SfM)  ── camera poses + sparse point cloud
        │
        ▼
COLMAP dense MVS (patch_match_stereo + stereo_fusion) ── dense fused point cloud
        │
        ▼
per-detection: sample points inside the bbox → look up depth per pixel → backproject to 3D
        │
        ▼
group by class label → DBSCAN cluster on 3D position → cluster count = deduplicated appliance count
```

An earlier version of this POC planned to train a 3D Gaussian Splat and detect on rendered views from it. That was deliberately dropped — see [`FINDINGS.md`](FINDINGS.md) for why (short version: trained Gaussian centers are optimized for photometric rendering quality, not geometric accuracy, and are strictly worse for point-localization than COLMAP's dense MVS output, which is purpose-built for exactly that).

## Result

**Validated end-to-end.** Two objects deliberately given an identical class label (to simulate the exact appearance-ambiguous case that motivated this) were correctly split into separate clusters based on position alone — real proof that geometry, not appearance, is the signal that solves this. A deliberately close-placed pair (~25cm apart) exposed a mistuned clustering threshold, which was corrected and re-validated against every prior test without regression.

Full methodology, every test run, the numbers behind each result, and — importantly — what's *not* yet validated: [`FINDINGS.md`](FINDINGS.md).

## Repo structure

```
positional_dedup.py       — the actual pipeline: depth-map backprojection + DBSCAN clustering
pick_bbox.py               — small helper to manually get bbox coordinates from a frame (stand-in for a real detector, which isn't built yet)
example_detections.json    — real test data used to produce the results in FINDINGS.md
docs/wsl2-setup.md         — environment setup (WSL2 + CUDA + COLMAP)
FINDINGS.md                — full methodology, results, and honest limitations
```

## Running this

1. Follow [`docs/wsl2-setup.md`](docs/wsl2-setup.md) for environment setup.
2. Capture a room as a single continuous walked video (see `FINDINGS.md` for what capture technique actually worked vs. didn't).
3. Run the COLMAP pipeline (sparse reconstruction → dense MVS) — commands in `FINDINGS.md`.
4. Use `pick_bbox.py` to manually label a few detections from real frames, or edit `example_detections.json` directly.
5. `pip install plyfile scikit-learn scipy numpy` then `python3 positional_dedup.py`.

## What this is not

- **Not production code.** No tests, minimal error handling, hardcoded paths — this is a spike meant to answer one technical question, not ship.
- **Not a claim that this is ready for real appliance data.** The clustering threshold was calibrated against arbitrary test objects, not real appliance spacing in a real office. See "What's explicitly not validated" in `FINDINGS.md`.
- **Not integrated into the SoapAI monorepo.** If the results here motivate building this into the real product, the relevant pieces get rebuilt properly as scoped SoapAI PRs — nothing here ships as-is.

## License

MIT — see [`LICENSE`](LICENSE).
