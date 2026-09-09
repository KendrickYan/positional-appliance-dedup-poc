# Findings

## Contents
1. [Summary](#summary)
2. [Why Gaussian Splatting and Postshot were rejected](#why-gaussian-splatting-and-postshot-were-rejected)
3. [Capture attempt comparison](#capture-attempt-comparison)
4. [Scale calibration](#scale-calibration)
5. [Positional deduplication tests](#positional-deduplication-tests)
6. [Conclusion](#conclusion)

---

## Summary

**Question:** can 3D geometry solve appliance deduplication in a way appearance-based re-identification (CLIP/OSNet-style embeddings) fundamentally can't — specifically, telling apart two visually identical appliances in different physical locations?

**Answer: yes, validated end-to-end.** The working pipeline is COLMAP sparse reconstruction → dense MVS (`patch_match_stereo` + `stereo_fusion`) → per-detection depth-map backprojection (with a fused-cloud fallback) → DBSCAN clustering on real-world-scaled 3D positions. No Gaussian Splat training is required — an earlier plan to train one and detect on rendered views was deliberately abandoned (see below).

The core proof: two objects were deliberately given an identical class label — simulating the exact case appearance-matching can't handle — and position-based clustering correctly kept them as two separate appliances. A follow-up test placing two distinct objects ~25cm apart exposed a mistuned clustering threshold; it was corrected and every earlier test was re-run to confirm no regression.

What this POC does **not** establish is covered honestly in the [Conclusion](#conclusion) — most importantly, real-world capture reliability for a non-technical user, and whether the tuned distance threshold holds for genuine appliance spacing rather than arbitrary test objects.

---

## Why Gaussian Splatting and Postshot were rejected

The original plan was to train a 3D Gaussian Splat from the room capture, render synthetic views from it, run object detection on those renders, and deduplicate by projecting detections back into the splat's 3D space.

**Dropped in favor of dense MVS directly, for a specific reason:** a trained Gaussian Splat's per-Gaussian centers are the output of minimizing *photometric* rendering loss across training views — they're optimized to make the splat look right when rendered, not to sit at geometrically accurate 3D positions. This is exactly why "floater" artifacts are a well-known property of trained splats: semi-transparent Gaussians that only exist to help reconstruct a handful of specific training views correctly, with no real correspondence to a physical surface. COLMAP's dense multi-view stereo pipeline (`patch_match_stereo` + `stereo_fusion`) is built for the opposite goal — every fused point passes photometric *and* geometric cross-view consistency checks specifically designed to reject unreliable estimates. For pure point-localization, MVS is the correct tool, and Gaussian Splat training would have added real cost (GPU-hours of training, plus the plumbing to render and re-detect) without improving — likely while degrading — the actual thing needed: accurate positions.

**Jawset Postshot was evaluated and rejected on the same grounds**, with an added practical issue: automating its splat training in a headless backend pipeline requires the paid Studio tier (€39/mo), a cost not justified for a purely positional use case that doesn't need Postshot's core value proposition (fast, high-quality *renderable* splats).

---

## Capture attempt comparison

Two capture attempts of the same real, small, irregularly-shaped room. The first (portrait, standard walking-orbit technique) failed to produce a unified reconstruction. The second (landscape, deliberate room-emphasis technique) succeeded completely. The dominant variable was **capture technique, not raw frame count or footage quantity** — attempt 2 used half as many frames and got a strictly better result.

| | Attempt 1 | Attempt 2 |
|---|---|---|
| Orientation | Portrait | Landscape |
| Frames extracted | 302 | 150 |
| Camera intrinsics | Default guess (`fx=fy=2304`, wrong — see below) | Corrected guess (`fx=fy=1600`), shared across all frames |
| Final registered | 239 / 302 (79%), split across 6–7 disconnected fragments | **150 / 150 (100%), one unified reconstruction** |
| Mean reprojection error | 0.65–0.93px per fragment (fragments never joined) | 0.897px (after merge + bundle adjustment) |

### Attempt 1 — what went wrong, and why it wasn't just "bad scanning"

Initial run registered only 2 of 302 images. Root cause was **two independent, stacking problems**, both diagnosed from the COLMAP database rather than assumed:

1. **Bad camera intrinsics.** ffmpeg-extracted PNG frames carry no EXIF data, so COLMAP fell back to its default heuristic (`1.2 × max(width, height)`), guessing `fx=fy=2304` — identically, for all 302 frames, each treated as a separate unconstrained camera rather than one shared lens (`--ImageReader.single_camera` wasn't set initially). A rough estimate for an iPhone main camera's actual pixel focal length at this resolution is closer to 1500–1650px — the default guess was off by roughly 40–50%.
2. **After fixing intrinsics** (shared camera, `fx=fy=1600` starting guess): registration jumped from 2 to 239 images, in a healthy 0.65–0.93px reprojection error range — confirming the fix was correct and the core pipeline works. But the 239 registered frames split into 6–7 disconnected fragments rather than one model.

Mapping fragments back to frame sequence numbers revealed **four to five genuine dead zones** — contiguous stretches of 9 to 23 consecutive frames that registered nowhere, not weak links between otherwise-good neighbors. Widening the sequential matcher's overlap window (testing whether this was a "matching window too narrow" problem) did not help — fragment count barely changed, and one previously-working segment broke. This ruled out a tunable-parameter fix and pointed at the capture itself: specific stretches of the walk (forced by the room's tight, irregular shape) likely had too little camera translation (parallax) and/or motion blur to be usable, regardless of pipeline tuning.

### Attempt 2 — what changed

- **Landscape orientation** instead of portrait — wider horizontal field of view per frame for the same walk.
- **Deliberate technique**, consciously different from a uniform walking-orbit: emphasized specific parts of the room rather than trying to cover everything evenly.
- Correct shared-camera intrinsics applied from the start (carried over from attempt 1's fix).

Result: all 150 frames registered on the first `mapper` run, split into only 2 fragments (with 20 frames of natural overlap between them — within `Mapper.max_model_overlap`'s default of 20). `model_merger` combined them successfully; the raw merge briefly showed a degraded reprojection error (2.92px) because `model_merger` only applies a rigid alignment transform without re-optimizing. Running `bundle_adjuster` on the merged result brought reprojection error back down to 0.897px — confirming the two halves were genuinely geometrically consistent, not just superficially stitched together.

### Reproducing this stage

```bash
colmap feature_extractor \
  --database_path colmap_output_v2/database.db \
  --image_path data/raw_images_v2 \
  --ImageReader.camera_model OPENCV \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_params "1600,1600,960,540,0,0,0,0" \
  --FeatureExtraction.use_gpu 1

colmap sequential_matcher \
  --database_path colmap_output_v2/database.db \
  --FeatureMatching.use_gpu 1

colmap mapper \
  --database_path colmap_output_v2/database.db \
  --image_path data/raw_images_v2 \
  --output_path colmap_output_v2/sparse \
  --Mapper.init_min_tri_angle 4

# If mapper produces >1 fragment, merge and re-optimize:
colmap model_merger \
  --input_path1 colmap_output_v2/sparse/0 \
  --input_path2 colmap_output_v2/sparse/1 \
  --output_path colmap_output_v2/sparse/merged

colmap bundle_adjuster \
  --input_path colmap_output_v2/sparse/merged \
  --output_path colmap_output_v2/sparse/merged_ba
```

**Interpreting mean reprojection error:** it measures internal self-consistency (do the estimated cameras and 3D points agree with each other), not absolute real-world accuracy. From general practice with consumer phone footage: under ~1px is solid, 1–2px is usable but noisy, 2px+ typically signals an unresolved problem. There's no universal standard — acceptable error scales with image resolution and what the reconstruction is being used for.

---

## Scale calibration

COLMAP's monocular SfM reconstruction is scale-ambiguous by construction — nothing in a plain image sequence fixes an absolute real-world scale. Before any distance-based threshold (like a DBSCAN `eps`) means anything, the reconstruction's arbitrary units need to be tied to a real measurement.

**Method:** picked a rigid, precisely-measurable real-world reference (a monitor's bottom bezel corners), measured it with a tape measure, then found the same two points in the dense point cloud (`fused.ply`) using CloudCompare's point-picking tool.

- Real-world distance: **54cm**
- Same two points in the reconstruction: **2.798143 COLMAP units**

**scale_factor = 54 / 2.798143 ≈ 19.2985 cm per COLMAP unit**

To convert a desired real-world distance threshold into COLMAP units: `eps = desired_cm / 19.2985`.

This came from a single point-pair measurement — accurate to within a few percent given normal tape-measure and point-picking precision, which is sufficient for tuning a clustering threshold but not a guaranteed-exact conversion.

---

## Positional deduplication tests

Implemented `positional_dedup.py`: per-detection 3D position estimated via COLMAP MVS depth-map backprojection (median of several sampled points inside an inset bbox grid, avoiding edges where background can bleed through), with fused-point-cloud reprojection as a fallback when depth-map lookup fails. Positions grouped by class label and clustered with DBSCAN, `eps` set in real-world centimeters via the scale calibration above.

**Test 1 — same object, adjacent frames** (frame_0084/85/86, monitor): three independent position estimates landed within ~0.9–3.7cm of each other. All resolved via depth map directly (no fallback needed), 8–16 valid samples per detection out of a 25-point grid.

**Test 2 — same object, wide baseline** (frame_0003/85/137, spanning nearly the full 150-frame walk): pairwise spread widened to ~3–8.6cm — looser than the adjacent-frame case, as expected with more viewpoint diversity, but still well within tolerance.

**Test 3 — two distinct objects, deliberately mislabeled with the same class string, ~3m apart in the room:** 6 detections (3 per object) correctly split into 2 DBSCAN clusters, 0 detections flagged as noise. This is the core case appearance-based re-id cannot handle — two objects made appearance-indistinguishable on purpose — and position-based clustering resolved it correctly.

**Test 4 — close-proximity distinct objects** (monitor + an unrelated small object placed directly beneath it, ~25cm away): with the original `eps=50cm`, 12 detections (4 objects × 3 frames) produced only 3 clusters, not 4 — the new object incorrectly merged with the monitor. Since the two objects are visually and physically unambiguous as separate items, this was a confirmed false merge, not a judgment call.

**Empirical bounds established across all tests:**
- Same-object position estimate noise: ~1–10cm
- Confirmed false merge (should have split, didn't): ~25cm
- Confirmed correct splits: ~71cm, ~260cm, ~300cm

**Threshold correction:** tightened `DEDUP_MIN_SPACING_CM` from 50cm to 20cm — above the measured noise floor with margin, below the one confirmed false-merge distance. Re-ran all four tests against the new threshold: all 4 objects across 12 detections correctly resolved into 4 clusters, 0 noise — confirming the retune fixed the false merge without regressing any prior result.

**Open limitation:** the 25cm reference point came from an arbitrary test object (a chewing gum packet), not a realistic appliance pairing. This establishes the *pipeline's* geometric precision ceiling, not the *product's* correct real-world threshold — actual minimum spacing between genuinely distinct appliances (adjacent desk peripherals, stacked network equipment) is a domain question that needs real detector output and real office layouts to answer, not further synthetic tests in this room.

### Reproducing this stage

```bash
colmap image_undistorter \
  --image_path data/raw_images_v2 \
  --input_path colmap_output_v2/sparse/merged_ba \
  --output_path colmap_output_v2/dense \
  --output_type COLMAP

colmap patch_match_stereo \
  --workspace_path colmap_output_v2/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
  --workspace_path colmap_output_v2/dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path colmap_output_v2/dense/fused.ply

mkdir -p colmap_output_v2/dense/sparse_txt
colmap model_converter \
  --input_path colmap_output_v2/dense/sparse \
  --output_path colmap_output_v2/dense/sparse_txt \
  --output_type TXT

python3 positional_dedup.py
```

On a 12GB laptop GPU, `patch_match_stereo` with `--PatchMatchStereo.geom_consistency true` took roughly an hour for 150 frames — this runs two full passes per image (photometric, then geometric-consistency refinement) at default resolution and iteration count. `--PatchMatchStereo.max_image_size 1600` and `--PatchMatchStereo.num_iterations 3` are reasonable knobs to cut this down if iterating repeatedly; `geom_consistency` itself is worth keeping, since it's what makes the fused output trustworthy.

---

## Conclusion

**What's validated:**
- COLMAP can reliably reconstruct a small, irregularly-shaped real room from a single continuous walked capture — but *not* on the first attempt with generic technique. It took a specific, deliberate approach (landscape orientation, room-emphasis rather than uniform coverage) to go from 6–7 disconnected fragments to one unified model.
- Dense MVS produces enough point density (~2M points from 150 frames) for reliable per-detection depth lookup, with a sensible fallback for the cases it misses.
- A single physical reference measurement is sufficient to convert the reconstruction's arbitrary units into real-world centimeters, which is what makes a clustering threshold meaningful at all.
- The full pipeline handles both target failure modes correctly: same object across many viewpoints stays one cluster (noise floor ~1–10cm); distinct objects close together correctly split once the threshold is calibrated against real evidence rather than guessed.

**What's explicitly not validated, and shouldn't be assumed from this POC:**
- **True multi-spot fusion** (separate, independent recording sessions merged together) was descoped early and never tested — this POC only covers one continuous scan per room. Merging genuinely separate capture sessions is new, unvalidated risk, and would need a retrieval-based matcher (`vocab_tree`/`exhaustive`), not the sequential matcher used here.
- **The 20cm threshold is calibrated against arbitrary test objects in one room**, not real appliance spacing in a real SME office. It's a defensible starting point, not a settled product constant.
- **Capture reliability for a non-technical end user is unknown, and the evidence gathered here is a caution, not a reassurance** — even a careful, technically motivated first capture attempt failed and needed real diagnosis to fix. Nothing here validates that an average SME employee, doing this once with no feedback loop, would reliably produce a usable scan.
- **Compute cost was not optimized or measured for production feasibility.** Dense stereo alone took ~1 hour wall-clock for one small room on a 12GB laptop GPU. That cost multiplies per room, per customer scan — not evaluated against realistic infrastructure budget or latency requirements.
- **Occlusion and reflective surfaces were deliberately excluded from scope** (a mirror in the test room was covered rather than tested).

**Recommendation:** the core technical bet — dense geometric reconstruction for positional dedup, no splat training required — is validated and sound, and directly solves the appearance-ambiguity problem that motivated exploring 3D approaches in the first place. The open items above are real, unresolved risks, not settled questions — they're the natural next round of work if this gets integrated into SoapAI, not proof the idea is production-ready as-is.
