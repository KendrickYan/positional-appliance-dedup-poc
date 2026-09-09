# Findings — Capture Attempt Comparison

## Summary

Two capture attempts of the same real, small, irregularly-shaped room. The first (portrait, standard walking-orbit technique) failed to produce a unified reconstruction. The second (landscape, deliberate room-emphasis technique) succeeded completely. The dominant variable was **capture technique, not raw frame count or footage quantity** — attempt 2 used half as many frames and got a strictly better result.

|                         | Attempt 1                                                 | Attempt 2                                                  |
| ----------------------- | --------------------------------------------------------- | ---------------------------------------------------------- |
| Orientation             | Portrait                                                  | Landscape                                                  |
| Frames extracted        | 302                                                       | 150                                                        |
| Camera intrinsics       | Default guess (`fx=fy=2304`, wrong — see below)        | Corrected guess (`fx=fy=1600`), shared across all frames |
| Final registered        | 239 / 302 (79%), split across 6–7 disconnected fragments | **150 / 150 (100%), one unified reconstruction**     |
| Mean reprojection error | 0.65–0.93px per fragment (fragments never joined)        | 0.897px (after merge + bundle adjustment)                  |

### Scale calibration

Reference: monitor bottom bezel corners, measured 54cm in real life.
Same two points in `fused.ply` (CloudCompare point-to-point): 2.798143 COLMAP units.

**scale_factor = 19.2985 cm / COLMAP unit**

To convert a desired real-world distance threshold into COLMAP units: `eps = desired_cm / 19.2985`

## Attempt 1 — what went wrong, and why it wasn't just "bad scanning"

Initial run registered only 2 of 302 images. Root cause was **two independent, stacking problems**, both diagnosed from the COLMAP database rather than assumed:

1. **Bad camera intrinsics.** ffmpeg-extracted PNG frames carry no EXIF data, so COLMAP fell back to its default heuristic (`1.2 × max(width, height)`), guessing `fx=fy=2304` — identically, for all 302 frames, each treated as a separate unconstrained camera rather than one shared lens (`--ImageReader.single_camera` wasn't set initially). A rough estimate for an iPhone main camera's actual pixel focal length at this resolution is closer to 1500–1650px — the default guess was off by roughly 40–50%.
2. **After fixing intrinsics** (shared camera, `fx=fy=1600` starting guess): registration jumped from 2 to 239 images, in a healthy 0.65–0.93px reprojection error range — confirming the fix was correct and the core pipeline works. But the 239 registered frames split into 6–7 disconnected fragments rather than one model.

Mapping fragments back to frame sequence numbers revealed **four to five genuine dead zones** — contiguous stretches of 9 to 23 consecutive frames that registered nowhere, not weak links between otherwise-good neighbors. Widening the sequential matcher's overlap window (testing whether this was a "matching window too narrow" problem) did not help — fragment count barely changed, and one previously-working segment broke. This ruled out a tunable-parameter fix and pointed at the capture itself: specific stretches of the walk (forced by the room's tight, irregular shape) likely had too little camera translation (parallax) and/or motion blur to be usable, regardless of pipeline tuning.

## Attempt 2 — what changed

- **Landscape orientation** instead of portrait — wider horizontal field of view per frame for the same walk.
- **Deliberate technique**, consciously different from a uniform walking-orbit: emphasized specific parts of the room rather than trying to cover everything evenly.
- Correct shared-camera intrinsics applied from the start (carried over from attempt 1's fix).

Result: all 150 frames registered on the first `mapper` run, split into only 2 fragments (with 20 frames of natural overlap between them — within `Mapper.max_model_overlap`'s default of 20). `model_merger` combined them successfully; the raw merge briefly showed a degraded reprojection error (2.92px) because `model_merger` only applies a rigid alignment transform without re-optimizing. Running `bundle_adjuster` on the merged result brought reprojection error back down to 0.897px — confirming the two halves were genuinely geometrically consistent, not just superficially stitched together.

## Conclusion

For a small, irregularly-shaped real room, **capture technique and orientation matter more than raw frame count or coverage completeness.** A less exhaustive but more deliberate capture outperformed a more thorough but less considered one. This has a direct implication for SoapAI if this approach were ever pursued further: a production capture flow would need to guide the user's technique (e.g. live feedback, suggested path, orientation lock) rather than simply asking them to "walk around the room" — an unguided capture, even from a technically motivated user, was not reliable on the first attempt.

## Open question not yet answered

Whether this reconstruction is clean enough for `gsplat` training to produce a usable splat is still untested — sparse SfM registering successfully doesn't guarantee downstream splat quality. That's the next step.

---

## Positional deduplication — first working test

Implemented `positional_dedup.py`: per-detection 3D position estimated via COLMAP MVS depth-map backprojection (median of several sampled points inside an inset bbox grid), with fused-point-cloud reprojection as a fallback when depth-map lookup fails. Positions grouped by class label and clustered with DBSCAN, `eps` set in real-world centimeters via the scale calibration above.

**Test 1 — same object, adjacent frames (frame_0084/85/86, monitor):** three independent position estimates landed within ~0.9–3.7cm of each other. All resolved via depth map directly (no fallback needed), 8–16 valid samples per detection out of a 25-point grid.

**Test 2 — same object, wide baseline (frame_0003/85/137, spanning nearly the full 150-frame walk):** pairwise spread widened to ~3–8.6cm — looser than the adjacent-frame case, as expected with more viewpoint diversity, but still well within tolerance.

**Test 3 — two distinct objects, deliberately mislabeled with the same class string, ~3m apart in the room:** 6 detections (3 per object) correctly split into 2 DBSCAN clusters, 0 detections flagged as noise. This is the core case appearance-based re-id (CLIP/OSNet embeddings) cannot handle — two objects made appearance-indistinguishable on purpose here — and position-based clustering resolved it correctly.

**Not yet tested:** whether the `eps` threshold (currently 50cm) correctly separates two distinct objects placed *close together* (e.g. 1–1.5m apart, closer to a realistic worst case like adjacent desk monitors) rather than the ~3m separation used in Test 3, which gave DBSCAN a wide margin regardless of estimate noise.
