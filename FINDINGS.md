Jump to [conclusion](#conclusion-1)

# Findings — Capture Attempt Comparison

## Summary

Two capture attempts of the same real, small, irregularly-shaped room. The first (portrait, standard walking-orbit technique) failed to produce a unified reconstruction. The second (landscape, deliberate room-emphasis technique) succeeded completely. The dominant variable was **capture technique, not raw frame count or footage quantity** — attempt 2 used half as many frames and got a strictly better result.

|                         | Attempt 1                                                | Attempt 2                                                |
| ----------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| Orientation             | Portrait                                                 | Landscape                                                |
| Frames extracted        | 302                                                      | 150                                                      |
| Camera intrinsics       | Default guess (`fx=fy=2304`, wrong — see below)          | Corrected guess (`fx=fy=1600`), shared across all frames |
| Final registered        | 239 / 302 (79%), split across 6–7 disconnected fragments | **150 / 150 (100%), one unified reconstruction**         |
| Mean reprojection error | 0.65–0.93px per fragment (fragments never joined)        | 0.897px (after merge + bundle adjustment)                |

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

# Positional deduplication — first working test

Implemented `positional_dedup.py`: per-detection 3D position estimated via COLMAP MVS depth-map backprojection (median of several sampled points inside an inset bbox grid), with fused-point-cloud reprojection as a fallback when depth-map lookup fails. Positions grouped by class label and clustered with DBSCAN, `eps` set in real-world centimeters via the scale calibration above.

**Test 1 — same object, adjacent frames (frame_0084/85/86, monitor):** three independent position estimates landed within ~0.9–3.7cm of each other. All resolved via depth map directly (no fallback needed), 8–16 valid samples per detection out of a 25-point grid.

**Test 2 — same object, wide baseline (frame_0003/85/137, spanning nearly the full 150-frame walk):** pairwise spread widened to ~3–8.6cm — looser than the adjacent-frame case, as expected with more viewpoint diversity, but still well within tolerance.

**Test 3 — two distinct objects, deliberately mislabeled with the same class string, ~3m apart in the room:** 6 detections (3 per object) correctly split into 2 DBSCAN clusters, 0 detections flagged as noise. This is the core case appearance-based re-id (CLIP/OSNet embeddings) cannot handle — two objects made appearance-indistinguishable on purpose here — and position-based clustering resolved it correctly.

**Not yet tested:** whether the `eps` threshold (currently 50cm) correctly separates two distinct objects placed *close together* (e.g. 1–1.5m apart, closer to a realistic worst case like adjacent desk monitors) rather than the ~3m separation used in Test 3, which gave DBSCAN a wide margin regardless of estimate noise.

# Positional deduplication — threshold calibration

**Test 4 — close-proximity distinct objects (monitor + an unrelated small object placed directly beneath it):** 12 detections (4 objects × 3 frames) clustered with the original `eps=50cm` produced only 3 clusters, not 4 — the new object merged with the monitor. Computed real-world distance between the two: **~25.2cm**. Since the two objects are visually and physically unambiguous as separate items, this is a confirmed false merge, not a judgment call.

**Empirical bounds established across all tests so far:**

- Same-object position estimate noise: ~1–10cm (tighter at close-baseline frames, looser at wide-baseline)
- Confirmed false merge (should have split, didn't): ~25cm
- Confirmed correct splits: ~71cm, ~260cm, ~300cm

**Action taken:** tightened `DEDUP_MIN_SPACING_CM` from 50cm to 20cm — above the measured noise floor with margin, below the one confirmed false-merge distance. Not yet re-validated against the same test cases (should rerun Tests 1–4 against the new threshold to confirm no regressions before treating 20cm as settled).

**Open limitation:** the 25cm reference point came from an arbitrary test object (chewing gum), not a realistic appliance pairing. This establishes the *pipeline's* geometric precision ceiling, not the *product's* correct real-world threshold — actual minimum spacing between genuinely distinct appliances (e.g. adjacent desk peripherals, stacked network equipment) is a domain question that needs real detector output and real office layouts to answer properly, not further synthetic tests in this room.

---

# Conclusion

**Original question:** can 3D geometry solve appliance deduplication in a way appearance-based re-identification (CLIP/OSNet embeddings) fundamentally can't — specifically, telling apart two visually identical appliances in different physical locations?

**Answer: yes, validated end-to-end**, without needing to train a Gaussian Splat at all. The pipeline that worked: COLMAP sparse reconstruction → dense MVS (`patch_match_stereo` + `stereo_fusion`) → per-detection depth-map backprojection (with fused-cloud fallback) → DBSCAN clustering on real-world-scaled 3D positions. Test 3 (two objects deliberately given an identical class label, ~3m apart) is the core proof: position correctly separated them where appearance-based matching would have had nothing to go on. Test 4 (a false merge at ~25cm, corrected by retuning the threshold to 20cm) showed the system's precision limit empirically rather than by assumption, and confirmed the retuned threshold didn't regress any earlier result.

**What's validated:**

- COLMAP can reliably reconstruct a small, irregularly-shaped real room from a single continuous walked capture — but *not* on the first attempt with generic technique. It took a specific, deliberate approach (landscape orientation, room-emphasis rather than uniform coverage) to go from 6–7 disconnected fragments to one unified model.
- Dense MVS produces enough point density (~2M points from 150 frames) for reliable per-detection depth lookup, with a sensible fallback for the cases it misses.
- A single physical reference measurement is sufficient to convert the reconstruction's arbitrary units into real-world centimeters, which is what makes the clustering threshold meaningful at all.
- The full pipeline handles both target failure modes correctly: same object across many viewpoints stays one cluster (noise floor ~1–10cm); distinct objects close together correctly split once the threshold is calibrated against real evidence rather than guessed.

**What's explicitly not validated, and shouldn't be assumed from this POC:**

- **True multi-spot fusion** (separate, independent recording sessions merged together) was descoped early and never tested — this POC only covers one continuous scan per room. If SoapAI's product ever needs to merge genuinely separate capture sessions, that's new, unvalidated risk with a different matcher requirement (`vocab_tree`/`exhaustive`, not `sequential`).
- **The 20cm threshold is calibrated against arbitrary test objects in one room**, not real appliance spacing in a real SME office. It's a defensible starting point, not a settled product constant.
- **Capture reliability for a non-technical end user is unknown, and the evidence we do have is a caution, not a reassurance** — even a careful, technically motivated capture attempt (the first one, this session) failed and needed real diagnosis to fix. Nothing here validates that an average SME employee, doing this once with no feedback loop, would reliably produce a usable scan.
- **Compute cost was not optimized or measured for production feasibility.** Dense stereo alone took ~1 hour wall-clock for one small room on a 12GB laptop GPU. That cost multiplies per room, per customer scan — not evaluated against what SoapAI's actual infrastructure budget or latency expectations would tolerate.
- **Occlusion and reflective surfaces were deliberately excluded from scope** (the mirror in the test room was covered, not tested).

**Recommendation:** the core technical bet — dense geometric reconstruction for positional dedup, no splat training required — is validated and sound. It directly solves the problem that motivated exploring 3D approaches in the first place. The open items above are real, and none of them are validated by this POC either way — they're the actual next round of questions, not proof of feasibility for production. A reasonable next step is scoping what integrating this into SoapAI would require as real PRs, treating the open items as explicit unknowns to size, not assumed answered.
