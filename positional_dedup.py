"""
positional_dedup.py — POC spike, not production code.

Given 2D appliance detections (bbox + class) across frames of a single
continuous room scan, estimate each detection's 3D world position using
COLMAP's per-frame MVS depth maps, then cluster same-class detections
with DBSCAN to get a deduplicated appliance count.

Pipeline assumed already run:
  colmap image_undistorter  -> colmap_output_v2/dense/
  colmap patch_match_stereo -> colmap_output_v2/dense/stereo/depth_maps/*.geometric.bin
  colmap stereo_fusion      -> colmap_output_v2/dense/fused.ply
  colmap model_converter (dense/sparse -> dense/sparse_txt, TXT format) — required before running this

Usage:
    python3 positional_dedup.py
"""

import json
import struct
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

try:
    from plyfile import PlyData
except ImportError:
    raise SystemExit("Missing dependency. Run: pip install plyfile scikit-learn scipy")


# --- Config -----------------------------------------------------------------

DENSE_DIR = Path("colmap_output_v2/dense")
SPARSE_TXT_DIR = DENSE_DIR / "sparse_txt"
DEPTH_MAPS_DIR = DENSE_DIR / "stereo" / "depth_maps"
FUSED_PLY_PATH = DENSE_DIR / "fused.ply"
DETECTIONS_PATH = Path("example_detections.json")

SCALE_CM_PER_UNIT = 19.2985  # from FINDINGS.md monitor calibration

BBOX_INSET_FRACTION = 0.175   # skip outer 17.5% of the box on each side
GRID_SIZE = 5                 # 5x5 sample grid inside the inset region

DEDUP_MIN_SPACING_CM = 20.0   # two detections closer than this = same appliance
                               # tuned from empirical bounds: same-object noise floor ~1-10cm,
                               # first observed false-merge (two distinct objects) at ~25cm.
                               # 20cm sits above the noise floor with margin, below the one
                               # confirmed too-close-together failure. Not yet validated against
                               # real appliance spacing — see FINDINGS.md.
DBSCAN_EPS_UNITS = DEDUP_MIN_SPACING_CM / SCALE_CM_PER_UNIT
DBSCAN_MIN_SAMPLES = 3        # tune once real detector output exists


# --- COLMAP depth map reader (standard COLMAP binary format) ----------------

def read_colmap_array(path: Path) -> np.ndarray:
    with open(path, "rb") as fid:
        header = b""
        while header.count(b"&") < 3:
            header += fid.read(1)
        width, height, channels = map(int, header.rstrip(b"&").split(b"&"))
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


# --- COLMAP TXT sparse model parser ------------------------------------------

def quat_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])


def load_cameras(cameras_txt: Path) -> dict:
    cameras = {}
    for line in cameras_txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = list(map(float, parts[4:]))
        if model != "PINHOLE":
            print(f"WARNING: camera {cam_id} is {model}, expected PINHOLE post-undistortion. "
                  f"Backprojection below assumes no distortion terms.")
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        cameras[cam_id] = dict(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
    return cameras


def load_images(images_txt: Path) -> dict:
    images = {}
    lines = images_txt.read_text().splitlines()
    data_lines = [l for l in lines if l and not l.startswith("#")]
    for i in range(0, len(data_lines), 2):
        parts = data_lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        R = quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        C = -R.T @ t  # camera center in world coordinates
        images[name] = dict(R=R, t=t, C=C, camera_id=cam_id)
    return images


# --- Fused point cloud --------------------------------------------------------

def load_fused_points(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


# --- Geometry -----------------------------------------------------------------

def sample_grid(bbox, image_width, image_height, inset=BBOX_INSET_FRACTION, grid_size=GRID_SIZE):
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    x_min_in = x_min + w * inset
    x_max_in = x_max - w * inset
    y_min_in = y_min + h * inset
    y_max_in = y_max - h * inset
    us = np.linspace(x_min_in, x_max_in, grid_size)
    vs = np.linspace(y_min_in, y_max_in, grid_size)
    points = []
    for u in us:
        for v in vs:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < image_width and 0 <= vi < image_height:
                points.append((ui, vi))
    return points


def pixel_depth_to_world(u, v, depth, cam, R, C):
    x_cam = (u - cam["cx"]) / cam["fx"] * depth
    y_cam = (v - cam["cy"]) / cam["fy"] * depth
    z_cam = depth
    p_cam = np.array([x_cam, y_cam, z_cam])
    return R.T @ p_cam + C


def project_world_points(points_world, R, t, cam):
    """Project Nx3 world points into this frame's pixel coords. Returns (uv, depth, valid_mask)."""
    p_cam = (R @ points_world.T).T + t  # Nx3
    depth = p_cam[:, 2]
    valid = depth > 0
    u = cam["fx"] * p_cam[:, 0] / np.where(valid, depth, 1) + cam["cx"]
    v = cam["fy"] * p_cam[:, 1] / np.where(valid, depth, 1) + cam["cy"]
    in_bounds = valid & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
    return u, v, depth, in_bounds


# --- Per-detection position estimation -----------------------------------------

def estimate_detection_position(detection, cameras, images, fused_points):
    image_name = detection["image"]
    bbox = detection["bbox"]

    if image_name not in images:
        print(f"  SKIP: {image_name} not found in reconstruction (frame didn't register)")
        return None

    img = images[image_name]
    cam = cameras[img["camera_id"]]

    depth_map_path = DEPTH_MAPS_DIR / f"{image_name}.geometric.bin"
    world_points = []

    if depth_map_path.exists():
        depth_map = read_colmap_array(depth_map_path)
        for u, v in sample_grid(bbox, cam["width"], cam["height"]):
            depth = depth_map[v, u]  # depth map indexed [row, col] = [v, u]
            if depth > 0:
                world_points.append(pixel_depth_to_world(u, v, depth, cam, img["R"], img["C"]))
    else:
        print(f"  WARNING: no depth map for {image_name}, going straight to fallback")

    if world_points:
        position = np.median(np.array(world_points), axis=0)
        return dict(position=position, source="depth_map", num_samples=len(world_points))

    # --- Fallback: reproject fused cloud into this frame, filter to bbox ---
    u, v, depth, in_bounds = project_world_points(fused_points, img["R"], img["t"], cam)
    x_min, y_min, x_max, y_max = bbox
    inset_w = (x_max - x_min) * BBOX_INSET_FRACTION
    inset_h = (y_max - y_min) * BBOX_INSET_FRACTION
    in_bbox = (
        in_bounds
        & (u >= x_min + inset_w) & (u <= x_max - inset_w)
        & (v >= y_min + inset_h) & (v <= y_max - inset_h)
    )
    matched_points = fused_points[in_bbox]

    if len(matched_points) == 0:
        print(f"  FAILED: no depth data and no fused-cloud fallback matches for "
              f"{image_name} bbox={bbox} (class={detection['class']})")
        return None

    position = np.median(matched_points, axis=0)
    return dict(position=position, source="fused_cloud_fallback", num_samples=len(matched_points))


# --- Main ------------------------------------------------------------------

def main():
    print("Loading COLMAP model...")
    cameras = load_cameras(SPARSE_TXT_DIR / "cameras.txt")
    images = load_images(SPARSE_TXT_DIR / "images.txt")
    print(f"  {len(cameras)} camera(s), {len(images)} image(s)")

    print("Loading fused point cloud (fallback source)...")
    fused_points = load_fused_points(FUSED_PLY_PATH)
    print(f"  {len(fused_points):,} points")

    if not DETECTIONS_PATH.exists():
        print(f"\nNo {DETECTIONS_PATH} found — writing a placeholder example file.")
        print("Edit it with real bbox coordinates from your frames, then rerun.\n")
        example = [
            {"image": "frame_0037.png", "class": "monitor", "bbox": [100, 200, 400, 500]},
            {"image": "frame_0041.png", "class": "monitor", "bbox": [150, 210, 430, 510]},
        ]
        DETECTIONS_PATH.write_text(json.dumps(example, indent=2))
        return

    detections = json.loads(DETECTIONS_PATH.read_text())
    print(f"\nProcessing {len(detections)} detections...")

    results = []
    for det in detections:
        r = estimate_detection_position(det, cameras, images, fused_points)
        if r is not None:
            r["class"] = det["class"]
            r["image"] = det["image"]
            results.append(r)
            print(f"  {det['image']} [{det['class']}]: "
                  f"pos={r['position'].round(3)} source={r['source']} n={r['num_samples']}")

    fallback_count = sum(1 for r in results if r["source"] == "fused_cloud_fallback")
    print(f"\n{len(results)}/{len(detections)} detections localized "
          f"({fallback_count} via fallback)")

    print(f"\nClustering (eps={DBSCAN_EPS_UNITS:.3f} units = {DEDUP_MIN_SPACING_CM}cm, "
          f"min_samples={DBSCAN_MIN_SAMPLES})...")

    by_class = defaultdict(list)
    for r in results:
        by_class[r["class"]].append(r)

    for class_name, class_results in by_class.items():
        positions = np.array([r["position"] for r in class_results])
        labels = DBSCAN(eps=DBSCAN_EPS_UNITS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(positions)
        num_clusters = len(set(labels) - {-1})
        num_noise = list(labels).count(-1)
        print(f"  {class_name}: {len(class_results)} detections -> "
              f"{num_clusters} deduplicated appliance(s) "
              f"({num_noise} flagged as noise/outliers)")


if __name__ == "__main__":
    main()
