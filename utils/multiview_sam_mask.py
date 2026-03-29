import os
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
from PIL import Image


@dataclass
class SamAutoMaskParams:
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    stability_score_offset: float = 1.0
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7
    crop_overlap_ratio: float = 512 / 1500
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 256


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _as_uint8_mask(seg: np.ndarray) -> np.ndarray:
    # seg: bool or {0,1}
    seg = seg.astype(np.uint8)
    return seg * 255


def _load_image_rgb(image_path: str) -> np.ndarray:
    im = Image.open(image_path).convert("RGB")
    return np.array(im)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_colmap_model(sparse0_dir: str):
    """Load COLMAP model (cameras/images/points3D) with track information."""
    from utils import read_write_model as rw

    images_bin = os.path.join(sparse0_dir, "images.bin")
    cameras_bin = os.path.join(sparse0_dir, "cameras.bin")
    points3d_bin = os.path.join(sparse0_dir, "points3D.bin")
    images_txt = os.path.join(sparse0_dir, "images.txt")
    cameras_txt = os.path.join(sparse0_dir, "cameras.txt")
    points3d_txt = os.path.join(sparse0_dir, "points3D.txt")

    if os.path.exists(images_bin) and os.path.exists(cameras_bin) and os.path.exists(points3d_bin):
        images = rw.read_images_binary(images_bin)
        cameras = rw.read_cameras_binary(cameras_bin)
        points3D = rw.read_points3D_binary(points3d_bin)
    else:
        images = rw.read_images_text(images_txt)
        cameras = rw.read_cameras_text(cameras_txt)
        points3D = rw.read_points3D_text(points3d_txt)
    return cameras, images, points3D


def build_sam(model_type: str, checkpoint: str, device: str):
    """Build SAM model using vendored segment_anything code."""
    # Prefer submodules/segment-anything (symlinked) for imports.
    # Callers should ensure repo root is current working directory.
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sam_submodule = os.path.join(repo_root, "submodules", "segment-anything")
    if sam_submodule not in sys.path:
        sys.path.insert(0, sam_submodule)

    from segment_anything import sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    return sam


def sam_automatic_masks(image_rgb: np.ndarray, sam, params: SamAutoMaskParams):
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sam_submodule = os.path.join(repo_root, "submodules", "segment-anything")
    if sam_submodule not in sys.path:
        sys.path.insert(0, sam_submodule)

    from segment_anything import SamAutomaticMaskGenerator

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=params.points_per_side,
        pred_iou_thresh=params.pred_iou_thresh,
        stability_score_thresh=params.stability_score_thresh,
        stability_score_offset=params.stability_score_offset,
        box_nms_thresh=params.box_nms_thresh,
        crop_n_layers=params.crop_n_layers,
        crop_nms_thresh=params.crop_nms_thresh,
        crop_overlap_ratio=params.crop_overlap_ratio,
        crop_n_points_downscale_factor=params.crop_n_points_downscale_factor,
        min_mask_region_area=params.min_mask_region_area,
    )
    masks = generator.generate(image_rgb)
    # Ensure deterministic order by sorting with (area desc, bbox)
    masks = sorted(
        masks,
        key=lambda m: (
            -int(m.get("area", 0)),
            tuple(int(x) for x in m.get("bbox", [0, 0, 0, 0])),
        ),
    )
    return masks


def _point_to_mask_index(
    image_entry,
    masks: List[dict],
    width: int,
    height: int,
    max_points: Optional[int] = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Dict[int, int]:
    """Assign each COLMAP point3D id (observed in this image) to a SAM mask index.

    Strategy: for each 2D keypoint, pick the first mask whose segmentation contains it.
    Because masks are sorted by descending area, you can control granularity via SAM params.
    """
    xys = image_entry.xys
    pids = image_entry.point3D_ids
    if max_points is not None and xys.shape[0] > max_points:
        # Uniform subsample to keep preprocessing bounded.
        idx = np.linspace(0, xys.shape[0] - 1, max_points).astype(np.int64)
        xys = xys[idx]
        pids = pids[idx]

    # Pre-pack segmentations.
    # Important: when assigning COLMAP keypoints to a SAM mask, prefer *smaller* masks first.
    # Otherwise large masks (often near-full-image) would absorb many points and break multi-view object linking.
    order = list(range(len(masks)))
    order.sort(key=lambda mi: int(masks[mi].get("area", 0)))
    segs = [masks[mi]["segmentation"] for mi in order]
    out: Dict[int, int] = {}

    for (xy, pid) in zip(xys, pids):
        if pid == -1:
            continue
        x = int(round(float(xy[0]) * float(scale_x)))
        y = int(round(float(xy[1]) * float(scale_y)))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        # Find first mask containing this pixel
        assigned = -1
        for local_i, seg in enumerate(segs):
            # seg is HxW bool
            if seg[y, x]:
                assigned = order[local_i]
                break
        if assigned != -1:
            out[int(pid)] = assigned
    return out


def _point_to_mask_indices(
    image_entry,
    masks: List[dict],
    width: int,
    height: int,
    max_points: Optional[int] = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    max_masks_per_point: int = 1,
    min_mask_area_for_linking: int = 0,
) -> Dict[int, List[int]]:
    """Assign each COLMAP point3D id (observed in this image) to up to K SAM mask indices.

    When a 2D keypoint lies in multiple nested SAM masks, allowing it to vote for a few
    *small* masks (instead of a single one) typically improves multi-view linkage stability.
    """

    if int(max_masks_per_point) <= 1:
        single = _point_to_mask_index(
            image_entry,
            masks=masks,
            width=width,
            height=height,
            max_points=max_points,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        return {pid: [mi] for pid, mi in single.items()}

    xys = image_entry.xys
    pids = image_entry.point3D_ids
    if max_points is not None and xys.shape[0] > max_points:
        idx = np.linspace(0, xys.shape[0] - 1, max_points).astype(np.int64)
        xys = xys[idx]
        pids = pids[idx]

    # Prefer smaller masks first.
    order = list(range(len(masks)))
    order.sort(key=lambda mi: int(masks[mi].get("area", 0)))

    segs: List[Tuple[int, np.ndarray]] = []
    for mi in order:
        area = int(masks[mi].get("area", 0))
        if int(min_mask_area_for_linking) > 0 and area < int(min_mask_area_for_linking):
            continue
        segs.append((int(mi), masks[mi]["segmentation"]))

    out: Dict[int, List[int]] = {}
    for (xy, pid) in zip(xys, pids):
        if pid == -1:
            continue
        x = int(round(float(xy[0]) * float(scale_x)))
        y = int(round(float(xy[1]) * float(scale_y)))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue

        chosen: List[int] = []
        for mi, seg in segs:
            if seg[y, x]:
                chosen.append(int(mi))
                if len(chosen) >= int(max_masks_per_point):
                    break
        if chosen:
            out[int(pid)] = chosen
    return out


def generate_multiview_consistent_masks(
    source_path: str,
    images_subdir: str = "images",
    sparse_subdir: str = os.path.join("sparse", "0"),
    out_subdir: str = os.path.join("images", "masks_sam"),
    model_type: str = "vit_h",
    checkpoint: str = "",
    device: str = "cuda",
    sam_params: Optional[SamAutoMaskParams] = None,
    min_shared_points: int = 20,
    min_shared_ratio: float = 0.0,
    max_points_per_image: Optional[int] = 5000,
    max_masks_per_point: int = 1,
    min_mask_area_for_linking: int = 0,
    max_images: Optional[int] = None,
    dry_run: bool = False,
    num_workers: int = 0,
) -> str:
    """Generate SAM masks for each image and merge them into multiview-consistent object ids.

    Output layout (per image):
      <source_path>/<out_subdir>/<image_stem>/obj_XXXX.png

    Returns absolute output directory.
    """
    if sam_params is None:
        sam_params = SamAutoMaskParams()

    sparse0_dir = os.path.join(source_path, sparse_subdir)
    cameras, images, points3D = load_colmap_model(sparse0_dir)

    # Sort images by name for determinism
    image_items = sorted(images.items(), key=lambda kv: kv[1].name)
    if max_images is not None:
        image_items = image_items[: int(max_images)]

    sam = None
    if not dry_run:
        if checkpoint == "":
            raise ValueError("SAM checkpoint path is empty. Please pass --sam_checkpoint.")
        sam = build_sam(model_type=model_type, checkpoint=checkpoint, device=device)

    # 1) Run SAM on each image
    per_image_masks: Dict[int, List[dict]] = {}
    per_image_pid_to_masks: Dict[int, Dict[int, List[int]]] = {}
    node_ids: Dict[Tuple[int, int], int] = {}  # (image_id, mask_idx) -> node id
    node_list: List[Tuple[int, int]] = []

    executor = ThreadPoolExecutor(max_workers=int(num_workers)) if int(num_workers) > 0 else None
    pid2ms_futures: Dict[int, object] = {}
    total_images = len(image_items)
    
    pbar = tqdm(image_items, total=total_images, desc="SAM masks")
    for image_id, im in pbar:
        cam = cameras[im.camera_id]
        orig_w, orig_h = int(cam.width), int(cam.height)
        image_path = os.path.join(source_path, images_subdir, im.name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        rgb = _load_image_rgb(image_path)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        if dry_run:
            masks = [{"segmentation": np.ones((height, width), dtype=bool), "area": height * width, "bbox": [0, 0, width, height]}]
        else:
            masks = sam_automatic_masks(rgb, sam, sam_params)
        per_image_masks[image_id] = masks

        scale_x = float(width) / float(orig_w) if orig_w > 0 else 1.0
        scale_y = float(height) / float(orig_h) if orig_h > 0 else 1.0

        if executor is None:
            pid2ms = _point_to_mask_indices(
                im,
                masks,
                width=width,
                height=height,
                max_points=max_points_per_image,
                scale_x=scale_x,
                scale_y=scale_y,
                max_masks_per_point=int(max_masks_per_point),
                min_mask_area_for_linking=int(min_mask_area_for_linking),
            )
            per_image_pid_to_masks[image_id] = pid2ms
        else:
            pid2ms_futures[image_id] = executor.submit(
                _point_to_mask_indices,
                im,
                masks,
                width=width,
                height=height,
                max_points=max_points_per_image,
                scale_x=scale_x,
                scale_y=scale_y,
                max_masks_per_point=int(max_masks_per_point),
                min_mask_area_for_linking=int(min_mask_area_for_linking),
            )

        for mi in range(len(masks)):
            node_ids[(image_id, mi)] = len(node_list)
            node_list.append((image_id, mi))

        pbar.update(1)
        pbar.set_postfix_str(f"{im.name} (masks={len(masks)})")
        # if (ii + 1) % 1 == 0 or (ii + 1) == total_images:
        #     print(f"[SAM] processed {ii + 1}/{total_images}: {im.name} (masks={len(masks)})")

    if executor is not None:
        for image_id, fut in tqdm(pid2ms_futures.items(), total=len(pid2ms_futures), desc="point-mask linking"):
            per_image_pid_to_masks[image_id] = fut.result()

    # 2) Accumulate segment co-visibility edges via COLMAP tracks
    pair_counts: Dict[Tuple[int, int], int] = {}

    # Node degree proxy: number of 3D points that vote for each node (image,mask).
    # Used by min_shared_ratio to avoid linking huge nodes via a tiny overlap.
    node_point_counts: List[int] = [0] * len(node_list)
    for img_id, pid2ms in per_image_pid_to_masks.items():
        for _, mis in pid2ms.items():
            for mi in mis:
                nid = node_ids.get((int(img_id), int(mi)))
                if nid is not None:
                    node_point_counts[nid] += 1

    for pid, p3d in points3D.items():
        # Collect observed segment nodes for this 3D point
        obs_nodes: List[int] = []
        for img_id in p3d.image_ids:
            img_id = int(img_id)
            if img_id not in per_image_pid_to_masks:
                continue
            mis = per_image_pid_to_masks[img_id].get(int(pid), [])
            if not mis:
                continue
            for mi in mis:
                n = node_ids.get((img_id, int(mi)), None)
                if n is None:
                    continue
                obs_nodes.append(n)

        if len(obs_nodes) < 2:
            continue

        # Unique per point to avoid overweighting from duplicate obs
        obs_nodes = sorted(set(obs_nodes))
        for a, b in itertools.combinations(obs_nodes, 2):
            if a == b:
                continue
            if a > b:
                a, b = b, a
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    # 3) Union segments into multiview objects
    uf = UnionFind(len(node_list))
    edges = []
    for (a, b), cnt in pair_counts.items():
        if cnt < int(min_shared_points):
            continue
        if float(min_shared_ratio) > 0.0:
            denom = min(int(node_point_counts[a]), int(node_point_counts[b]))
            if denom <= 0:
                continue
            if (float(cnt) / float(denom)) < float(min_shared_ratio):
                continue
        edges.append((cnt, a, b))
    edges.sort(key=lambda t: -t[0])
    for cnt, a, b in edges:
        uf.union(a, b)

    # 4) Build component id mapping
    root_to_obj: Dict[int, int] = {}
    node_to_obj: List[int] = [-1] * len(node_list)
    for nid in range(len(node_list)):
        r = uf.find(nid)
        if r not in root_to_obj:
            root_to_obj[r] = len(root_to_obj)
        node_to_obj[nid] = root_to_obj[r]

    # Quick consistency report: how often a COLMAP points3D votes into a single object id.
    # This is not a proof, but a useful sanity metric to detect weak multi-view linkage.
    total_pts = 0
    consistent_pts = 0
    ambiguous_pts = 0
    for pid, p3d in points3D.items():
        obj_ids = set()
        for img_id in p3d.image_ids:
            img_id = int(img_id)
            if img_id not in per_image_pid_to_masks:
                continue
            for mi in per_image_pid_to_masks[img_id].get(int(pid), []):
                nid = node_ids.get((img_id, int(mi)))
                if nid is None:
                    continue
                obj_ids.add(int(node_to_obj[nid]))
        if not obj_ids:
            continue
        total_pts += 1
        if len(obj_ids) == 1:
            consistent_pts += 1
        else:
            ambiguous_pts += 1
    if total_pts > 0:
        frac = 100.0 * float(consistent_pts) / float(total_pts)
        print(
            f"[Consistency] points3D single-object={consistent_pts}/{total_pts} ({frac:.2f}%), "
            f"multi-object={ambiguous_pts}/{total_pts}"
        )

    # 5) Write masks per image and object
    out_dir = os.path.join(source_path, out_subdir)
    _ensure_dir(out_dir)

    # Pre-group masks per image and per object
    per_image_obj_masks: Dict[int, Dict[int, np.ndarray]] = {}
    for nid, (img_id, mi) in enumerate(node_list):
        obj_id = node_to_obj[nid]
        seg = per_image_masks[img_id][mi]["segmentation"]
        if img_id not in per_image_obj_masks:
            per_image_obj_masks[img_id] = {}
        if obj_id not in per_image_obj_masks[img_id]:
            per_image_obj_masks[img_id][obj_id] = seg.copy()
        else:
            per_image_obj_masks[img_id][obj_id] |= seg

    save_futures = []
    for image_id, im in image_items:
        stem = os.path.splitext(os.path.basename(im.name))[0]
        out_img_dir = os.path.join(out_dir, stem)
        _ensure_dir(out_img_dir)

        obj_masks = per_image_obj_masks.get(image_id, {})
        if executor is None:
            for obj_id in sorted(obj_masks.keys()):
                m = _as_uint8_mask(obj_masks[obj_id])
                Image.fromarray(m, mode="L").save(os.path.join(out_img_dir, f"obj_{obj_id:04d}.png"))
        else:
            for obj_id in sorted(obj_masks.keys()):
                seg = obj_masks[obj_id]
                save_futures.append(executor.submit(_save_single_obj_mask, out_img_dir, int(obj_id), seg))

    if executor is not None and save_futures:
        for fut in tqdm(as_completed(save_futures), total=len(save_futures), desc="saving masks"):
            fut.result()

    if executor is not None:
        executor.shutdown(wait=True)

    return os.path.abspath(out_dir)


def _save_single_obj_mask(out_img_dir: str, obj_id: int, seg: np.ndarray) -> None:
    m = _as_uint8_mask(seg)
    Image.fromarray(m, mode="L").save(os.path.join(out_img_dir, f"obj_{obj_id:04d}.png"))
