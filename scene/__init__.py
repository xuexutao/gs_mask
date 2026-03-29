#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import geom_transform_points
import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], mask_opt=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        # Keep dataset/mask context for optional per-object export.
        self._source_path = getattr(args, "source_path", None)
        self._images_subdir = "images" if getattr(args, "images", None) is None else str(getattr(args, "images"))
        self._mask_opt = mask_opt

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, True)

        print("donw")
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def assign_semantic_labels(self, semantic_labels: Dict, category_to_id: Dict[str, int] = None):
        """为高斯点云分配语义标签。
        
        参数:
            semantic_labels: 从对象ID到类别信息的映射，格式为 {obj_id: {"category": "chair", "confidence": 0.9}}
            category_to_id: 从类别名称到整数ID的映射。如果为None，则自动创建。
        """
        print(f"[DEBUG assign_semantic_labels] Called with {len(semantic_labels) if semantic_labels else 0} labels")
        if self._mask_opt is not None:
            print(f"[DEBUG] mask_opt attributes: {dir(self._mask_opt)}")
            for attr in ['mask_dir', 'mask_dirname']:
                if hasattr(self._mask_opt, attr):
                    print(f"[DEBUG] mask_opt.{attr} = {getattr(self._mask_opt, attr)}")
        print(f"[DEBUG] self._source_path: {self._source_path}")
        print(f"[DEBUG] self._images_subdir: {self._images_subdir}")
        
        if semantic_labels is None or len(semantic_labels) == 0:
            print("Warning: No semantic labels provided")
            return
        
        if self._mask_opt is None:
            print("Warning: Cannot assign semantic labels without mask options")
            print("[DEBUG] Returning early because self._mask_opt is None")
            return
        
        if not self._source_path:
            print("Warning: Cannot assign semantic labels without source path")
            print("[DEBUG] Returning early because self._source_path is empty")
            return
        
        print(f"Assigning semantic labels to {len(semantic_labels)} objects...")
        
        # 创建类别到ID的映射
        if category_to_id is None:
            # 收集所有唯一的类别
            categories = set()
            for obj_info in semantic_labels.values():
                categories.add(obj_info["category"])
            categories = sorted(list(categories))
            category_to_id = {cat: i for i, cat in enumerate(categories)}
            print(f"Created category mapping: {category_to_id}")
        
        # 获取所有对象ID
        obj_ids = []
        for obj_id_str in semantic_labels.keys():
            try:
                obj_id = int(obj_id_str)
                obj_ids.append(obj_id)
            except ValueError:
                print(f"Warning: Invalid object ID format: {obj_id_str}")
                continue
        
        if not obj_ids:
            print("Warning: No valid object IDs found")
            return
        
        print(f"[DEBUG] Using all {len(obj_ids)} object IDs for semantic labeling")
        
        # 优先使用 mask_opt 中的 mask_dir（完整路径）
        mask_root = None
        if hasattr(self._mask_opt, 'mask_dir') and self._mask_opt.mask_dir:
            mask_root = str(self._mask_opt.mask_dir)
            print(f"Using mask directory from mask_opt.mask_dir: {mask_root}")
            # 尝试从 mask_dir 推断 images 子目录
            try:
                # mask_dir 格式应为: <source_path>/<images_subdir>/<mask_dirname>
                rel_path = os.path.relpath(mask_root, self._source_path)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2 and parts[0] != '..':
                    inferred_images = parts[0]
                    if inferred_images != self._images_subdir:
                        print(f"[INFO] Inferred images subdir from mask_dir: {inferred_images} (was {self._images_subdir})")
                        self._images_subdir = inferred_images
            except ValueError:
                pass  # 无法推断，保持原样
        else:
            mask_dirname = str(getattr(self._mask_opt, "mask_dirname", "masks_sam"))
            mask_root = os.path.join(self._source_path, self._images_subdir, mask_dirname)
            print(f"Constructed mask directory: {mask_root}")
        
        print(f"[DEBUG] Checking mask directory existence: {mask_root}")
        if not os.path.isdir(mask_root):
            print(f"Warning: Mask directory not found: {mask_root}")
            return
        print(f"[DEBUG] Mask directory found, proceeding with voting...")
        
        # 使用与 _export_object_gaussians 类似的投票逻辑
        cams = self.getTrainCameras()
        if not cams:
            print("Warning: No training cameras available")
            return
        
        # 选择包含每个对象掩码的视图进行投票
        cam_by_stem = {}
        for cam in cams:
            stem = os.path.splitext(os.path.basename(cam.image_name))[0]
            cam_by_stem[stem] = cam
        
        vote_views = []
        used = set()
        for obj_id in obj_ids:
            found = None
            for stem, cam in cam_by_stem.items():
                p = os.path.join(mask_root, stem, f"obj_{int(obj_id):04d}.png")
                if os.path.exists(p):
                    found = cam
                    break
            if found is not None:
                key = os.path.splitext(os.path.basename(found.image_name))[0]
                if key not in used:
                    used.add(key)
                    vote_views.append(found)
        
        # 回退：如果没有找到视图，使用一些视图
        if not vote_views:
            max_views = 24
            if len(cams) <= max_views:
                vote_views = cams
            else:
                step = max(int(len(cams) // max_views), 1)
                vote_views = cams[::step][:max_views]
        
        print(f"[DEBUG] Using {len(vote_views)} vote views for semantic labeling")
        
        xyz = self.gaussians.get_xyz.detach()  # (N,3) on cuda
        N = int(xyz.shape[0])
        if N == 0:
            print("Warning: No Gaussians to assign labels to")
            return
        
        # 初始化语义标签为-1（未标记）
        semantic_tensor = torch.full((N,), -1, dtype=torch.long, device=xyz.device)
        
        # 缓存掩码以避免重复加载
        _mask_cache = {}
        def _load_mask(stem: str, obj_id: int, H: int, W: int, device) -> Optional[torch.Tensor]:
            cache_key = (stem, obj_id, H, W)
            if cache_key in _mask_cache:
                return _mask_cache[cache_key].to(device=device)
            p = os.path.join(mask_root, stem, f"obj_{int(obj_id):04d}.png")
            if not os.path.exists(p):
                return None
            m = Image.open(p).convert("L")
            arr = torch.from_numpy(np.array(m, dtype=np.uint8)).float() / 255.0
            if int(arr.shape[0]) != int(H) or int(arr.shape[1]) != int(W):
                arr = F.interpolate(arr[None, None, ...], size=(int(H), int(W)), mode="nearest")[0, 0]
            tensor = arr[None, ...].to(device=device)
            _mask_cache[cache_key] = tensor.cpu()  # 存储到CPU以减少GPU内存
            return tensor
        
        # 为每个对象投票
        # total_objects = len(obj_ids)
        # processed = 0
        # for cam in vote_views:
        #     H, W = int(cam.image_height), int(cam.image_width)
        #     stem = os.path.splitext(os.path.basename(cam.image_name))[0]
        #     # 投影到NDC
        #     ndc = geom_transform_points(xyz, cam.full_proj_transform)
        #     x = ndc[:, 0]
        #     y = ndc[:, 1]
        #     z = ndc[:, 2]
        #     ix = ((x + 1.0) * 0.5 * float(W)).long()
        #     iy = ((1.0 - y) * 0.5 * float(H)).long()
        #     valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (z > 0)
        #     if not bool(valid.any().item()):
        #         continue
        #     vidx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        #     ixv = ix[vidx]
        #     iyv = iy[vidx]
            
        #     # for obj_id in obj_ids:
        #     obj_files = os.listdir(os.path.join(mask_root, stem))
        #     obj_ids_in_view = [
        #         int(f.split('_')[1].split('.')[0])
        #         for f in obj_files if f.startswith("obj_")
        #     ]
        #     for obj_id in obj_ids_in_view:
        #         # 获取对象的类别
        #         obj_info = semantic_labels.get(str(obj_id))
        #         if obj_info is None:
        #             continue
        #         category = obj_info["category"]
        #         category_id = category_to_id.get(category)
        #         if category_id is None:
        #             print(f"Warning: Unknown category '{category}' for object {obj_id}")
        #             continue
                
        #         m = _load_mask(stem, obj_id, H, W, xyz.device)
        #         if m is None:
        #             continue
                
        #         # 检查哪些高斯点在掩码内
        #         inside = m[0, iyv, ixv] > 0.5
        #         if inside.any():
        #             # 为在掩码内的高斯点分配类别ID
        #             # 如果有冲突（同一个高斯点属于多个对象），使用第一个遇到的类别
        #             mask_indices = vidx[inside]
        #             for idx in mask_indices:
        #                 if semantic_tensor[idx] == -1:  # 只分配未标记的点
        #                     semantic_tensor[idx] = category_id
        # ===== 参数 =====
        MAX_WORKERS = 8  # 可调：4~16（看磁盘IO能力）

        # ===== 主逻辑 =====
        total_objects = len(obj_ids)
        processed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for cam in tqdm(vote_views, desc="Semantic voting (views)"):
                H, W = int(cam.image_height), int(cam.image_width)
                stem = os.path.splitext(os.path.basename(cam.image_name))[0]
                ndc = geom_transform_points(xyz, cam.full_proj_transform)
                x, y, z = ndc[:, 0], ndc[:, 1], ndc[:, 2]

                ix = ((x + 1.0) * 0.5 * float(W)).long()
                iy = ((1.0 - y) * 0.5 * float(H)).long()

                valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (z > 0)
                if not bool(valid.any().item()):
                    continue

                vidx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                ixv = ix[vidx]
                iyv = iy[vidx]

                # ===== 当前view有哪些obj =====
                view_dir = os.path.join(mask_root, stem)
                if not os.path.isdir(view_dir):
                    continue

                obj_files = os.listdir(view_dir)
                obj_ids_in_view = [
                    int(f.split('_')[1].split('.')[0])
                    for f in obj_files if f.startswith("obj_")
                ]

                if not obj_ids_in_view:
                    continue

                # ===== 多线程加载 masks =====
                futures = {}
                for obj_id in obj_ids_in_view:
                    futures[executor.submit(_load_mask, stem, obj_id, H, W, xyz.device)] = obj_id

                # ===== 处理返回结果 =====
                for future in as_completed(futures):
                    obj_id = futures[future]

                    try:
                        m = future.result()
                    except Exception as e:
                        print(f"[WARN] mask load failed for obj {obj_id}: {e}")
                        continue

                    if m is None:
                        continue

                    obj_info = semantic_labels.get(str(obj_id))
                    if obj_info is None:
                        continue

                    category = obj_info["category"]
                    category_id = category_to_id.get(category)
                    if category_id is None:
                        continue

                    # ===== mask 判断 =====
                    inside = m[0, iyv, ixv] > 0.5
                    if not inside.any():
                        continue

                    mask_indices = vidx[inside]

                    # ===== 向量化赋值（关键优化）=====
                    unassigned = semantic_tensor[mask_indices] == -1
                    if unassigned.any():
                        semantic_tensor[mask_indices[unassigned]] = category_id

                processed += 1

                # ===== 可选 debug =====
                if processed % 20 == 0:
                    print(f"[DEBUG] processed {processed}/{len(vote_views)} views")
        
        # 将语义标签分配给高斯模型
        print(f"[DEBUG] Setting gaussians._semantic with shape {semantic_tensor.shape}")
        self.gaussians._semantic = semantic_tensor
        
        # 统计分配情况
        assigned = (semantic_tensor != -1).sum().item()
        print(f"Assigned semantic labels to {assigned}/{N} Gaussians ({assigned/N*100:.1f}%)")
        
        # 打印每个类别的统计信息
        category_counts = {}
        for cat_name, cat_id in category_to_id.items():
            count = (semantic_tensor == cat_id).sum().item()
            if count > 0:
                category_counts[cat_name] = count
        
        print("Category distribution:")
        for cat_name, count in category_counts.items():
            print(f"  {cat_name}: {count} Gaussians")

    def assign_semantic_labels_multi(self, semantic_labels: Dict, category_to_id: Dict[str, int] = None):
        

        print(f"[DEBUG] assign_semantic_labels_multi called")

        if not semantic_labels:
            print("Warning: No semantic labels provided")
            return

        if self._mask_opt is None or not self._source_path:
            print("Warning: Missing mask options or source path")
            return

        # ===== 类别映射 =====
        if category_to_id is None:
            categories = sorted({v["category"] for v in semantic_labels.values()})
            category_to_id = {c: i for i, c in enumerate(categories)}
            print(f"[DEBUG] category_to_id: {category_to_id}")

        # ===== obj_ids =====
        obj_ids = []
        for k in tqdm(semantic_labels.keys(), desc="Loading semantic labels"):
            try:
                obj_ids.append(int(k))
            except:
                continue

        if not obj_ids:
            print("Warning: no valid object ids")
            return

        # ===== mask_root =====
        if hasattr(self._mask_opt, 'mask_dir') and self._mask_opt.mask_dir:
            mask_root = str(self._mask_opt.mask_dir)
        else:
            mask_root = os.path.join(
                self._source_path,
                self._images_subdir,
                getattr(self._mask_opt, "mask_dirname", "masks_sam")
            )

        if not os.path.isdir(mask_root):
            print(f"Warning: mask dir not found: {mask_root}")
            return

        # ===== cameras =====
        cams = self.getTrainCameras()
        if not cams:
            print("Warning: no cameras")
            return

        cam_by_stem = {
            os.path.splitext(os.path.basename(c.image_name))[0]: c
            for c in cams
        }

        # ===== vote_views =====
        # vote_views = []
        # used = set()
        # for obj_id in tqdm(obj_ids, desc="Loading vote views"):
        #     for stem, cam in cam_by_stem.items():
        #         p = os.path.join(mask_root, stem, f"obj_{obj_id:04d}.png")
        #         if os.path.exists(p):
        #             if stem not in used:
        #                 vote_views.append(cam)
        #                 used.add(stem)
        #             break

        # if not vote_views:
        #     vote_views = cams[:24]

        # print(f"[DEBUG] vote views: {len(vote_views)}")
        vote_views = []
        used = set()
        def check_obj(obj_id):
            local_result = []
            for stem, cam in cam_by_stem.items():
                p = os.path.join(mask_root, stem, f"obj_{obj_id:04d}.png")
                if os.path.exists(p):
                    local_result.append((stem, cam))
                    break
            return local_result

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(check_obj, obj_id): obj_id for obj_id in obj_ids}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Loading vote views"):
                for stem, cam in f.result():
                    if stem not in used:
                        vote_views.append(cam)
                        used.add(stem)

        if not vote_views:
            vote_views = cams[:24]

        print(f"[DEBUG] vote views: {len(vote_views)}")

        # ===== gaussians =====
        xyz = self.gaussians.get_xyz.detach()
        device = xyz.device
        N = xyz.shape[0]

        if N == 0:
            print("Warning: no gaussians")
            return

        semantic_tensor = torch.full((N,), -1, dtype=torch.long, device=device)

        # ===== 预计算投影（重要优化）=====
        print("[DEBUG] precomputing projections...")
        proj_cache = []
        for cam in tqdm(vote_views, desc="Precomputing projections"):
            ndc = geom_transform_points(xyz, cam.full_proj_transform)
            proj_cache.append(ndc)

        # ===== 主循环 =====
        print("[DEBUG] start semantic voting...")
        for cam_idx, cam in enumerate(tqdm(vote_views, desc="Semantic voting (views)")):
            H, W = int(cam.image_height), int(cam.image_width)
            stem = os.path.splitext(os.path.basename(cam.image_name))[0]

            ndc = proj_cache[cam_idx]
            x, y, z = ndc[:, 0], ndc[:, 1], ndc[:, 2]

            ix = ((x + 1.0) * 0.5 * W).long()
            iy = ((1.0 - y) * 0.5 * H).long()

            valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (z > 0)
            if not valid.any():
                continue

            vidx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            ixv = ix[vidx]
            iyv = iy[vidx]

            view_dir = os.path.join(mask_root, stem)
            if not os.path.isdir(view_dir):
                continue

            obj_files = [f for f in os.listdir(view_dir) if f.startswith("obj_")]
            if not obj_files:
                continue

            masks = []
            cat_ids = []

            # ===== 加载所有mask =====
            for f in obj_files:
                try:
                    obj_id = int(f.split('_')[1].split('.')[0])
                except:
                    continue

                obj_info = semantic_labels.get(str(obj_id))
                if obj_info is None:
                    continue

                cat_id = category_to_id.get(obj_info["category"])
                if cat_id is None:
                    continue

                p = os.path.join(view_dir, f)

                try:
                    m = Image.open(p).convert("L")
                    arr = torch.from_numpy(np.array(m, dtype=np.uint8)).float() / 255.0
                except:
                    continue

                if arr.shape[0] != H or arr.shape[1] != W:
                    arr = F.interpolate(arr[None, None], size=(H, W), mode="nearest")[0, 0]

                masks.append(arr)
                cat_ids.append(cat_id)

            if not masks:
                continue

            masks = torch.stack(masks, dim=0).to(device)   # (K, H, W)
            cat_tensor = torch.tensor(cat_ids, device=device)

            # ===== 核心：一次性判断 =====
            sampled = masks[:, iyv, ixv] > 0.5   # (K, M)

            hit_any = sampled.any(dim=0)
            if not hit_any.any():
                continue

            first_hit = sampled.float().argmax(dim=0)

            valid_pts = vidx[hit_any]
            chosen = first_hit[hit_any]

            assign_ids = cat_tensor[chosen]

            # ===== 只赋值未标记 =====
            unassigned = semantic_tensor[valid_pts] == -1
            semantic_tensor[valid_pts[unassigned]] = assign_ids[unassigned]

        # device = semantic_tensor.device
        # print("[DEBUG] start semantic voting (GPU)...")
        # all_masks = []
        # all_cat_ids = []
        # all_ixv = []
        # all_iyv = []
        # all_vidx = []

        # # ===== 1. 准备所有mask和投影索引 =====
        # for cam_idx, cam in enumerate(tqdm(vote_views, desc="Preparing masks")):
        #     H, W = int(cam.image_height), int(cam.image_width)
        #     stem = os.path.splitext(os.path.basename(cam.image_name))[0]

        #     ndc = proj_cache[cam_idx]
        #     x, y, z = ndc[:,0], ndc[:,1], ndc[:,2]

        #     ix = ((x+1.0)*0.5*W).long()
        #     iy = ((1.0-y)*0.5*H).long()
        #     valid = (ix>=0)&(ix<W)&(iy>=0)&(iy<H)&(z>0)
        #     if not valid.any():
        #         continue

        #     vidx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        #     ixv, iyv = ix[vidx], iy[vidx]

        #     view_dir = os.path.join(mask_root, stem)
        #     if not os.path.isdir(view_dir):
        #         continue

        #     obj_files = [f for f in os.listdir(view_dir) if f.startswith("obj_")]
        #     if not obj_files:
        #         continue

        #     masks = []
        #     cat_ids = []

        #     for f in obj_files:
        #         try:
        #             obj_id = int(f.split("_")[1].split(".")[0])
        #         except:
        #             continue
        #         obj_info = semantic_labels.get(str(obj_id))
        #         if obj_info is None:
        #             continue
        #         cat_id = category_to_id.get(obj_info["category"])
        #         if cat_id is None:
        #             continue

        #         p = os.path.join(view_dir, f)
        #         try:
        #             arr = torch.from_numpy(np.array(Image.open(p).convert("L"), dtype=np.uint8)).float()/255
        #             if arr.shape[0]!=H or arr.shape[1]!=W:
        #                 arr = F.interpolate(arr[None,None], size=(H,W), mode="nearest")[0,0]
        #         except:
        #             continue

        #         masks.append(arr)
        #         cat_ids.append(cat_id)

        #     if not masks:
        #         continue

        #     masks = torch.stack(masks, dim=0).to(device)   # (K, H, W)
        #     cat_tensor = torch.tensor(cat_ids, device=device)

        #     all_masks.append(masks)
        #     all_cat_ids.append(cat_tensor)
        #     all_ixv.append(ixv)
        #     all_iyv.append(iyv)
        #     all_vidx.append(vidx)

        # # ===== 2. 批量GPU处理 =====
        # for masks, cat_tensor, ixv, iyv, vidx in tqdm(zip(all_masks, all_cat_ids, all_ixv, all_iyv, all_vidx), desc="Voting"):
        #     # 取mask上对应像素
        #     sampled = masks[:, iyv, ixv] > 0.5   # (K, M)
        #     hit_any = sampled.any(dim=0)
        #     if not hit_any.any():
        #         continue

        #     first_hit = sampled.float().argmax(dim=0)
        #     valid_pts = vidx[hit_any]
        #     chosen = first_hit[hit_any]
        #     assign_ids = cat_tensor[chosen]

        #     # 只赋值未标记
        #     unassigned = semantic_tensor[valid_pts] == -1
        #     semantic_tensor[valid_pts[unassigned]] = assign_ids[unassigned]

        # ===== 写回 =====
        self.gaussians._semantic = semantic_tensor

        # ===== 统计 =====
        assigned = (semantic_tensor != -1).sum().item()
        print(f"[RESULT] assigned {assigned}/{N} ({assigned/N*100:.2f}%)")

        for cat, cid in category_to_id.items():
            count = (semantic_tensor == cid).sum().item()
            if count > 0:
                print(f"  {cat}: {count}")