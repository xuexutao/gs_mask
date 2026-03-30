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
import sys
# Set LD_LIBRARY_PATH to include torch libs
torch_lib = os.path.join(sys.prefix, 'lib', 'python{}.{}'.format(sys.version_info.major, sys.version_info.minor), 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib):
    os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, semantic_labels_path=None, mask_dir=None, refine_args=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # 加载语义标签文件（如果提供）
    semantic_labels = None
    if semantic_labels_path and os.path.exists(semantic_labels_path):
        print(f"Loading semantic labels from {semantic_labels_path}")
        with open(semantic_labels_path, 'r') as f:
            semantic_labels = json.load(f)
        print(f"Loaded {len(semantic_labels)} semantic labels")
    elif semantic_labels_path:
        print(f"Warning: Semantic labels file not found: {semantic_labels_path}")
    
    # 传递 mask_dir 到 mask_opt 以便语义标签分配
    if mask_dir:
        opt.mask_dir = mask_dir
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, mask_opt=opt)
    gaussians.training_setup(opt)
    
    # 分配语义标签（如果提供了）
    if semantic_labels is not None:
        print(f"[DEBUG train.py] Calling assign_semantic_labels with {len(semantic_labels)} labels")
        scene.assign_semantic_labels_multi(semantic_labels)
        
        # 语义标签细化（如果启用）
        if refine_args is not None and refine_args.get('enabled', False):
            print("Refining semantic labels...")
            # 获取高斯位置和语义标签
            xyz = gaussians.get_xyz.detach().cpu().numpy()
            semantic = gaussians._semantic.cpu().numpy()
            
            # 获取尺度和不透明度（如果可用）
            scale_mag = None
            opacity = None
            if hasattr(gaussians, 'get_scaling'):
                scales = gaussians.get_scaling.detach().cpu().numpy()
                scale_mag = np.linalg.norm(scales, axis=1)
            if hasattr(gaussians, 'get_opacity'):
                opacity = gaussians.get_opacity.detach().cpu().numpy().flatten()
            
            # 1. 基于空间聚类移除离群点
            if refine_args.get('use_clustering', True):
                eps = refine_args.get('eps', 0.1)
                min_samples = refine_args.get('min_samples', 10)
                min_cluster_size = refine_args.get('min_cluster_size', 50)
                
                unique_labels = np.unique(semantic)
                new_semantic = semantic.copy()
                
                for cat in unique_labels:
                    if cat == -1:
                        continue
                    mask = semantic == cat
                    if np.sum(mask) < min_cluster_size:
                        # 类别点数太少，直接标记为未分类
                        new_semantic[mask] = -1
                        continue
                    
                    cat_points = xyz[mask]
                    # 使用DBSCAN聚类（需要sklearn）
                    try:
                        from sklearn.cluster import DBSCAN
                        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(cat_points)
                        cluster_labels = clustering.labels_
                        # 统计每个聚类的大小
                        unique_clusters, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
                        if len(unique_clusters) == 0:
                            # 所有点都是噪声，标记为未分类
                            new_semantic[mask] = -1
                            continue
                        
                        # 保留点数大于阈值的聚类
                        large_clusters = unique_clusters[counts >= min_cluster_size]
                        if len(large_clusters) == 0:
                            # 没有大聚类，保留最大的聚类
                            largest_cluster = unique_clusters[np.argmax(counts)]
                            large_clusters = [largest_cluster]
                        
                        # 创建新的掩码：只保留大聚类中的点
                        keep_mask = np.isin(cluster_labels, large_clusters)
                        # 将不属于大聚类的点标记为-1
                        cat_indices = np.where(mask)[0]
                        for idx, keep in zip(cat_indices, keep_mask):
                            if not keep:
                                new_semantic[idx] = -1
                        
                    except ImportError:
                        print("Warning: sklearn not installed, skipping clustering")
                
                semantic = new_semantic
            
            # 2. 基于尺度过滤
            scale_threshold = refine_args.get('scale_threshold', 2.0)
            if scale_mag is not None:
                large_scale = scale_mag > scale_threshold
                semantic[large_scale] = -1
                print(f"Filtered {np.sum(large_scale)} points with scale > {scale_threshold}")
            
            # 3. 基于不透明度过滤
            opacity_threshold = refine_args.get('opacity_threshold', 0.5)
            if opacity is not None:
                low_opacity = opacity < opacity_threshold
                # 注意：不透明度较低的点可能是前景的透明部分，不一定属于背景
                # 这里我们只过滤非常不透明的点（例如 >0.5），但用户可能希望调整
                # 暂时注释掉，因为可能过于激进
                # semantic[low_opacity] = -1
                # print(f"Filtered {np.sum(low_opacity)} points with opacity < {opacity_threshold}")
            
            # 将更新后的语义标签写回高斯模型
            gaussians._semantic = torch.tensor(semantic, dtype=torch.long, device='cuda')
            
            # 统计细化后的标签分布
            assigned = (semantic != -1).sum()
            print(f"After refinement: {assigned}/{len(semantic)} points labeled ({assigned/len(semantic)*100:.1f}%)")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--semantic_labels", type=str, default=None,
                        help="Path to labels.json file containing semantic label mapping")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Path to mask directory (e.g., data/gs_data/room/images_4/masks_sam)")
    parser.add_argument("--refine_semantic", action="store_true", default=False,
                        help="Enable semantic label refinement after assignment")
    parser.add_argument("--refine_eps", type=float, default=0.1,
                        help="DBSCAN epsilon for clustering")
    parser.add_argument("--refine_min_samples", type=int, default=10,
                        help="DBSCAN min_samples for clustering")
    parser.add_argument("--refine_min_cluster_size", type=int, default=50,
                        help="Minimum cluster size to keep")
    parser.add_argument("--refine_scale_threshold", type=float, default=2.0,
                        help="Scale threshold for background filtering")
    parser.add_argument("--refine_opacity_threshold", type=float, default=0.5,
                        help="Opacity threshold for background filtering")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 构建语义标签细化参数
    refine_args = None
    if args.refine_semantic:
        refine_args = {
            'enabled': True,
            'eps': args.refine_eps,
            'min_samples': args.refine_min_samples,
            'min_cluster_size': args.refine_min_cluster_size,
            'scale_threshold': args.refine_scale_threshold,
            'opacity_threshold': args.refine_opacity_threshold,
            'use_clustering': True
        }
        print(f"Semantic refinement enabled with parameters: {refine_args}")
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.semantic_labels, args.mask_dir, refine_args)

    # All done
    print("\nTraining complete.")
