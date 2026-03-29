#!/usr/bin/env python3
"""
语义场景查询工具
加载带有语义标签的3D高斯点云，支持查询、过滤和可视化
"""

import os
import sys
# Set LD_LIBRARY_PATH to include torch libs
torch_lib = os.path.join(sys.prefix, 'lib', 'python{}.{}'.format(sys.version_info.major, sys.version_info.minor), 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib):
    os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import torch
from plyfile import PlyData, PlyElement

class SemanticSceneQuery:
    """语义场景查询类"""
    
    def __init__(self, ply_path: str):
        """
        初始化语义场景查询
        
        参数:
            ply_path: PLY文件路径
        """
        self.ply_path = ply_path
        self.points = None  # (N, 3) 点云坐标
        self.semantic_labels = None  # (N,) 语义标签ID
        self.opacities = None  # (N,) 不透明度
        self.scales = None  # (N, 3) 缩放
        self.rotations = None  # (N, 4) 旋转四元数
        self.features_dc = None  # (N, 3) DC特征
        self.features_rest = None  # (N, 45) 其余特征
        
        # 类别映射
        self.category_to_id = {}  # 类别名称 -> 整数ID
        self.id_to_category = {}  # 整数ID -> 类别名称
        
        # 加载点云
        self.load_ply()
    
    def load_ply(self):
        """加载PLY文件"""
        print(f"Loading PLY file: {self.ply_path}")
        
        if not os.path.exists(self.ply_path):
            raise FileNotFoundError(f"PLY file not found: {self.ply_path}")
        
        plydata = PlyData.read(self.ply_path)
        
        # 读取坐标
        x = np.asarray(plydata.elements[0]["x"])
        y = np.asarray(plydata.elements[0]["y"])
        z = np.asarray(plydata.elements[0]["z"])
        self.points = np.stack([x, y, z], axis=1)
        
        # 读取语义标签
        if "semantic" in plydata.elements[0]:
            self.semantic_labels = np.asarray(plydata.elements[0]["semantic"]).astype(np.int32)
        else:
            print("Warning: No semantic labels found in PLY file")
            self.semantic_labels = np.full(len(x), -1, dtype=np.int32)
        
        # 读取其他属性
        self.opacities = np.asarray(plydata.elements[0]["opacity"])
        
        # 读取缩放
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        self.scales = np.zeros((len(x), len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            self.scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # 读取旋转
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        self.rotations = np.zeros((len(x), len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            self.rotations[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # 读取DC特征
        f_dc_0 = np.asarray(plydata.elements[0]["f_dc_0"])
        f_dc_1 = np.asarray(plydata.elements[0]["f_dc_1"])
        f_dc_2 = np.asarray(plydata.elements[0]["f_dc_2"])
        self.features_dc = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)
        
        # 读取其余特征
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        self.features_rest = np.zeros((len(x), len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            self.features_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        print(f"Loaded {len(self.points)} points")
        print(f"Semantic labels: {np.unique(self.semantic_labels)}")
    
    def load_category_mapping(self, mapping_path: str):
        """加载类别映射文件
        
        参数:
            mapping_path: JSON文件路径，包含类别映射
        """
        if not os.path.exists(mapping_path):
            print(f"Warning: Category mapping file not found: {mapping_path}")
            return
        
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
        if isinstance(mapping, dict):
            # 假设格式为 {"category_name": id, ...}
            self.category_to_id = mapping
            self.id_to_category = {v: k for k, v in mapping.items()}
        else:
            # 假设格式为 [{"name": "chair", "id": 0}, ...]
            for item in mapping:
                if isinstance(item, dict) and "name" in item and "id" in item:
                    self.category_to_id[item["name"]] = item["id"]
                    self.id_to_category[item["id"]] = item["name"]
        
        print(f"Loaded category mapping: {self.category_to_id}")
    
    def get_category_name(self, label_id: int) -> str:
        """根据标签ID获取类别名称"""
        if label_id == -1:
            return "unlabeled"
        return self.id_to_category.get(label_id, f"unknown_{label_id}")
    
    def get_category_id(self, category_name: str) -> Optional[int]:
        """根据类别名称获取标签ID"""
        return self.category_to_id.get(category_name)
    
    def get_statistics(self):
        """获取场景统计信息"""
        total_points = len(self.points)
        labeled_points = np.sum(self.semantic_labels != -1)
        
        print(f"Total points: {total_points}")
        print(f"Labeled points: {labeled_points} ({labeled_points/total_points*100:.1f}%)")
        print(f"Unlabeled points: {total_points - labeled_points} ({(total_points - labeled_points)/total_points*100:.1f}%)")
        
        # 统计每个类别的点数量
        unique_labels = np.unique(self.semantic_labels)
        for label in unique_labels:
            if label == -1:
                continue
            count = np.sum(self.semantic_labels == label)
            category_name = self.get_category_name(label)
            print(f"  {category_name} (ID {label}): {count} points ({count/total_points*100:.1f}%)")
    
    def query_by_category(self, category_name: str) -> np.ndarray:
        """根据类别名称查询点云
        
        参数:
            category_name: 类别名称
            
        返回:
            布尔掩码数组，表示属于该类别的点
        """
        category_id = self.get_category_id(category_name)
        if category_id is None:
            print(f"Warning: Unknown category '{category_name}'")
            return np.zeros(len(self.points), dtype=bool)
        
        mask = self.semantic_labels == category_id
        count = np.sum(mask)
        print(f"Found {count} points of category '{category_name}' (ID {category_id})")
        return mask
    
    def query_by_bounding_box(self, bbox_min: List[float], bbox_max: List[float]) -> np.ndarray:
        """根据边界框查询点云
        
        参数:
            bbox_min: 边界框最小值 [x_min, y_min, z_min]
            bbox_max: 边界框最大值 [x_max, y_max, z_max]
            
        返回:
            布尔掩码数组，表示在边界框内的点
        """
        bbox_min = np.array(bbox_min)
        bbox_max = np.array(bbox_max)
        
        mask = np.all((self.points >= bbox_min) & (self.points <= bbox_max), axis=1)
        count = np.sum(mask)
        print(f"Found {count} points within bounding box")
        return mask
    
    def query_by_sphere(self, center: List[float], radius: float) -> np.ndarray:
        """根据球体查询点云
        
        参数:
            center: 球心坐标 [x, y, z]
            radius: 球体半径
            
        返回:
            布尔掩码数组，表示在球体内的点
        """
        center = np.array(center)
        distances = np.linalg.norm(self.points - center, axis=1)
        mask = distances <= radius
        count = np.sum(mask)
        print(f"Found {count} points within sphere (center={center}, radius={radius})")
        return mask
    
    def query_combined(self, category_name: Optional[str] = None, 
                      bbox_min: Optional[List[float]] = None,
                      bbox_max: Optional[List[float]] = None,
                      center: Optional[List[float]] = None,
                      radius: Optional[float] = None) -> np.ndarray:
        """组合查询
        
        参数:
            category_name: 类别名称
            bbox_min: 边界框最小值
            bbox_max: 边界框最大值
            center: 球心坐标
            radius: 球体半径
            
        返回:
            布尔掩码数组
        """
        mask = np.ones(len(self.points), dtype=bool)
        
        if category_name is not None:
            category_mask = self.query_by_category(category_name)
            mask = mask & category_mask
        
        if bbox_min is not None and bbox_max is not None:
            bbox_mask = self.query_by_bounding_box(bbox_min, bbox_max)
            mask = mask & bbox_mask
        
        if center is not None and radius is not None:
            sphere_mask = self.query_by_sphere(center, radius)
            mask = mask & sphere_mask
        
        count = np.sum(mask)
        print(f"Combined query found {count} points")
        return mask
    
    def save_filtered_ply(self, output_path: str, mask: np.ndarray):
        """保存过滤后的点云到PLY文件
        
        参数:
            output_path: 输出PLY文件路径
            mask: 布尔掩码数组
        """
        if not np.any(mask):
            print("Warning: No points to save")
            return
        
        # 读取原始PLY数据
        plydata = PlyData.read(self.ply_path)
        original_elements = plydata.elements[0]
        
        # 创建新的元素数组
        new_data = []
        for i in range(len(original_elements.data)):
            if mask[i]:
                new_data.append(original_elements.data[i])
        
        # 创建新的PlyElement
        new_element = PlyElement.describe(
            np.array(new_data, dtype=original_elements.data.dtype),
            name='vertex'
        )
        
        # 保存到文件
        new_plydata = PlyData([new_element])
        new_plydata.write(output_path)
        print(f"Saved {len(new_data)} points to {output_path}")
    
    def visualize(self, mask: Optional[np.ndarray] = None, 
                 category_colors: Optional[Dict[str, List[float]]] = None):
        """可视化点云（需要matplotlib）
        
        参数:
            mask: 要可视化的点掩码
            category_colors: 类别颜色映射
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Warning: matplotlib not installed. Skipping visualization.")
            return
        
        if mask is None:
            mask = np.ones(len(self.points), dtype=bool)
        
        points_to_plot = self.points[mask]
        labels_to_plot = self.semantic_labels[mask]
        
        # 创建颜色映射
        unique_labels = np.unique(labels_to_plot)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制点云
        for label in unique_labels:
            if label == -1:
                color = [0.5, 0.5, 0.5, 0.5]  # 灰色表示未标记
            else:
                color = label_to_color[label]
            
            label_mask = labels_to_plot == label
            if np.any(label_mask):
                points = points_to_plot[label_mask]
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=[color], s=1, label=self.get_category_name(label))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Semantic 3D Gaussian Splatting')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def export_statistics(self, output_path: str):
        """导出统计信息到JSON文件
        
        参数:
            output_path: 输出JSON文件路径
        """
        stats = {
            "total_points": int(len(self.points)),
            "labeled_points": int(np.sum(self.semantic_labels != -1)),
            "categories": {}
        }
        
        unique_labels = np.unique(self.semantic_labels)
        for label in unique_labels:
            if label == -1:
                continue
            count = int(np.sum(self.semantic_labels == label))
            category_name = self.get_category_name(label)
            stats["categories"][category_name] = {
                "id": int(label),
                "count": count,
                "percentage": float(count / len(self.points) * 100)
            }
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="语义场景查询工具")
    parser.add_argument("--ply_path", required=True, help="输入PLY文件路径")
    parser.add_argument("--category_mapping", help="类别映射JSON文件路径")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--query_category", help="查询指定类别的点")
    parser.add_argument("--query_bbox", nargs=6, type=float, 
                       help="查询边界框内的点: x_min y_min z_min x_max y_max z_max")
    parser.add_argument("--query_sphere", nargs=4, type=float,
                       help="查询球体内的点: x y z radius")
    parser.add_argument("--save_filtered", help="保存过滤后的点云到PLY文件")
    parser.add_argument("--visualize", action="store_true", help="可视化点云")
    parser.add_argument("--export_stats", help="导出统计信息到JSON文件")
    
    args = parser.parse_args()
    
    # 创建查询对象
    query = SemanticSceneQuery(args.ply_path)
    
    # 加载类别映射
    if args.category_mapping:
        query.load_category_mapping(args.category_mapping)
    
    # 显示统计信息
    if args.stats:
        query.get_statistics()
    
    # 执行查询
    mask = None
    if args.query_category or args.query_bbox or args.query_sphere:
        bbox_min = None
        bbox_max = None
        center = None
        radius = None
        
        if args.query_bbox:
            bbox_min = args.query_bbox[:3]
            bbox_max = args.query_bbox[3:]
        
        if args.query_sphere:
            center = args.query_sphere[:3]
            radius = args.query_sphere[3]
        
        mask = query.query_combined(
            category_name=args.query_category,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            center=center,
            radius=radius
        )
    
    # 保存过滤后的点云
    if args.save_filtered and mask is not None:
        query.save_filtered_ply(args.save_filtered, mask)
    
    # 可视化
    if args.visualize:
        query.visualize(mask)
    
    # 导出统计信息
    if args.export_stats:
        query.export_statistics(args.export_stats)
    
    # 如果没有指定任何操作，显示帮助信息
    if not any([args.stats, args.query_category, args.query_bbox, 
                args.query_sphere, args.save_filtered, args.visualize, 
                args.export_stats]):
        print("No operation specified. Use --help for usage information.")


if __name__ == "__main__":
    main()