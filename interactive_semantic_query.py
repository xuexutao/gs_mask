#!/usr/bin/env python3
"""
交互式语义场景查询工具
提供命令行界面进行语义场景查询
"""

import os
import sys
import json
import readline  # 用于命令行历史记录
from typing import List, Optional
import numpy as np
from query_semantic_scene import SemanticSceneQuery


class InteractiveSemanticQuery:
    """交互式语义查询类"""
    
    def __init__(self, ply_path: str, category_mapping: Optional[str] = None):
        """
        初始化交互式查询
        
        参数:
            ply_path: PLY文件路径
            category_mapping: 类别映射文件路径
        """
        self.query = SemanticSceneQuery(ply_path)
        
        if category_mapping:
            self.query.load_category_mapping(category_mapping)
        
        self.commands = {
            "help": self.show_help,
            "stats": self.show_stats,
            "categories": self.list_categories,
            "query": self.query_category,
            "bbox": self.query_bbox,
            "sphere": self.query_sphere,
            "save": self.save_filtered,
            "visualize": self.visualize,
            "export": self.export_stats,
            "clear": self.clear_screen,
            "exit": self.exit_program,
            "quit": self.exit_program,
        }
        
        self.current_mask = None
    
    def show_help(self, args: List[str]):
        """显示帮助信息"""
        print("\n可用命令:")
        print("  help                    显示此帮助信息")
        print("  stats                   显示场景统计信息")
        print("  categories              列出所有类别")
        print("  query <category>        查询指定类别的点")
        print("  bbox <xmin> <ymin> <zmin> <xmax> <ymax> <zmax>  查询边界框内的点")
        print("  sphere <x> <y> <z> <radius>  查询球体内的点")
        print("  save <filename>         保存当前查询结果到PLY文件")
        print("  visualize               可视化当前查询结果")
        print("  export <filename>       导出统计信息到JSON文件")
        print("  clear                   清屏")
        print("  exit/quit               退出程序")
        print("\n示例:")
        print("  query chair             查询所有椅子")
        print("  bbox -1 -1 -1 1 1 1     查询边界框内的点")
        print("  sphere 0 0 0 2          查询半径为2的球体内的点")
        print("  save filtered.ply       保存当前查询结果")
    
    def show_stats(self, args: List[str]):
        """显示统计信息"""
        self.query.get_statistics()
    
    def list_categories(self, args: List[str]):
        """列出所有类别"""
        if not self.query.id_to_category:
            print("未加载类别映射。请使用 --category_mapping 参数加载映射文件。")
            return
        
        print("\n可用类别:")
        for category_id, category_name in self.query.id_to_category.items():
            count = np.sum(self.query.semantic_labels == category_id)
            print(f"  {category_name} (ID {category_id}): {count} 个点")
    
    def query_category(self, args: List[str]):
        """查询类别"""
        if len(args) < 1:
            print("用法: query <category_name>")
            return
        
        category_name = args[0]
        self.current_mask = self.query.query_by_category(category_name)
    
    def query_bbox(self, args: List[str]):
        """查询边界框"""
        if len(args) < 6:
            print("用法: bbox <xmin> <ymin> <zmin> <xmax> <ymax> <zmax>")
            return
        
        try:
            bbox_min = [float(args[0]), float(args[1]), float(args[2])]
            bbox_max = [float(args[3]), float(args[4]), float(args[5])]
            self.current_mask = self.query.query_by_bounding_box(bbox_min, bbox_max)
        except ValueError:
            print("错误: 坐标必须是数字")
    
    def query_sphere(self, args: List[str]):
        """查询球体"""
        if len(args) < 4:
            print("用法: sphere <x> <y> <z> <radius>")
            return
        
        try:
            center = [float(args[0]), float(args[1]), float(args[2])]
            radius = float(args[3])
            self.current_mask = self.query.query_by_sphere(center, radius)
        except ValueError:
            print("错误: 坐标和半径必须是数字")
    
    def save_filtered(self, args: List[str]):
        """保存过滤后的点云"""
        if len(args) < 1:
            print("用法: save <filename>")
            return
        
        if self.current_mask is None:
            print("错误: 没有当前查询结果。请先执行查询。")
            return
        
        filename = args[0]
        self.query.save_filtered_ply(filename, self.current_mask)
    
    def visualize(self, args: List[str]):
        """可视化"""
        self.query.visualize(self.current_mask)
    
    def export_stats(self, args: List[str]):
        """导出统计信息"""
        if len(args) < 1:
            print("用法: export <filename>")
            return
        
        filename = args[0]
        self.query.export_statistics(filename)
    
    def clear_screen(self, args: List[str]):
        """清屏"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def exit_program(self, args: List[str]):
        """退出程序"""
        print("再见！")
        sys.exit(0)
    
    def run(self):
        """运行交互式查询"""
        print("=" * 60)
        print("交互式语义场景查询工具")
        print("=" * 60)
        print(f"加载的点云: {self.query.ply_path}")
        print(f"总点数: {len(self.query.points)}")
        print("输入 'help' 查看可用命令")
        print("输入 'exit' 或 'quit' 退出程序")
        print("=" * 60)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n查询> ").strip()
                if not user_input:
                    continue
                
                # 解析命令
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:]
                
                # 执行命令
                if command in self.commands:
                    self.commands[command](args)
                else:
                    print(f"未知命令: {command}")
                    print("输入 'help' 查看可用命令")
            
            except KeyboardInterrupt:
                print("\n使用 'exit' 或 'quit' 退出程序")
            except Exception as e:
                print(f"错误: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="交互式语义场景查询工具")
    parser.add_argument("--ply_path", required=True, help="输入PLY文件路径")
    parser.add_argument("--category_mapping", help="类别映射JSON文件路径")
    
    args = parser.parse_args()
    
    
    # 运行交互式查询
    app = InteractiveSemanticQuery(args.ply_path, args.category_mapping)
    app.run()


if __name__ == "__main__":
    main()