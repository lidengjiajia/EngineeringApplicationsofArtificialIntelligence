"""
FedLGO损失景观实验脚本

运行损失景观分析，比较FedLGO与基线方法的收敛性质

作者: FedLGO Team
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loss_landscape import (
    LossLandscapeAnalyzer, 
    LossLandscapeVisualizer,
    run_loss_landscape_analysis
)
from utils.data_utils import read_client_data


def load_model_from_h5(model_path, model_class, device='cuda'):
    """从h5文件加载模型"""
    import h5py
    
    model = model_class().to(device)
    
    with h5py.File(model_path, 'r') as f:
        state_dict = {}
        for key in f.keys():
            state_dict[key] = torch.tensor(np.array(f[key]))
        model.load_state_dict(state_dict)
    
    return model


def compare_methods_landscape(dataset: str = 'Uci',
                              heterogeneity: str = 'label',
                              methods: list = None,
                              device: str = 'cuda',
                              save_dir: str = None):
    """
    对比多种方法的损失景观
    
    Args:
        dataset: 数据集名称
        heterogeneity: 异质性类型
        methods: 要对比的方法列表
        device: 计算设备
        save_dir: 保存目录
    """
    if methods is None:
        methods = ['FedAvg', 'FedProx', 'FedLGO']
    
    if save_dir is None:
        save_dir = f'./results/loss_landscape_{dataset}_{heterogeneity}'
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print(f"损失景观对比分析: {dataset} - {heterogeneity}")
    print("="*70)
    print(f"对比方法: {methods}")
    print(f"保存目录: {save_dir}")
    print()
    
    # 加载测试数据
    from flcore.trainmodel.models import MLP_Credit
    
    # 模拟测试数据（实际使用时替换为真实数据）
    print("加载测试数据...")
    # test_data = read_client_data(dataset, 0, is_train=False)
    # 这里创建模拟数据用于演示
    X_test = torch.randn(1000, 23)
    y_test = torch.randint(0, 2, (1000,))
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    
    # 分析各方法
    results = {}
    visualizer = LossLandscapeVisualizer(save_dir)
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"分析方法: {method}")
        print(f"{'='*50}")
        
        # 查找模型文件
        result_dir = f'./results/{dataset}_{method}_{heterogeneity}'
        model_files = []
        if os.path.exists(result_dir):
            model_files = [f for f in os.listdir(result_dir) if f.endswith('.h5')]
        
        if not model_files:
            print(f"  [警告] 未找到{method}的模型文件，使用随机初始化模型")
            model = MLP_Credit(input_dim=23, hidden_dim=64, output_dim=2).to(device)
        else:
            model_path = os.path.join(result_dir, model_files[0])
            print(f"  加载模型: {model_path}")
            model = load_model_from_h5(model_path, 
                                       lambda: MLP_Credit(input_dim=23, hidden_dim=64, output_dim=2),
                                       device)
        
        # 创建分析器
        analyzer = LossLandscapeAnalyzer(model, device)
        
        # 2D损失景观
        print("  计算2D损失景观...")
        X, Y, Z = analyzer.random_direction_2d(
            test_loader,
            x_range=(-0.5, 0.5),
            y_range=(-0.5, 0.5),
            steps=15
        )
        
        results[method] = {
            'X': X, 'Y': Y, 'Z': Z,
            'min_loss': Z.min(),
            'max_loss': Z.max()
        }
        
        # 单独保存每个方法的景观
        np.savez(os.path.join(save_dir, f'{method}_landscape.npz'), X=X, Y=Y, Z=Z)
        
        visualizer.plot_2d_contour(
            X, Y, Z,
            title=f'{method} Loss Landscape ({dataset}-{heterogeneity})',
            save_name=f'{method}_contour.png'
        )
        
        visualizer.plot_3d_surface(
            X, Y, Z,
            title=f'{method} Loss Landscape 3D ({dataset}-{heterogeneity})',
            save_name=f'{method}_3d.png'
        )
        
        # 锐度分析
        print("  分析锐度...")
        sharpness = analyzer.analyze_sharpness(test_loader, epsilon=0.01, num_samples=10)
        results[method]['sharpness'] = sharpness
        print(f"  最大锐度: {sharpness['max_sharpness']:.6f}")
        print(f"  平均锐度: {sharpness['avg_sharpness']:.6f}")
    
    # 绘制对比图
    print("\n生成对比图...")
    
    landscape_data = {m: {'X': results[m]['X'], 'Y': results[m]['Y'], 'Z': results[m]['Z']} 
                      for m in methods}
    visualizer.plot_comparison(
        landscape_data,
        title=f'Loss Landscape Comparison ({dataset}-{heterogeneity})',
        save_name='comparison_contour.png'
    )
    
    sharpness_data = {m: results[m]['sharpness'] for m in methods}
    visualizer.plot_sharpness_comparison(
        sharpness_data,
        title=f'Sharpness Comparison ({dataset}-{heterogeneity})',
        save_name='comparison_sharpness.png'
    )
    
    # 保存汇总结果
    summary = {
        'dataset': dataset,
        'heterogeneity': heterogeneity,
        'methods': methods,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {
            m: {
                'min_loss': float(results[m]['min_loss']),
                'max_loss': float(results[m]['max_loss']),
                'sharpness': results[m]['sharpness']
            }
            for m in methods
        }
    }
    
    import json
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("损失景观分析完成!")
    print(f"结果保存至: {save_dir}")
    print("="*70)
    
    # 打印汇总
    print("\n汇总结果:")
    print("-"*50)
    for method in methods:
        r = results[method]
        print(f"{method}:")
        print(f"  最小损失: {r['min_loss']:.4f}")
        print(f"  最大损失: {r['max_loss']:.4f}")
        print(f"  最大锐度: {r['sharpness']['max_sharpness']:.6f}")
        print(f"  平均锐度: {r['sharpness']['avg_sharpness']:.6f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='FedLGO损失景观分析')
    parser.add_argument('--dataset', type=str, default='Uci', choices=['Uci', 'Xinwang'])
    parser.add_argument('--heterogeneity', type=str, default='label', 
                        choices=['iid', 'feature', 'label', 'quantity'])
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['FedAvg', 'FedProx', 'FedLGO'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[警告] CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    compare_methods_landscape(
        dataset=args.dataset,
        heterogeneity=args.heterogeneity,
        methods=args.methods,
        device=args.device,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
