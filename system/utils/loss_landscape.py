"""
损失景观可视化 (Loss Landscape Visualization)

用于可视化联邦学习模型的损失景观，分析：
1. 全局模型的收敛性质
2. 不同异质性场景下的景观平滑度
3. FedLGO与基线方法的景观对比

基于论文:
- Li, H., et al. (2018). Visualizing the Loss Landscape of Neural Nets. NeurIPS.
- Hao, Y., et al. (2019). Visualizing and Understanding the Effectiveness of BERT. EMNLP.

作者: FedLGO Team
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import h5py
from typing import Optional, Tuple, List, Dict
import json
from datetime import datetime


class LossLandscapeAnalyzer:
    """
    损失景观分析器
    
    支持:
    1. 1D线性插值可视化 (两个模型之间)
    2. 2D等高线和3D曲面可视化
    3. 曲率和平滑度分析
    4. Hessian特征值分析
    """
    
    def __init__(self, model, device='cuda'):
        """
        初始化损失景观分析器
        
        Args:
            model: PyTorch模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
    def get_model_params(self, model=None) -> List[torch.Tensor]:
        """获取模型参数作为向量列表"""
        if model is None:
            model = self.model
        return [p.data.clone() for p in model.parameters()]
    
    def set_model_params(self, params: List[torch.Tensor], model=None):
        """设置模型参数"""
        if model is None:
            model = self.model
        for p, new_p in zip(model.parameters(), params):
            p.data.copy_(new_p)
    
    def get_random_direction(self, normalize=True) -> List[torch.Tensor]:
        """
        生成随机方向（用于可视化）
        
        Args:
            normalize: 是否归一化（filter-wise normalization）
        
        Returns:
            随机方向向量
        """
        direction = []
        for p in self.model.parameters():
            d = torch.randn_like(p)
            if normalize and len(p.shape) > 1:
                # Filter-wise normalization: 与参数范数对齐
                d = d / (d.norm() + 1e-10) * p.norm()
            direction.append(d)
        return direction
    
    def compute_loss(self, data_loader, criterion=None) -> float:
        """
        计算给定数据上的损失
        
        Args:
            data_loader: 数据加载器
            criterion: 损失函数（默认CrossEntropyLoss）
        
        Returns:
            平均损失值
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * len(target)
                total_samples += len(target)
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def linear_interpolation_1d(self, params1: List[torch.Tensor], 
                                 params2: List[torch.Tensor],
                                 data_loader, 
                                 steps: int = 51,
                                 criterion=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        1D线性插值：在两个模型参数之间进行插值
        
        $\theta(\alpha) = (1-\alpha)\theta_1 + \alpha\theta_2$
        
        Args:
            params1: 第一个模型参数
            params2: 第二个模型参数
            data_loader: 数据加载器
            steps: 插值步数
            criterion: 损失函数
        
        Returns:
            alphas: 插值系数数组
            losses: 对应的损失值数组
        """
        alphas = np.linspace(0, 1, steps)
        losses = []
        
        for alpha in alphas:
            # 计算插值参数
            interpolated = []
            for p1, p2 in zip(params1, params2):
                interp_p = (1 - alpha) * p1 + alpha * p2
                interpolated.append(interp_p)
            
            # 设置模型参数并计算损失
            self.set_model_params(interpolated)
            loss = self.compute_loss(data_loader, criterion)
            losses.append(loss)
            print(f"  α={alpha:.2f}, Loss={loss:.4f}")
        
        return alphas, np.array(losses)
    
    def random_direction_2d(self, data_loader,
                            x_range: Tuple[float, float] = (-1.0, 1.0),
                            y_range: Tuple[float, float] = (-1.0, 1.0),
                            steps: int = 21,
                            criterion=None,
                            direction1=None,
                            direction2=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D损失景观：沿两个随机方向可视化
        
        $\theta(\alpha, \beta) = \theta^* + \alpha d_1 + \beta d_2$
        
        Args:
            data_loader: 数据加载器
            x_range: x轴范围
            y_range: y轴范围
            steps: 网格点数
            criterion: 损失函数
            direction1, direction2: 自定义方向（可选）
        
        Returns:
            X, Y: 网格坐标
            Z: 损失值矩阵
        """
        # 保存原始参数
        original_params = self.get_model_params()
        
        # 生成或使用指定的随机方向
        if direction1 is None:
            direction1 = self.get_random_direction(normalize=True)
        if direction2 is None:
            direction2 = self.get_random_direction(normalize=True)
        
        # 生成网格
        x_coords = np.linspace(x_range[0], x_range[1], steps)
        y_coords = np.linspace(y_range[0], y_range[1], steps)
        X, Y = np.meshgrid(x_coords, y_coords)
        Z = np.zeros_like(X)
        
        total_points = steps * steps
        for i, alpha in enumerate(x_coords):
            for j, beta in enumerate(y_coords):
                # 计算扰动后的参数
                perturbed = []
                for p, d1, d2 in zip(original_params, direction1, direction2):
                    new_p = p + alpha * d1 + beta * d2
                    perturbed.append(new_p)
                
                self.set_model_params(perturbed)
                loss = self.compute_loss(data_loader, criterion)
                Z[j, i] = loss
                
                progress = (i * steps + j + 1) / total_points * 100
                print(f"\r  进度: {progress:.1f}% ({i * steps + j + 1}/{total_points})", end='')
        
        print()
        
        # 恢复原始参数
        self.set_model_params(original_params)
        
        return X, Y, Z
    
    def compute_gradient_norm(self, data_loader, criterion=None) -> float:
        """
        计算梯度范数
        
        Args:
            data_loader: 数据加载器
            criterion: 损失函数
        
        Returns:
            梯度范数
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        self.model.zero_grad()
        
        total_loss = 0.0
        total_samples = 0
        
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            total_samples += len(target)
        
        # 计算梯度范数
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = np.sqrt(grad_norm)
        
        return grad_norm
    
    def estimate_hessian_eigenvalues(self, data_loader, num_eigenvalues: int = 5,
                                      criterion=None, iterations: int = 100) -> np.ndarray:
        """
        使用幂迭代法估计Hessian矩阵的最大特征值
        
        用于分析损失景观的曲率
        
        Args:
            data_loader: 数据加载器
            num_eigenvalues: 要估计的特征值数量
            criterion: 损失函数
            iterations: 迭代次数
        
        Returns:
            估计的特征值数组
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        eigenvalues = []
        
        for k in range(num_eigenvalues):
            # 初始化随机向量
            v = self.get_random_direction(normalize=False)
            
            for _ in range(iterations):
                # Hessian-vector product: ∇²L·v
                self.model.zero_grad()
                
                # 前向和后向传播计算梯度
                for data, target in data_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    # 计算梯度
                    grads = torch.autograd.grad(loss, self.model.parameters(), 
                                               create_graph=True)
                    
                    # 计算Hessian-vector product
                    gv = sum((g * vv).sum() for g, vv in zip(grads, v))
                    Hv = torch.autograd.grad(gv, self.model.parameters())
                    
                    break  # 只用一个batch近似
                
                # 更新v为Hv的归一化版本
                v_norm = np.sqrt(sum((hv ** 2).sum().item() for hv in Hv))
                v = [hv / (v_norm + 1e-10) for hv in Hv]
            
            # 特征值估计
            eigenvalue = v_norm
            eigenvalues.append(eigenvalue)
            print(f"  第{k+1}个特征值估计: {eigenvalue:.4f}")
        
        return np.array(eigenvalues)
    
    def analyze_sharpness(self, data_loader, criterion=None,
                         epsilon: float = 0.01, num_samples: int = 10) -> Dict:
        """
        分析损失景观的锐度 (Sharpness)
        
        锐度定义: $S = \max_{\|\delta\| \leq \epsilon} [L(\theta + \delta) - L(\theta)]$
        
        Args:
            data_loader: 数据加载器
            criterion: 损失函数
            epsilon: 扰动范围
            num_samples: 采样数量
        
        Returns:
            包含锐度分析结果的字典
        """
        original_params = self.get_model_params()
        original_loss = self.compute_loss(data_loader, criterion)
        
        max_loss_increase = 0.0
        avg_loss_increase = 0.0
        
        for i in range(num_samples):
            # 生成随机扰动
            perturbation = self.get_random_direction(normalize=False)
            
            # 归一化到epsilon范围
            pert_norm = np.sqrt(sum((p ** 2).sum().item() for p in perturbation))
            scale = epsilon / (pert_norm + 1e-10)
            perturbation = [p * scale for p in perturbation]
            
            # 应用扰动
            perturbed_params = [p + d for p, d in zip(original_params, perturbation)]
            self.set_model_params(perturbed_params)
            
            # 计算扰动后的损失
            perturbed_loss = self.compute_loss(data_loader, criterion)
            loss_increase = perturbed_loss - original_loss
            
            max_loss_increase = max(max_loss_increase, loss_increase)
            avg_loss_increase += loss_increase
        
        avg_loss_increase /= num_samples
        
        # 恢复原始参数
        self.set_model_params(original_params)
        
        return {
            'original_loss': original_loss,
            'max_sharpness': max_loss_increase,
            'avg_sharpness': avg_loss_increase,
            'epsilon': epsilon,
            'num_samples': num_samples
        }


class LossLandscapeVisualizer:
    """损失景观可视化器"""
    
    def __init__(self, save_dir: str = './loss_landscape_plots'):
        """
        初始化可视化器
        
        Args:
            save_dir: 保存图片的目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置matplotlib样式
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['figure.figsize'] = (10, 8)
    
    def plot_1d_interpolation(self, alphas: np.ndarray, losses: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: str = 'Loss Landscape: Linear Interpolation',
                              save_name: str = 'linear_interpolation_1d.png') -> str:
        """
        绘制1D线性插值图
        
        Args:
            alphas: 插值系数 (可以是多组)
            losses: 损失值 (可以是多组)
            labels: 图例标签
            title: 图标题
            save_name: 保存文件名
        
        Returns:
            保存的文件路径
        """
        plt.figure(figsize=(10, 6))
        
        if isinstance(losses, list):
            for i, (a, l) in enumerate(zip(alphas, losses)):
                label = labels[i] if labels else f'Model {i+1}'
                plt.plot(a, l, linewidth=2, label=label, marker='o', markersize=4)
        else:
            plt.plot(alphas, losses, linewidth=2, color='blue', marker='o', markersize=4)
        
        plt.xlabel(r'Interpolation Coefficient $\alpha$')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if labels:
            plt.legend()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_2d_contour(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                        title: str = 'Loss Landscape Contour',
                        save_name: str = 'loss_landscape_contour.png',
                        levels: int = 50) -> str:
        """
        绘制2D等高线图
        
        Args:
            X, Y: 网格坐标
            Z: 损失值矩阵
            title: 图标题
            save_name: 保存文件名
            levels: 等高线数量
        
        Returns:
            保存的文件路径
        """
        plt.figure(figsize=(10, 8))
        
        # 等高线图
        contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
        plt.colorbar(contour, label='Loss')
        
        # 添加等高线
        plt.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        
        # 标记最小值点
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        plt.plot(X[min_idx], Y[min_idx], 'r*', markersize=15, label=f'Min Loss: {Z[min_idx]:.4f}')
        
        plt.xlabel(r'Direction 1 ($\alpha$)')
        plt.ylabel(r'Direction 2 ($\beta$)')
        plt.title(title)
        plt.legend()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_3d_surface(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                        title: str = 'Loss Landscape 3D Surface',
                        save_name: str = 'loss_landscape_3d.png',
                        elevation: int = 30, azimuth: int = -60) -> str:
        """
        绘制3D曲面图
        
        Args:
            X, Y: 网格坐标
            Z: 损失值矩阵
            title: 图标题
            save_name: 保存文件名
            elevation: 仰角
            azimuth: 方位角
        
        Returns:
            保存的文件路径
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制曲面
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # 添加等高线投影
        ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap=cm.viridis, alpha=0.5)
        
        # 标记最小值点
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        ax.scatter([X[min_idx]], [Y[min_idx]], [Z[min_idx]], 
                   color='red', s=100, marker='*', label=f'Min: {Z[min_idx]:.4f}')
        
        ax.set_xlabel(r'Direction 1 ($\alpha$)')
        ax.set_ylabel(r'Direction 2 ($\beta$)')
        ax.set_zlabel('Loss')
        ax.set_title(title)
        ax.view_init(elev=elevation, azim=azimuth)
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Loss')
        
        ax.legend()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_comparison(self, results: Dict[str, Dict],
                        title: str = 'Loss Landscape Comparison',
                        save_name: str = 'loss_landscape_comparison.png') -> str:
        """
        绘制多方法对比图
        
        Args:
            results: 各方法的景观数据 {method_name: {'X': X, 'Y': Y, 'Z': Z}}
            title: 图标题
            save_name: 保存文件名
        
        Returns:
            保存的文件路径
        """
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
        
        if n_methods == 1:
            axes = [axes]
        
        for ax, (method_name, data) in zip(axes, results.items()):
            X, Y, Z = data['X'], data['Y'], data['Z']
            
            contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis')
            ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
            
            min_idx = np.unravel_index(np.argmin(Z), Z.shape)
            ax.plot(X[min_idx], Y[min_idx], 'r*', markersize=10)
            
            ax.set_xlabel(r'$\alpha$')
            ax.set_ylabel(r'$\beta$')
            ax.set_title(f'{method_name}\nMin Loss: {Z[min_idx]:.4f}')
            plt.colorbar(contour, ax=ax)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_sharpness_comparison(self, sharpness_results: Dict[str, Dict],
                                   title: str = 'Sharpness Analysis Comparison',
                                   save_name: str = 'sharpness_comparison.png') -> str:
        """
        绘制锐度对比柱状图
        
        Args:
            sharpness_results: 各方法的锐度分析结果
            title: 图标题
            save_name: 保存文件名
        
        Returns:
            保存的文件路径
        """
        methods = list(sharpness_results.keys())
        max_sharpness = [sharpness_results[m]['max_sharpness'] for m in methods]
        avg_sharpness = [sharpness_results[m]['avg_sharpness'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, max_sharpness, width, label='Max Sharpness', color='coral')
        bars2 = ax.bar(x + width/2, avg_sharpness, width, label='Avg Sharpness', color='steelblue')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Sharpness')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path


def run_loss_landscape_analysis(model_path: str, 
                                data_loader,
                                model_class,
                                device: str = 'cuda',
                                save_dir: str = './loss_landscape_results'):
    """
    运行完整的损失景观分析
    
    Args:
        model_path: 模型权重文件路径
        data_loader: 测试数据加载器
        model_class: 模型类
        device: 计算设备
        save_dir: 结果保存目录
    
    Returns:
        分析结果字典
    """
    print("="*60)
    print("损失景观分析 (Loss Landscape Analysis)")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 初始化分析器和可视化器
    analyzer = LossLandscapeAnalyzer(model, device)
    visualizer = LossLandscapeVisualizer(save_dir)
    
    results = {}
    
    # 1. 2D损失景观
    print("\n1. 计算2D损失景观...")
    X, Y, Z = analyzer.random_direction_2d(
        data_loader, 
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        steps=21
    )
    
    # 保存数据
    np.savez(os.path.join(save_dir, 'landscape_2d.npz'), X=X, Y=Y, Z=Z)
    
    # 可视化
    visualizer.plot_2d_contour(X, Y, Z, 
                               title='Loss Landscape (2D Contour)',
                               save_name='landscape_2d_contour.png')
    visualizer.plot_3d_surface(X, Y, Z,
                               title='Loss Landscape (3D Surface)',
                               save_name='landscape_3d_surface.png')
    
    results['landscape_2d'] = {'X': X, 'Y': Y, 'Z': Z}
    
    # 2. 锐度分析
    print("\n2. 分析损失景观锐度...")
    sharpness = analyzer.analyze_sharpness(data_loader, epsilon=0.01, num_samples=20)
    results['sharpness'] = sharpness
    print(f"  最大锐度: {sharpness['max_sharpness']:.6f}")
    print(f"  平均锐度: {sharpness['avg_sharpness']:.6f}")
    
    # 3. 梯度范数
    print("\n3. 计算梯度范数...")
    grad_norm = analyzer.compute_gradient_norm(data_loader)
    results['gradient_norm'] = grad_norm
    print(f"  梯度范数: {grad_norm:.6f}")
    
    # 保存结果
    with open(os.path.join(save_dir, 'analysis_results.json'), 'w') as f:
        json.dump({
            'sharpness': sharpness,
            'gradient_norm': float(grad_norm),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("损失景观分析完成!")
    print(f"结果保存至: {save_dir}")
    print("="*60)
    
    return results


if __name__ == '__main__':
    print("损失景观分析模块")
    print("用法: 从run_loss_landscape_analysis函数调用")
    print("或导入LossLandscapeAnalyzer和LossLandscapeVisualizer类使用")
