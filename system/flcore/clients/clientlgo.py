"""
FedLGO Client Implementation
Federated Landscape-Guided Optimization with Gradient Correction

客户端功能：
1. 本地训练（自适应学习率）
2. 景观特征计算（Steepness + Flatness）
3. 梯度校正（Momentum-Guided）
4. 特征统计计算

作者：Anonymous
日期：2026-02-05
"""

import copy
import time
import torch
import torch.nn as nn
import numpy as np
from flcore.clients.clientbase import Client


class clientLGO(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # ==================== 景观特征 ====================
        self.steepness = 0.0          # 梯度幅度 ||∇F_k||²
        self.flatness = 0.0           # Hessian迹（曲率）
        self.landscape_score = 0.0    # 景观得分 = steepness / (1 + flatness)
        
        # ==================== 梯度信息 ====================
        self.gradient = None          # 本地梯度（伪梯度：θ_old - θ_new）
        self.corrected_gradient = None  # 校正后的梯度
        
        # ==================== 自适应学习率 ====================
        self.adaptive_lr = self.learning_rate  # 自适应学习率
        
        # ==================== 特征统计缓存 ====================
        self.feature_stats = None
        self.label_dist = None
        
        # ==================== Hutchinson估计参数 ====================
        self.num_hutchinson_samples = getattr(args, 'lgo_hutchinson_samples', 10)
        
        # 计算初始统计
        self._compute_statistics()
    
    def compute_landscape_features(self):
        """计算景观特征：Steepness和Flatness"""
        trainloader = self.load_train_data()
        
        self.model.train()
        
        # 1. 计算梯度幅度（Steepness）
        total_grad_norm = 0.0
        num_batches = 0
        
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            
            # 计算梯度范数
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            total_grad_norm += grad_norm
            num_batches += 1
            
            if num_batches >= 5:  # 只用前5个batch估计
                break
        
        self.steepness = total_grad_norm / max(num_batches, 1)
        
        # 2. 计算Flatness（Hessian迹估计 - Hutchinson方法）
        self.flatness = self._estimate_hessian_trace()
        
        # 3. 计算景观得分
        self.landscape_score = self.steepness / (1.0 + self.flatness + 1e-10)
        
        return {
            'steepness': self.steepness,
            'flatness': self.flatness,
            'score': self.landscape_score
        }
    
    def _estimate_hessian_trace(self):
        """使用Hutchinson方法估计Hessian迹"""
        trainloader = self.load_train_data()
        
        self.model.train()
        
        # 获取一个batch用于估计
        try:
            x, y = next(iter(trainloader))
        except StopIteration:
            return 0.0
        
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        
        # 计算损失
        output = self.model(x)
        loss = self.loss(output, y)
        
        # 计算梯度
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True, allow_unused=True)
        
        # Hutchinson估计
        trace_estimate = 0.0
        
        for _ in range(self.num_hutchinson_samples):
            # 生成随机Rademacher向量 (±1)，使用float类型以兼容梯度计算
            v = [(torch.randint_like(p, high=2).float() * 2 - 1) for p in self.model.parameters()]
            
            # 计算 v^T H v = v^T ∇(∇L · v)
            grad_v = sum(
                (g * vi).sum() for g, vi in zip(grads, v) if g is not None
            )
            
            if grad_v.requires_grad:
                hvp = torch.autograd.grad(grad_v, self.model.parameters(), retain_graph=True, allow_unused=True)
                
                # v^T H v
                vhv = sum(
                    (h * vi).sum().item() for h, vi in zip(hvp, v) if h is not None
                )
                trace_estimate += vhv
        
        trace_estimate /= self.num_hutchinson_samples
        
        return max(trace_estimate, 0.0)  # 确保非负
    
    def set_adaptive_learning_rate(self, base_lr, alpha, avg_flatness):
        """设置自适应学习率"""
        # Eq. (7): η_k = η_base × (1 + α × (Φ_avg - Φ_k) / (Φ_avg + ε))
        if avg_flatness > 1e-10:
            lr_factor = 1.0 + alpha * (avg_flatness - self.flatness) / (avg_flatness + 1e-10)
            lr_factor = np.clip(lr_factor, 0.5, 2.0)  # 限制范围
        else:
            lr_factor = 1.0
        
        self.adaptive_lr = base_lr * lr_factor
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.adaptive_lr
    
    def train(self):
        """客户端本地训练"""
        trainloader = self.load_train_data()
        
        self.model.train()
        
        # 保存训练前的模型参数（用于计算伪梯度）
        old_params = [param.data.clone() for param in self.model.parameters()]
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                output = self.model(x)
                loss = self.loss(output, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（防止爆炸）
                if self.enable_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
        
        # 计算伪梯度：g_k = θ_old - θ_new
        self.gradient = [
            old_param - new_param.data
            for old_param, new_param in zip(old_params, self.model.parameters())
        ]
        
        # 学习率调度
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
    
    def apply_gradient_correction(self, global_momentum, gamma):
        """
        应用梯度校正
        Eq. (9): g̃_k = g_k - γ(g_k - m_{t-1})
        """
        if self.gradient is None:
            return
        
        if global_momentum is None:
            self.corrected_gradient = self.gradient
            return
        
        self.corrected_gradient = []
        for g_k, m in zip(self.gradient, global_momentum):
            # g̃_k = g_k - γ(g_k - m) = (1-γ)g_k + γm
            corrected = (1 - gamma) * g_k + gamma * m
            self.corrected_gradient.append(corrected)
    
    def get_gradient(self):
        """返回原始梯度"""
        return self.gradient
    
    def get_corrected_gradient(self):
        """返回校正后的梯度"""
        if self.corrected_gradient is not None:
            return self.corrected_gradient
        return self.gradient
    
    def get_landscape_features(self):
        """返回景观特征"""
        return {
            'steepness': self.steepness,
            'flatness': self.flatness,
            'score': self.landscape_score
        }
    
    def _compute_statistics(self):
        """计算特征统计和标签分布"""
        trainloader = self.load_train_data()
        
        all_features = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                
                # 收集特征
                if isinstance(x, list):
                    features = x[0].cpu().numpy()
                else:
                    features = x.cpu().numpy()
                
                # 展平特征
                features = features.reshape(features.shape[0], -1)
                all_features.append(features)
                
                # 收集标签
                all_labels.append(y.numpy())
        
        # 合并所有批次
        if len(all_features) > 0:
            all_features = np.vstack(all_features)
            all_labels = np.concatenate(all_labels)
            
            # 计算特征统计
            self.feature_stats = {
                'mean': np.mean(all_features, axis=0),
                'std': np.std(all_features, axis=0) + 1e-10
            }
            
            # 计算标签分布
            num_classes = self.num_classes if hasattr(self, 'num_classes') else 2
            label_counts = np.bincount(all_labels.astype(int), minlength=num_classes)
            self.label_dist = label_counts.astype(np.float32)
        else:
            self.feature_stats = {'mean': np.zeros(1), 'std': np.ones(1)}
            self.label_dist = np.ones(2)
    
    def get_feature_statistics(self):
        """返回特征统计"""
        if self.feature_stats is None:
            self._compute_statistics()
        return self.feature_stats
    
    def get_label_distribution(self):
        """返回标签分布"""
        if self.label_dist is None:
            self._compute_statistics()
        return self.label_dist
    
    def test_metrics_simple(self):
        """在本地测试集上评估模型，返回准确率"""
        testloader = self.load_test_data()
        
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def set_parameters(self, model):
        """设置模型参数"""
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
