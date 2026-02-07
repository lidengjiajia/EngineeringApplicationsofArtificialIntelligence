"""
FedLGO: Federated Landscape-Guided Optimization with Gradient Correction

核心创新：
1. 景观引导优化 (Landscape-Guided Optimization, LGO)
   - 景观特征提取：Steepness（梯度幅度）和Flatness（Hessian迹）
   - 自适应客户端选择：基于景观得分的概率选择
   - 自适应学习率：根据曲率调整步长

2. 动量引导梯度校正 (Momentum-Guided Gradient Correction, MGC)
   - 全局动量估计：累积历史梯度信息
   - 梯度偏差校正：将局部梯度对齐到全局动量方向

消融实验变体：
- FedLGO: 完整框架
- w/o LGO: 移除景观引导（均匀选择 + 固定学习率）
- w/o MGC: 移除梯度校正（γ=0, λ=0）
- w/o Both: 移除两者（等同于FedAvg）

作者：Anonymous
日期：2026-02-05
"""

import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from flcore.clients.clientlgo import clientLGO
from flcore.servers.serverbase import Server
from threading import Thread


class FedLGO(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # 设置客户端
        self.set_slow_clients()
        self.set_clients(clientLGO)
        
        # ==================== 消融实验开关 ====================
        self.use_landscape = getattr(args, 'lgo_use_landscape', True)      # 景观引导
        self.use_gradient_correction = getattr(args, 'lgo_use_correction', True)  # 梯度校正
        
        # ==================== 景观引导参数 (LGO) ====================
        # 温度参数（客户端选择）
        self.tau_max = getattr(args, 'lgo_tau_max', 2.0)          # 初始温度
        self.tau_min = getattr(args, 'lgo_tau_min', 0.5)          # 最小温度
        
        # 自适应学习率参数
        self.alpha_lr = getattr(args, 'lgo_alpha_lr', 0.3)        # 学习率调整强度
        
        # 聚合权重参数
        self.delta_weight = getattr(args, 'lgo_delta_weight', 0.2)  # 景观得分影响系数
        
        # ==================== 梯度校正参数 (MGC) ====================
        self.beta_momentum = getattr(args, 'lgo_beta', 0.9)       # 动量系数
        self.gamma_correction = getattr(args, 'lgo_gamma', 0.5)   # 校正强度
        self.lambda_momentum = getattr(args, 'lgo_lambda', 0.1)   # 动量贡献系数
        
        # ==================== 状态变量 ====================
        self.global_momentum = None                # 全局动量缓冲
        self.landscape_features = {}               # 客户端景观特征
        self.aggregation_weights = {}              # 聚合权重
        
        # ==================== 监控和记录 ====================
        self.landscape_history = []                # 景观特征历史
        self.weight_history = []                   # 权重历史
        self.selection_history = []                # 选择概率历史
        
        # 打印配置
        print(f"\n{'='*80}")
        print(f"FedLGO - Landscape-Guided Optimization with Gradient Correction")
        print(f"{'='*80}")
        print(f"客户端数量: {self.num_clients}")
        print(f"参与率: {self.join_ratio}")
        print(f"\n消融实验配置:")
        print(f"  use_landscape (LGO): {self.use_landscape}")
        print(f"  use_gradient_correction (MGC): {self.use_gradient_correction}")
        print(f"\n景观引导参数 (LGO):")
        print(f"  温度退火: τ_max={self.tau_max}, τ_min={self.tau_min}")
        print(f"  自适应学习率: α={self.alpha_lr}")
        print(f"  权重系数: δ={self.delta_weight}")
        print(f"\n梯度校正参数 (MGC):")
        print(f"  动量系数: β={self.beta_momentum}")
        print(f"  校正强度: γ={self.gamma_correction}")
        print(f"  动量贡献: λ={self.lambda_momentum}")
        print(f"{'='*80}\n")
        
        self.Budget = []
    
    def train(self):
        """主训练循环"""
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            
            # 步骤1：收集景观特征并选择客户端
            if i > 0 and self.use_landscape:
                self._collect_landscape_features()
                self.selected_clients = self._landscape_aware_selection(i)
            else:
                self.selected_clients = self.select_clients()
            
            # 步骤2：发送全局模型和动量
            self.send_models()
            
            # 步骤3：评估（周期性）
            if i % self.eval_gap == 0:
                print(f"\n{'='*80}")
                print(f"Round {i}/{self.global_rounds}")
                print(f"{'='*80}")
                self.evaluate()
            
            # 步骤4：客户端本地训练
            if i > 0 and self.use_landscape:
                self._set_adaptive_learning_rates()
            
            for client in self.selected_clients:
                client.train()
            
            # 步骤5：接收模型
            self.receive_models()
            
            # 步骤6：梯度校正和聚合
            if i > 0:
                if self.use_gradient_correction:
                    self._apply_gradient_correction()
                
                self._landscape_guided_aggregation(i)
            else:
                # 第0轮：标准FedAvg
                self.aggregate_parameters()
            
            # 打印诊断
            if i % self.eval_gap == 0 and i > 0:
                self._print_diagnostics(i)
            
            self.Budget.append(time.time() - s_t)
            print(f"\nRound {i} 耗时: {self.Budget[-1]:.2f}秒")
            
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        print("\n" + "="*80)
        print("训练完成！")
        print("="*80)
        self.print_results()
        self.save_results()
        self.save_global_model()
    
    def _collect_landscape_features(self):
        """收集所有客户端的景观特征"""
        print("  收集景观特征...")
        
        for client in self.clients:
            features = client.compute_landscape_features()
            self.landscape_features[client.id] = features
        
        # 记录历史
        self.landscape_history.append({
            'round': len(self.landscape_history),
            'features': copy.deepcopy(self.landscape_features)
        })
    
    def _landscape_aware_selection(self, round_idx):
        """基于景观特征的客户端选择"""
        # 计算当前温度
        progress = round_idx / self.global_rounds
        tau = self.tau_max - progress * (self.tau_max - self.tau_min)
        
        # 获取景观得分
        scores = []
        available_clients = []
        
        for client in self.clients:
            if client.id in self.landscape_features:
                score = self.landscape_features[client.id]['score']
                scores.append(score)
                available_clients.append(client)
        
        if len(available_clients) == 0:
            return self.select_clients()
        
        scores = np.array(scores)
        
        # Softmax选择概率
        scores_normalized = (scores - scores.mean()) / (scores.std() + 1e-10)
        probs = np.exp(scores_normalized / tau)
        probs = probs / probs.sum()
        
        # 采样客户端
        num_select = min(self.current_num_join_clients, len(available_clients))
        selected_indices = np.random.choice(
            len(available_clients),
            size=num_select,
            replace=False,
            p=probs
        )
        
        selected = [available_clients[i] for i in selected_indices]
        
        # 记录选择概率
        self.selection_history.append({
            'round': round_idx,
            'probs': probs.tolist(),
            'selected': [c.id for c in selected]
        })
        
        return selected
    
    def _set_adaptive_learning_rates(self):
        """为选中的客户端设置自适应学习率"""
        if not self.use_landscape:
            return
        
        # 计算平均Flatness
        flatness_values = [
            self.landscape_features[c.id]['flatness']
            for c in self.selected_clients
            if c.id in self.landscape_features
        ]
        
        if len(flatness_values) == 0:
            return
        
        avg_flatness = np.mean(flatness_values)
        
        # 设置每个客户端的自适应学习率
        for client in self.selected_clients:
            client.set_adaptive_learning_rate(
                self.learning_rate,
                self.alpha_lr,
                avg_flatness
            )
    
    def _apply_gradient_correction(self):
        """应用梯度校正"""
        if not self.use_gradient_correction:
            return
        
        for client in self.selected_clients:
            client.apply_gradient_correction(
                self.global_momentum,
                self.gamma_correction
            )
    
    def _landscape_guided_aggregation(self, round_idx):
        """景观引导的聚合"""
        
        # 收集梯度
        corrected_gradients = []
        sample_weights = []
        landscape_scores = []
        
        total_samples = sum(c.train_samples for c in self.selected_clients)
        
        for client in self.selected_clients:
            # 获取（校正后的）梯度
            if self.use_gradient_correction:
                grad = client.get_corrected_gradient()
            else:
                grad = client.get_gradient()
            
            if grad is not None:
                corrected_gradients.append(grad)
                sample_weights.append(client.train_samples / total_samples)
                
                # 获取景观得分
                if client.id in self.landscape_features and self.use_landscape:
                    score = self.landscape_features[client.id]['score']
                else:
                    score = 1.0
                landscape_scores.append(score)
        
        if len(corrected_gradients) == 0:
            print("Warning: No gradients received, skipping aggregation")
            return
        
        # 计算聚合权重
        sample_weights = np.array(sample_weights)
        landscape_scores = np.array(landscape_scores)
        
        if self.use_landscape and landscape_scores.std() > 1e-10:
            # Eq. (10): w_k = p_k × (1 + δ × score_k) / Σ(...)
            score_factor = 1.0 + self.delta_weight * (landscape_scores - landscape_scores.mean()) / (landscape_scores.std() + 1e-10)
            score_factor = np.clip(score_factor, 0.5, 2.0)
            weights = sample_weights * score_factor
        else:
            weights = sample_weights
        
        weights = weights / weights.sum()
        
        # 保存权重记录
        self.aggregation_weights = {
            client.id: w for client, w in zip(self.selected_clients, weights)
        }
        self.weight_history.append({
            'round': round_idx,
            'weights': copy.deepcopy(self.aggregation_weights)
        })
        
        # 聚合梯度：g̅_t = Σ w_k g̃_k
        aggregated_gradient = []
        for param_idx in range(len(corrected_gradients[0])):
            weighted_sum = sum(
                w * grad[param_idx]
                for w, grad in zip(weights, corrected_gradients)
            )
            aggregated_gradient.append(weighted_sum)
        
        # 更新全局动量：m_t = β m_{t-1} + (1-β) g̅_t
        if self.use_gradient_correction:
            if self.global_momentum is None:
                self.global_momentum = [g.clone() for g in aggregated_gradient]
            else:
                self.global_momentum = [
                    self.beta_momentum * m + (1 - self.beta_momentum) * g
                    for m, g in zip(self.global_momentum, aggregated_gradient)
                ]
        
        # 更新全局模型：θ_{t+1} = θ_t - η_g (g̅_t + λ m_t)
        with torch.no_grad():
            for param, grad in zip(self.global_model.parameters(), aggregated_gradient):
                update = grad
                if self.use_gradient_correction and self.global_momentum is not None:
                    # 找到对应的动量
                    idx = list(self.global_model.parameters()).index(param)
                    update = grad + self.lambda_momentum * self.global_momentum[idx]
                param.data -= self.learning_rate * update
    
    def _print_diagnostics(self, round_idx):
        """打印诊断信息"""
        print(f"\n--- FedLGO 诊断 (Round {round_idx}) ---")
        
        if self.use_landscape and len(self.landscape_features) > 0:
            steepness_vals = [f['steepness'] for f in self.landscape_features.values()]
            flatness_vals = [f['flatness'] for f in self.landscape_features.values()]
            score_vals = [f['score'] for f in self.landscape_features.values()]
            
            print(f"景观特征统计:")
            print(f"  Steepness: mean={np.mean(steepness_vals):.4f}, std={np.std(steepness_vals):.4f}")
            print(f"  Flatness:  mean={np.mean(flatness_vals):.4f}, std={np.std(flatness_vals):.4f}")
            print(f"  Score:     mean={np.mean(score_vals):.4f}, std={np.std(score_vals):.4f}")
        
        if len(self.aggregation_weights) > 0:
            weights = list(self.aggregation_weights.values())
            print(f"聚合权重: mean={np.mean(weights):.4f}, std={np.std(weights):.4f}, "
                  f"max={np.max(weights):.4f}, min={np.min(weights):.4f}")
        
        if self.global_momentum is not None:
            momentum_norms = [m.norm().item() for m in self.global_momentum]
            print(f"动量范数: mean={np.mean(momentum_norms):.4f}")
    
    def send_models(self):
        """发送模型（覆盖基类方法以发送动量）"""
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)
            
            # 动量信息通过client的gradient correction使用
            # 不需要显式发送，因为在apply_gradient_correction时使用self.global_momentum

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    
    def receive_models(self):
        """接收模型"""
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


# ============================================================================
# 消融实验变体工厂函数
# ============================================================================

def create_fedlgo_variant(args, times, use_landscape=True, use_correction=True):
    """
    创建FedLGO变体用于消融实验
    
    Args:
        args: 参数
        times: 运行次数
        use_landscape: 是否使用景观引导 (LGO模块)
        use_correction: 是否使用梯度校正 (MGC模块)
    
    Returns:
        FedLGO server实例
    """
    args.lgo_use_landscape = use_landscape
    args.lgo_use_correction = use_correction
    return FedLGO(args, times)


class FedLGO_woLGO(FedLGO):
    """FedLGO without Landscape-Guided Optimization"""
    def __init__(self, args, times):
        args.lgo_use_landscape = False
        args.lgo_use_correction = True
        super().__init__(args, times)


class FedLGO_woMGC(FedLGO):
    """FedLGO without Momentum-Guided Gradient Correction"""
    def __init__(self, args, times):
        args.lgo_use_landscape = True
        args.lgo_use_correction = False
        super().__init__(args, times)


class FedLGO_woBoth(FedLGO):
    """FedLGO without both modules (baseline)"""
    def __init__(self, args, times):
        args.lgo_use_landscape = False
        args.lgo_use_correction = False
        super().__init__(args, times)
