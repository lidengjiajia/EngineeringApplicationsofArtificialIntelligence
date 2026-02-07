# FedLGO Paper - Engineering Applications of Artificial Intelligence

## 文件说明

- `main.tex` - LaTeX 论文源文件（使用 elsarticle 模板）
- `elsarticle-num-names.bst` - 参考文献格式文件

## 编译命令

```bash
pdflatex main.tex
pdflatex main.tex  # 运行两次以解决交叉引用
```

## 论文概述

**FedLGO: Federated Landscape-Guided Optimization with Gradient Correction for Collaborative Credit Scoring**

### 核心创新

1. **Landscape-Guided Optimization (LGO)** - 景观引导优化
   - Steepness (梯度幅度): 量化客户端距离最优点的距离
   - Flatness (Hessian迹): 估计局部曲率，关联泛化能力
   - 自适应客户端选择 + 自适应学习率

2. **Momentum-Guided Gradient Correction (MGC)** - 动量引导梯度校正
   - 全局动量缓冲对齐异质梯度方向
   - 通信开销与FedAvg相同（不像SCAFFOLD需要翻倍）

### 实验设置

- 数据集: UCI Credit Card, Xinwang Credit
- 异质性场景: IID, Feature Skew, Label Skew, Quantity Skew
- Baseline: FedAvg, FedProx, SCAFFOLD, MOON, FedGen, FedProto, FedRep, Ditto, Per-FedAvg
