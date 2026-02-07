# FedLGO: Federated Landscape-Guided Optimization with Gradient Correction

FedLGO is a novel federated learning algorithm that leverages loss landscape analysis to improve convergence in non-IID scenarios.

## Project Structure

```
FedLGO/
├── dataset/                    # Datasets
│   ├── Uci/                    # UCI Credit Card Dataset
│   ├── Xinwang/                # Xinwang Credit Dataset
│   └── utils/                  # Data utilities
├── system/                     # Federated learning system
│   ├── flcore/clients/         # Client implementations
│   ├── flcore/servers/         # Server implementations
│   ├── flcore/trainmodel/      # Model definitions
│   └── main.py                 # Main entry
├── EAAI/                       # Paper for EAAI submission
├── run_comparison.sh           # Comparison experiments (Ubuntu)
└── run_ablation.sh             # Ablation experiments (Ubuntu)
```

## Setup (Ubuntu/Linux)

```bash
conda create -n fedlgo python=3.8
conda activate fedlgo
pip install torch>=1.10.0 numpy pandas scikit-learn h5py matplotlib tqdm
```

## Data Preparation

```bash
cd dataset
python generate_all_datasets_auto.py
```

## Run Experiments

Single algorithm:
```bash
cd system
python main.py -algo FedLGO -data Uci -go label -gr 100 -nc 10 -t 3
```

Comparison experiments:
```bash
chmod +x run_comparison.sh
./run_comparison.sh
```

Ablation study:
```bash
chmod +x run_ablation.sh
./run_ablation.sh
```

## Supported Algorithms

| Category | Algorithms |
|----------|-----------|
| Baseline | FedAvg |
| Regularization | FedProx, SCAFFOLD, MOON |
| Knowledge Transfer | FedGen, FedProto |
| Personalization | FedRep, Ditto, Per-FedAvg |
| **Ours** | **FedLGO** |

## FedLGO Algorithm

FedLGO addresses non-IID data through two core components:

### 1. Landscape-Guided Optimization (LGO)
- **Steepness**: Gradient norm indicating optimization difficulty
- **Flatness**: Hessian trace approximation indicating generalization potential
- Adaptive client selection based on landscape features
- Landscape-aware learning rate adjustment

### 2. Momentum-Guided Gradient Correction (MGC)
- Global momentum buffer for gradient direction smoothing
- Gradient alignment to reduce client drift
- Variance reduction through momentum updates

### Landscape Score:
```
score_k = steepness_k + λ·flatness_k
```

### Gradient Correction:
```
Δw = Δw_local + γ·(momentum_global - momentum_local)
```

## Ablation Variants

| Variant | LGO | MGC |
|---------|-----|-----|
| FedLGO (Full) | ✓ | ✓ |
| FedLGO_woLGO | ✗ | ✓ |
| FedLGO_woMGC | ✓ | ✗ |
| FedLGO_woBoth | ✗ | ✗ |

## Parameters

FedLGO parameters:
| Parameter | Description | Default |
|-----------|-------------|---------|
| lgo_tau_max | Initial temperature for selection | 2.0 |
| lgo_tau_min | Minimum temperature | 0.5 |
| lgo_alpha_lr | Adaptive learning rate factor | 0.3 |
| lgo_beta | Momentum coefficient | 0.9 |
| lgo_gamma | Gradient correction strength | 0.5 |

General parameters:
| Parameter | Description | Default |
|-----------|-------------|---------|
| -gr | Global rounds | 100 |
| -ls | Local epochs | 5 |
| -lbs | Batch size | 64 |
| -lr | Learning rate | 0.005 |
| -nc | Number of clients | 10 |
| -t | Repetitions | 3 |

## Datasets

**UCI Credit**: Taiwan credit card default dataset with 30,000 records, 23 features, binary classification.

**Xinwang**: Chinese online lending platform credit data with 50,000+ records, 38 features, binary classification.

Four heterogeneity types: IID, Feature skew, Label skew, Quantity skew.

## Citation

If you use this code, please cite our paper (under review at EAAI).

## License

MIT License
