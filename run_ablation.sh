#!/bin/bash
#============================================================================
# FedLGO Ablation Experiments Script (Ubuntu/Linux)
#============================================================================
# Variants:
#   - FedLGO: Full model (LGO + MGC)
#   - FedLGO_woLGO: Without Landscape-Guided Optimization
#   - FedLGO_woMGC: Without Momentum-Guided Gradient Correction
#   - FedLGO_woBoth: Without both (baseline)
#============================================================================

set -e

# Configuration
DATASETS=("Uci" "Xinwang")
HETEROGENEITY=("iid" "feature" "label" "quantity")
ABLATION_VARIANTS=("FedLGO" "FedLGO_woLGO" "FedLGO_woMGC" "FedLGO_woBoth")

GLOBAL_ROUNDS=100
LOCAL_EPOCHS=5
BATCH_SIZE=64
LEARNING_RATE=0.005
NUM_CLIENTS=10
JOIN_RATIO=1.0
REPETITIONS=3
DEVICE="cuda"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================================"
echo "FedLGO Ablation Experiments"
echo "============================================================"
echo "Datasets: ${DATASETS[*]}"
echo "Variants: ${ABLATION_VARIANTS[*]}"
echo "Heterogeneity: ${HETEROGENEITY[*]}"
echo "============================================================"

cd system

for dataset in "${DATASETS[@]}"; do
    for hetero in "${HETEROGENEITY[@]}"; do
        for variant in "${ABLATION_VARIANTS[@]}"; do
            echo -e "${YELLOW}Running: $dataset / $variant / $hetero${NC}"
            
            for rep in $(seq 0 $((REPETITIONS-1))); do
                result_dir="../system/results/${dataset}_${variant}_${hetero}"
                
                if [ -d "$result_dir" ] && [ $(ls -1 "$result_dir"/*.h5 2>/dev/null | wc -l) -ge $REPETITIONS ]; then
                    echo -e "${GREEN}[SKIP] Already completed${NC}"
                    continue
                fi
                
                echo "  Repetition $((rep+1))/$REPETITIONS"
                
                python -u main.py \
                    -data "$dataset" \
                    -m credit \
                    -algo "$variant" \
                    -gr $GLOBAL_ROUNDS \
                    -ls $LOCAL_EPOCHS \
                    -lbs $BATCH_SIZE \
                    -lr $LEARNING_RATE \
                    -nc $NUM_CLIENTS \
                    -jr $JOIN_RATIO \
                    -dev $DEVICE \
                    -go "$hetero" \
                    -t 1 \
                    -pv $rep
                
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}  [OK] Completed${NC}"
                else
                    echo -e "${RED}  [FAIL] Error occurred${NC}"
                fi
            done
        done
    done
done

echo "============================================================"
echo "All ablation experiments completed!"
echo "============================================================"
