#!/bin/bash
#============================================================================
# FedLGO Comparison Experiments Script (Ubuntu/Linux)
#============================================================================
# Baselines: FedAvg, FedProx, SCAFFOLD, MOON, FedGen, FedProto, FedRep, Ditto, Per-FedAvg
# Our method: FedLGO
# Datasets: Uci, Xinwang
# Heterogeneity: iid, feature, label, quantity
#============================================================================

set -e

# Configuration
DATASETS=("Uci" "Xinwang")
HETEROGENEITY=("iid" "feature" "label" "quantity")
ALGORITHMS=("FedAvg" "FedProx" "SCAFFOLD" "MOON" "FedGen" "FedProto" "FedRep" "Ditto" "Per-FedAvg" "FedLGO")

GLOBAL_ROUNDS=100
LOCAL_EPOCHS=5
BATCH_SIZE=64
LEARNING_RATE=0.005
NUM_CLIENTS=10
JOIN_RATIO=1.0
REPETITIONS=3
DEVICE="cuda"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "FedLGO Comparison Experiments"
echo "============================================================"
echo "Datasets: ${DATASETS[*]}"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Heterogeneity types: ${HETEROGENEITY[*]}"
echo "Repetitions: $REPETITIONS"
echo "============================================================"

cd system

for dataset in "${DATASETS[@]}"; do
    for hetero in "${HETEROGENEITY[@]}"; do
        for algo in "${ALGORITHMS[@]}"; do
            echo -e "${YELLOW}Running: $dataset / $algo / $hetero${NC}"
            
            for rep in $(seq 0 $((REPETITIONS-1))); do
                result_dir="../system/results/${dataset}_${algo}_${hetero}"
                
                # Skip if already completed
                if [ -d "$result_dir" ] && [ $(ls -1 "$result_dir"/*.h5 2>/dev/null | wc -l) -ge $REPETITIONS ]; then
                    echo -e "${GREEN}[SKIP] Already completed${NC}"
                    continue
                fi
                
                echo "  Repetition $((rep+1))/$REPETITIONS"
                
                python -u main.py \
                    -data "$dataset" \
                    -m credit \
                    -algo "$algo" \
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
echo "All comparison experiments completed!"
echo "============================================================"
