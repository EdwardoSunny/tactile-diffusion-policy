#!/bin/bash
# Run 3 xarm variants on GPU 6 with 2-at-a-time concurrency:
#   Phase 1: raw + arrow start in parallel
#   Phase 2: whichever finishes first triggers bar
#   Phase 3: bar continues with the slower of (raw, arrow)
# Wall time is similar to sequential when GPU is the bottleneck (95% util),
# but keeps 2 jobs active most of the time as requested.
set -uo pipefail

source /home/edward/miniforge3/etc/profile.d/conda.sh
conda activate /home/edward/.local/share/mamba/envs/robodiff

cd /data/edward/diffusion_policy
export CUDA_VISIBLE_DEVICES=6
export WANDB_MODE=disabled

run_variant() {
    local variant=$1
    local log_file="data/outputs/xarm_${variant}_run.log"
    echo "=== Starting xarm_${variant} at $(date) ===" | tee -a "$log_file"
    python -u train.py --config-name=train_diffusion_unet_image_workspace \
        task=xarm_image \
        task.dataset.zarr_path=/data/edward/diffusion_policy_data/xarm_${variant}.zarr \
        training.device=cuda:0 \
        training.seed=42 \
        training.num_epochs=1000 \
        dataloader.num_workers=2 \
        val_dataloader.num_workers=2 \
        logging.mode=disabled \
        checkpoint.topk.k=0 \
        hydra.run.dir=data/outputs/xarm_${variant} >> "$log_file" 2>&1
    local rc=$?
    echo "=== Done xarm_${variant} at $(date) (exit ${rc}) ===" | tee -a "$log_file"
    return $rc
}

mkdir -p data/outputs

run_variant raw &
PID_RAW=$!
echo "[$(date)] launched raw (pid ${PID_RAW})"

run_variant arrow &
PID_ARROW=$!
echo "[$(date)] launched arrow (pid ${PID_ARROW})"

# Wait for the first of raw/arrow to complete
wait -n
echo "[$(date)] one of raw/arrow finished — starting bar"

run_variant bar &
PID_BAR=$!
echo "[$(date)] launched bar (pid ${PID_BAR})"

# Wait for the remaining two
wait
echo "[$(date)] === All 3 runs complete ==="
