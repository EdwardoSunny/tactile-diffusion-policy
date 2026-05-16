#!/bin/bash
set -e
set -o pipefail

source /home/edward/miniforge3/etc/profile.d/conda.sh
conda activate /home/edward/.local/share/mamba/envs/robodiff

cd /data/edward/diffusion_policy
export CUDA_VISIBLE_DEVICES=6
export WANDB_MODE=disabled

for variant in raw arrow bar; do
    echo ""
    echo "================================================================"
    echo "=== Starting xarm_${variant} at $(date) ==="
    echo "================================================================"
    python -u train.py --config-name=train_diffusion_unet_image_workspace \
        task=xarm_image \
        task.dataset.zarr_path=/data/edward/diffusion_policy_data/xarm_${variant}.zarr \
        training.device=cuda:0 \
        training.seed=42 \
        training.num_epochs=1000 \
        logging.mode=disabled \
        checkpoint.topk.k=0 \
        hydra.run.dir=data/outputs/xarm_${variant}
    echo "=== Done xarm_${variant} at $(date) ==="
done

echo ""
echo "================================================================"
echo "=== All 3 runs complete at $(date) ==="
echo "================================================================"
