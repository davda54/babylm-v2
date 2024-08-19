#!/bin/bash
PROJECT="project_465000144"
CONTAINER="/scratch/project_465000498/HPLT-WP4/pytorch-lumi_sles-rocm-5.5.1-python-3.10-pytorch-v2.0.1-apex-torchvision-torchdata-torchtext-torchaudio.sif"
BIND="/scratch/project_465000144"
srun \
    --account="$PROJECT" \
    --partition=dev-g \
    --ntasks=1 \
    --gres=gpu:mi250:1 \
    --time=3:00:00 \
    --mem=256G \
    --pty \
    singularity exec -B "$BIND" "$CONTAINER" \
    bash