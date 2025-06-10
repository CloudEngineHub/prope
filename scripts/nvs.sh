#! /bin/bash
# 
# Usage (2 GPUs)
# ./scripts/nvs.sh -m plucker-prope -g 0,1
# ./scripts/nvs.sh -m plucker-gta -g 0,1

while getopts ":m:g:" opt; do
  case $opt in
    m) MODE="$OPTARG" # e.g, "raymap"
    ;;
    g) GPUS="$OPTARG" # e.g, "0,1"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done


NGPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

NAME="lvsm-b8-s1-20k-qknorm"
BASE_CMD=(
    "NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=$NGPUS"
    "nvs/trainval.py lvsm"
    "--amp --amp_dtype fp16"
    "--dataset_batch_scenes 8"
    "--dataset_supervise_views 1"
    "--model_config.encoder.num_layers 6"
    "--model_config.encoder.layer.d_model 768"
    "--model_config.encoder.layer.nhead 16"
    "--model_config.encoder.layer.dim_feedforward 1024"
    "--model_config.encoder.layer.qk_norm"
    "--max_steps 20000 --test_every 2000"
)

case $MODE in
    plucker-none)
        CUDA_VISIBLE_DEVICES=$GPUS eval "${BASE_CMD[@]}" \
            --model_config.ray_encoding plucker \
            --model_config.pos_enc none \
            --output_dir "results/${NAME}-plucker-none"
        exit 0
        ;;
    plucker-prope)
        CUDA_VISIBLE_DEVICES=$GPUS eval "${BASE_CMD[@]}" \
            --model_config.ray_encoding plucker \
            --model_config.pos_enc prope \
            --output_dir "results/${NAME}-plucker-prope"
        exit 0
        ;;
    plucker-gta)
        CUDA_VISIBLE_DEVICES=$GPUS eval "${BASE_CMD[@]}" \
            --model_config.ray_encoding plucker \
            --model_config.pos_enc gta \
            --output_dir "results/${NAME}-plucker-gta"
        exit 0
        ;;
    *)
        echo "Invalid mode: $MODE"
        exit 1
        ;;
esac