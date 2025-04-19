#!/usr/bin/env bash
#
#SBATCH --job-name=pretrain
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --partition=a100
#SBATCH --account=lgarci27_gpu
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out

module load cuda/12.5.0
/bin/hostname
nvidia-smi
nvcc --version

. ~/.bashrc
conda activate $HOME/data_lgarci27/cxiao7/miniconda3/envs/mult5
export LD_LIBRARY_PATH=$HOME/data_lgarci27/cxiao7/miniconda3/envs/mult5/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$PWD/fairseq
# Forces all CUDA operations to execute in order and completely finish before moving forward.
export CUDA_LAUNCH_BLOCKING=1
# Catch device-side assertions and errors
export TORCH_USE_CUDA_DSA=1

# Debug silent hangs
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=nccl_debug.log  # Log file for NCCL debug output
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Provides granular communication debugging
export NCCL_ASYNC_ERROR_HANDLING=1     # Ensures errors propagate immediately

# The following two seem to help with hangs
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not used
export NCCL_SOCKET_IFNAME=eth0  # Use the correct network interface

# Disable P2P to avoid hangs (doesn't quite work though)
export NCCL_P2P_DISABLE=1

# export CUDA_VISIBLE_DEVICES=0,1

set -eou pipefail

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

data_dir=data
expdir=exp

stage=1
stop_stage=1

nshard=1
split=train
lab_dir=${data_dir}/hubert_km_labels

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Run the pre-training script..."
    # JOBID=$(date +%Y%m%d%H%M%S)
    JOBID=pretrain
    DATA_ROOT=${data_dir}/pretrain
    SAVE_DIR=${expdir}/pretrain/${JOBID}
    LABEL_DIR=${lab_dir}
    TRAIN_SET="speech_train|text_train"
    VALID_SET="speech_valid|text_valid"

    mkdir -p ${SAVE_DIR}

    fairseq-train ${DATA_ROOT} \
        --save-dir ${SAVE_DIR} \
        --tensorboard-logdir ${SAVE_DIR} \
        --train-subset ${TRAIN_SET} \
        --valid-subset ${VALID_SET} \
        --hubert-label-dir ${LABEL_DIR} \
        --distributed-world-size 4 \
        --distributed-port 0 \
        --ddp-backend pytorch_ddp \
        --user-dir speecht5 \
        --log-format simple \
        --seed 1337 \
        --fp16 \
        --fp16-scale-tolerance=0.25 \
        --gradient-checkpointing \
        \
        --task speecht5 \
        --t5-task pretrain \
        --label-rates 50 \
        --sample-rate 16000 \
        --random-crop \
        \
        --num-workers 0 \
        --max-tokens 5000000 \
        --encoder-seq-len 999 \
        --batch-size 16 \
        --batch-size-valid 24 \
        --max-speech-sample-size 320000 \
        --mel-hop-scale 2 \
        --pad-audio \
        --update-freq 2 \
        --batch-ratio "[1,0.0032]" \
        \
        --criterion speecht5 \
        --optimizer adam \
        --reset-optimizer \
        --adam-betas "(0.9, 0.98)" \
        --adam-eps 1e-06 \
        --weight-decay 0.01 \
        --power 1 \
        --clip-norm 5.0 \
        --lr 0.0002 \
        --lr-scheduler polynomial_decay \
        \
        --max-update 100000 \
        --warmup-updates 10000 \
        --total-num-update 100000 \
        --save-interval-updates 1000 \
        --log-interval 20 \
        --skip-invalid-size-inputs-valid-test \
        --required-batch-size-multiple 1 \
        --keep-last-epochs 4 \
        \
        --arch t5_transformer_base \
        --encoder-speech-prenet mel \
        --encoder-layers 8 \
        --speech-prenet-encoder-layers 10 \
        --share-input-output-embed \
        --find-unused-parameters \
        --bert-init \
        --relative-position-embedding \
        --use-codebook \
        --codebook-prob 0.2 \
        --loss-weights="[10,0.1]" \
        --max-text-positions 999
fi

        # --pad-audio-with-max \
# --no-reshard-after-forward \