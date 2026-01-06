#!/usr/bin/env bash

nvidia-smi
nvcc --version

. $(conda info --base)/etc/profile.d/conda.sh && conda deactivate && conda activate mult5
export LD_LIBRARY_PATH=$HOME/research/discrete/espnet_meili/tools/miniconda/envs/mult5/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$PWD/fairseq
# Forces all CUDA operations to execute in order and completely finish before moving forward.
export CUDA_LAUNCH_BLOCKING=1
# Catch device-side assertions and errors
export TORCH_USE_CUDA_DSA=1

# Debug silent hangs
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=nccl_debug.log    # Log file for NCCL debug output
export TORCH_DISTRIBUTED_DEBUG=DETAIL    # Provides granular communication debugging
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # Ensures errors propagate immediately

# To avoid fragmentation issues, turns out doesn't help
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# To avoid fragmentation issues based on the advice of the PyTorch team
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disabling CUDA caching for debugging, in fact this seems to reduce the memory overhead significantly but
# at the cost of speed (which turns out to be significant as well)
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8

# The following two seem to help with hangs
# export NCCL_IB_DISABLE=1  # Disable InfiniBand if not used
# export NCCL_P2P_LEVEL=SYS
# export NCCL_SOCKET_IFNAME=eth0  # Use the correct network interface

# Disable P2P to avoid hangs (doesn't quite work though)
# export NCCL_P2P_DISABLE=1

# export CUDA_VISIBLE_DEVICES=0,1

set -eou pipefail

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

data_dir=data/mtl-asr-tts
expdir=exp
spm_model=/home/cxiao7/research/mult5/SpeechT5/SpeechT5/models/spm_char.model
pretrain_model=data/downloads/p4_v2/pretrain/exp/checkpoint_best.pt

stage=1
stop_stage=1

lab_dir=${data_dir}/asr-tts

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Run the MTL fine-tuning script..."
    JOBID=v0-$(date +%Y%m%d%H%M%S)
    JOBID=debug
    DATA_ROOT=${lab_dir}
    SAVE_DIR=${expdir}/mtl-asr-tts/${JOBID}
    LABEL_DIR=${lab_dir}
    TRAIN_SET="asr-train-clean-100|tts-train-clean-460"
    VALID_SET="asr-dev-clean|tts-dev"
    PT_CHECKPOINT_PATH=${pretrain_model}

    mkdir -p ${SAVE_DIR}

    fairseq-train ${DATA_ROOT} \
        --save-dir ${SAVE_DIR} \
        --tensorboard-logdir ${SAVE_DIR} \
        --train-subset ${TRAIN_SET} \
        --valid-subset ${VALID_SET} \
        --hubert-label-dir ${LABEL_DIR} \
        --distributed-world-size 1 \
        --distributed-port 0 \
        --ddp-backend pytorch_ddp \
        --user-dir speecht5 \
        --log-format simple \
        --seed 1 \
        --fp16 \
        --fp16-scale-tolerance=0.05 \
        --gradient-checkpointing \
        \
        --task speecht5 \
        --t5-task mtl-asr-tts \
        --sample-rate 16000 \
        --sync-matrix-len 512 \
        --num-workers 0 \
        --max-tokens 1600000 \
        --update-freq 2 \
        --bpe-tokenizer ${spm_model} \
        --max-tokens-valid 1600000 \
        \
        --criterion speecht5 \
        --report-accuracy \
        --sentence-avg \
        \
        --optimizer adam \
        --reset-optimizer \
        --reset-dataloader \
        --adam-betas "(0.9, 0.98)" \
        --dropout 0.15 \
        --activation-dropout 0.15 \
        --attention-dropout 0.15 \
        --encoder-layerdrop 0.0 \
        --decoder-layerdrop 0.0 \
        --weight-decay 0.0 \
        --clip-norm 25.0 \
        --lr 0.0001 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 500 \
        --feature-grad-mult 1.0 \
        \
        --max-update 160000 \
        --max-text-positions 600 \
        --min-speech-sample-size 1056 \
        --max-speech-sample-size 480256 \
        --max-speech-positions 1876 \
        --required-batch-size-multiple 1 \
        --validate-after-updates 8000 \
        --skip-invalid-size-inputs-valid-test \
        --validate-interval 50 \
        --save-interval-updates 4000 \
        --log-interval 10 \
        \
        --arch t5_transformer_base_asr \
        --share-input-output-embed \
        --find-unused-parameters \
        --bert-init \
        --relative-position-embedding \
        --freeze-encoder-updates 100 \
        \
        --keep-last-epochs 2 \
        --restore-file ${PT_CHECKPOINT_PATH} \
        --clear-cache-threshold 36000 \
        --load-checkpoint-on-all-dp-ranks
fi
        # --finetune-from-model ${PT_CHECKPOINT_PATH} \