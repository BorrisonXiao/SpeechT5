#!/usr/bin/env bash

nvidia-smi
nvcc --version

. $(conda info --base)/etc/profile.d/conda.sh && conda deactivate && conda activate mult5
export PYTHONPATH=$PYTHONPATH:$PWD/fairseq
# Forces all CUDA operations to execute in order and completely finish before moving forward.
export CUDA_LAUNCH_BLOCKING=1
# Catch device-side assertions and errors
export TORCH_USE_CUDA_DSA=1

# Debug silent hangs
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=nccl_debug.log  # Log file for NCCL debug output
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Provides granular communication debugging
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1     # Ensures errors propagate immediately

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

data_dir=data
expdir=exp
spm_model=models/spm_char.model
pretrain_model=exp/pretrain/v3/checkpoint_2_200000.pt

stage=1
stop_stage=1

nshard=1
split=train
lab_dir=${data_dir}/asr

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Run the ASR fine-tuning script..."
    JOBID=$(date +%Y%m%d%H%M%S)
    # JOBID=debug
    DATA_ROOT=${lab_dir}
    SAVE_DIR=${expdir}/asr/${JOBID}
    LABEL_DIR=${lab_dir}
    # TRAIN_SET="speech_train"
    TRAIN_SET="speech_valid"
    VALID_SET="speech_valid"
    PT_CHECKPOINT_PATH=${pretrain_model}

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
        --fp16-scale-tolerance=0.2 \
        --fp16-init-scale 32 \
        --gradient-checkpointing \
        \
        --task speecht5 \
        --t5-task s2t \
        --sample-rate 16000 \
        --sync-matrix-len 512 \
        --num-workers 0 \
        --max-tokens 4000000 \
        --update-freq 1 \
        --bpe-tokenizer ${spm_model} \
        \
        --criterion speecht5 \
        --report-accuracy \
        --zero-infinity \
        --ce-weight 0.5 \
        --ctc-weight 0.5 \
        --sentence-avg \
        \
        --optimizer adam \
        --adam-betas "(0.9, 0.98)" \
        --adam-eps 1e-08 \
        --weight-decay 0.1 \
        --clip-norm 25.0 \
        --lr 0.0001 \
        --lr-scheduler tri_stage \
        --phase-ratio "[0.1, 0.3, 0.6]" \
        --final-lr-scale 0.05 \
        \
        --max-update 80000 \
        --max-text-positions 600 \
        --required-batch-size-multiple 1 \
        --save-interval-updates 2000 \
        --log-interval 10 \
        --skip-invalid-size-inputs-valid-test \
        \
        --arch t5_transformer_base_asr \
        --encoder-speech-prenet mel \
        --share-input-output-embed \
        --find-unused-parameters \
        --bert-init \
        --relative-position-embedding \
        --freeze-encoder-updates 100 \
        \
        --keep-last-epochs 4 \
        --feature-grad-mult 1.0 \
        --best-checkpoint-metric s2t_accuracy \
        --maximize-best-checkpoint-metric \
        --clear-cache-threshold 35840 \
        --finetune-from-model ${PT_CHECKPOINT_PATH} \
        --load-checkpoint-on-all-dp-ranks
fi

# fairseq-train ${DATA_ROOT} \
    #     --save-dir ${SAVE_DIR} \
    #     --tensorboard-logdir ${SAVE_DIR} \
    #     --train-subset ${TRAIN_SET} \
    #     --valid-subset ${VALID_SET} \
    #     --hubert-label-dir ${LABEL_DIR} \
    #     --distributed-world-size 4 \
    #     --distributed-port 0 \
    #     --ddp-backend pytorch_ddp \
    #     --user-dir speecht5 \
    #     --log-format simple \
    #     --seed 1337 \
    #     --fp16 \
    #     --fp16-scale-tolerance=0.25 \
    #     --gradient-checkpointing \
    #     \
    #     --task speecht5 \
    #     --t5-task pretrain \
    #     --label-rates 50 \
    #     --sample-rate 16000 \
    #     --random-crop \
    #     \
    #     --num-workers 0 \
    #     --max-tokens 5000000 \
    #     --encoder-seq-len 999 \
    #     --batch-size 8 \
    #     --batch-size-valid 12 \
    #     --max-speech-sample-size 320000 \
    #     --mel-hop-scale 2 \
    #     --pad-audio \
    #     --update-freq 2 \
    #     --batch-ratio "[1,0.0032]" \
    #     \
    #     --criterion speecht5 \
    #     --optimizer adam \
    #     --reset-optimizer \
    #     --adam-betas "(0.9, 0.98)" \
    #     --adam-eps 1e-06 \
    #     --weight-decay 0.01 \
    #     --power 1 \
    #     --clip-norm 5.0 \
    #     --lr 0.0002 \
    #     --lr-scheduler polynomial_decay \
    #     \
    #     --max-update 100000 \
    #     --warmup-updates 10000 \
    #     --total-num-update 100000 \
    #     --save-interval-updates 1000 \
    #     --log-interval 20 \
    #     --skip-invalid-size-inputs-valid-test \
    #     --required-batch-size-multiple 1 \
    #     --keep-last-epochs 4 \
    #     \
    #     --arch t5_transformer_base \
    #     --encoder-speech-prenet mel \
    #     --encoder-layers 8 \
    #     --speech-prenet-encoder-layers 10 \
    #     --share-input-output-embed \
    #     --find-unused-parameters \
    #     --bert-init \
    #     --relative-position-embedding \
    #     --use-codebook \
    #     --codebook-prob 0.2 \
    #     --loss-weights="[10,0.1]" \
    #     --max-text-positions 999 \
    #     --clear-cache-threshold 20480

        # --pad-audio-with-max \
# --no-reshard-after-forward \

# print(torch.cuda.memory_summary())

# import gc
# import torch

# total_mem = 0.0
# for obj in gc.get_objects():
#     try:
#         # if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 10000000:
#         # if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 10000:
#         if torch.is_tensor(obj) and obj.is_cuda:
#             size_mb = obj.numel() * obj.element_size() / 1e6
#             total_mem += size_mb
#             print(f"Large Tensor: {obj.shape}, dtype={obj.dtype}, size={size_mb:.2f} MB, requires_grad: {obj.requires_grad if hasattr(obj, 'requires_grad') else 'N/A'}")
#     except:
#         pass

# print(f"Total memory of large tensors: {total_mem:.2f} MB")