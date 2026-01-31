#!/usr/bin/env bash

nvidia-smi
nvcc --version

. $(conda info --base)/etc/profile.d/conda.sh && conda deactivate && conda activate mult5
export PYTHONPATH=$PYTHONPATH:$PWD/fairseq
# Forces all CUDA operations to execute in order and completely finish before moving forward.
export CUDA_LAUNCH_BLOCKING=1
# Catch device-side assertions and errors
export TORCH_USE_CUDA_DSA=1

# TQDM-related
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=5  # Force more frequent updates

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

export CUDA_VISIBLE_DEVICES=7

set -eou pipefail

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

data_dir=data
spm_model=models/spm_char.model
expdir=exp

lab_dir=${data_dir}/asr

eval_script=scripts/wer.py

# CHECKPOINT_PATH=exp/asr/v1.4/checkpoint_best.pt
CHECKPOINT_PATH=exp/asr/debug/checkpoint_1_2000.pt
tag=debug
# CHECKPOINT_PATH=exp/asr/v1.1/checkpoint_2_5000.pt
# CHECKPOINT_PATH=exp/asr/v1.3/checkpoint_5_16000.pt
DATA_ROOT=${lab_dir}
# SUBSETS="dev_clean dev_other test-clean test-other"  # List of subsets
# SUBSETS="test-clean test-other" # List of subsets
# SUBSETS="test-clean" # List of subsets
SUBSETS="speech_train" # List of subsets
BPE_TOKENIZER=$spm_model
LABEL_DIR=$DATA_ROOT
USER_DIR=speecht5
BEAM=1 #10
MAX_TOKENS=4000000
BATCH_SIZE=1
CTC_WEIGHT=0
LM_WEIGHT=0

. scripts/parse_options.sh || exit 1;

# Loop over all subsets
for SUBSET in $SUBSETS; do
    echo "Processing subset: ${SUBSET}"

    if [ "$CTC_WEIGHT" != "0" ]; then
        SAVE_DIR=${expdir}/inference_asr/att_ctc_${CTC_WEIGHT}/${tag}
        mkdir -p ${SAVE_DIR}
        MAX_TOKENS=
        echo "max tokens: ${MAX_TOKENS}"
        echo "batch size: ${BATCH_SIZE}"
        fairseq-generate ${DATA_ROOT} \
            --gen-subset ${SUBSET} \
            --bpe-tokenizer ${BPE_TOKENIZER} \
            --user-dir ${USER_DIR} \
            --task speecht5 \
            --t5-task s2t \
            --model-parallel-size 1 \
            --path ${CHECKPOINT_PATH} \
            --hubert-label-dir ${LABEL_DIR} \
            --ctc-weight ${CTC_WEIGHT} \
            --lm-weight ${LM_WEIGHT} \
            --batch-size ${BATCH_SIZE} \
            --beam ${BEAM} \
            --scoring wer \
            --max-len-a 0 \
            --max-len-b 620 \
            --sample-rate 16000 \
            --results-path ${SAVE_DIR}/${SUBSET}_BEAM_${BEAM}.out # \
        # --num-workers 2 >${SAVE_DIR}/${SUBSET}.log
    else
        # Run fairseq-generate and save only the last line of output
        SAVE_DIR=${expdir}/inference_asr/att/${tag}

        mkdir -p ${SAVE_DIR}
        # python -m pdb $(which fairseq-generate) ${DATA_ROOT} \
        fairseq-generate ${DATA_ROOT} \
            --gen-subset ${SUBSET} \
            --bpe-tokenizer ${BPE_TOKENIZER} \
            --user-dir ${USER_DIR} \
            --task speecht5 \
            --t5-task s2t \
            --model-parallel-size 1 \
            --path ${CHECKPOINT_PATH} \
            --hubert-label-dir ${LABEL_DIR} \
            --lm-weight ${LM_WEIGHT} \
            --beam ${BEAM} \
            --max-tokens ${MAX_TOKENS} \
            --batch-size ${BATCH_SIZE} \
            --scoring wer \
            --log-interval 1 \
            --max-len-a 0 \
            --max-len-b 620 \
            --sample-rate 16000 \
            --results-path ${SAVE_DIR}/${SUBSET}_BEAM_${BEAM}.out \
            --num-workers 0

        echo "Finished processing subset: ${SUBSET}. Last line saved to ${SAVE_DIR}/${SUBSET}_BEAM_${BEAM}.log"
    fi

    if [ -n "$eval_script" ]; then
        # Run the evaluation script
        python $eval_script -i ${SAVE_DIR}/${SUBSET}_BEAM_${BEAM}.out/generate-${SUBSET}.txt \
            -t ${DATA_ROOT}/${SUBSET}.txt
        echo "Evaluation results saved to ${SAVE_DIR}/${SUBSET}_BEAM_${BEAM}.eval"
    fi
done

# --mel-hop-scale 2 \
