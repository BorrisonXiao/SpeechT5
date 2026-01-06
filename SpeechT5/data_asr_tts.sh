#!/usr/bin/env bash

set -eou pipefail

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

data_dir=data/mtl-asr-tts
asr_data_dir=data/ASR
tts_data_dir=data/libriTTS
dict_path=data/ASR/asr/dict.txt

stage=1
stop_stage=2

asr_train_set="train-clean-100"
tts_train_set="train-clean-460"
asr_valid_set="dev-clean"
tts_valid_set="dev"

tgt_asr_train_set_name="asr-train-clean-100"
tgt_tts_train_set_name="tts-train-clean-460"
tgt_asr_valid_set_name="asr-dev-clean"
tgt_tts_valid_set_name="tts-dev"

asr_tsv_dir=${asr_data_dir}/tsv
tts_tsv_dir=${tts_data_dir}/tsv
tgt_dir=${data_dir}/asr-tts

. $(conda info --base)/etc/profile.d/conda.sh && conda deactivate && conda activate mult5

# if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
#     log "Stage 1: Preparing training directories..."
#     mkdir -p ${tsv_dir}/raw/train

#     # Build the train_tsvs string
#     asr_sets_array=($asr_train_sets)
#     tts_sets_array=($tts_train_sets)
#     train_tsvs=""
#     train_txts=""
#     for split in "${asr_sets_array[@]}"; do
#         train_tsvs+="${asr_tsv_dir}/${split}_spk.tsv "
#         train_txts+="${asr_tsv_dir}/${split}.lc.wrd "
#     done
#     for split in "${tts_sets_array[@]}"; do
#         train_tsvs+="${tts_tsv_dir}/${split}_spk.tsv "
#         train_txts+="${tts_tsv_dir}/${split}.wrd "
#     done
#     # Remove the trailing space
#     train_tsvs="${train_tsvs% }"
#     train_txts="${train_txts% }"

#     scripts/join_tsvs.py \
#         -i ${train_tsvs} \
#         -t ${train_txts} \
#         --raw-dir ${PWD}/${tsv_dir}/raw/train \
#         -o ${tgt_dir}/${tgt_train_set_name}

#     cp ${dict_path} ${tgt_dir}/dict.txt
#     cp ${xvector_dir} ${PWD}/${tsv_dir}/raw/train
# fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Preparing training directories..."
    mkdir -p ${tgt_dir}

    # Simply copy over the ASR valid set
    cp ${asr_tsv_dir}/${asr_train_set}_spk.tsv ${tgt_dir}/${tgt_asr_train_set_name}.tsv
    cp ${asr_tsv_dir}/${asr_train_set}.lc.wrd ${tgt_dir}/${tgt_asr_train_set_name}.txt
    # Simply copy over the TTS valid set
    cp ${tts_tsv_dir}/${tts_train_set}_spk.tsv ${tgt_dir}/${tgt_tts_train_set_name}.tsv
    cp ${tts_tsv_dir}/${tts_train_set}.wrd ${tgt_dir}/${tgt_tts_train_set_name}.txt

    cp ${dict_path} ${tgt_dir}/dict.txt
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Preparing validation directories..."

    # Simply copy over the ASR valid set
    cp ${asr_tsv_dir}/${asr_valid_set}_spk.tsv ${tgt_dir}/${tgt_asr_valid_set_name}.tsv
    cp ${asr_tsv_dir}/${asr_valid_set}.lc.wrd ${tgt_dir}/${tgt_asr_valid_set_name}.txt
    # Simply copy over the TTS valid set
    cp ${tts_tsv_dir}/${tts_valid_set}_spk.tsv ${tgt_dir}/${tgt_tts_valid_set_name}.tsv
    cp ${tts_tsv_dir}/${tts_valid_set}.wrd ${tgt_dir}/${tgt_tts_valid_set_name}.txt
fi
