#!/usr/bin/env bash

set -eou pipefail

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

data_dir=data/mtl-asr-tts
asr_data_dir=/home/ec2-user/mult5/SpeechT5/data/ASR
tts_data_dir=/home/ec2-user/mult5/SpeechT5/data/libriTTS
dict_path=data/ASR/asr/dict.txt

stage=1
stop_stage=2

asr_train_set="train-clean-100"
tts_train_set="train-clean-norm-460"
asr_valid_set="dev-clean"
tts_valid_set="dev-clean-norm"

tgt_asr_train_set_name="asr-train-clean-100"
tgt_tts_train_set_name="tts-train-clean-norm-460"
tgt_asr_valid_set_name="asr-dev-clean"
tgt_tts_valid_set_name="tts-dev-clean-norm"

asr_tsv_dir=${asr_data_dir}/asr
tts_tsv_dir=${tts_data_dir}/tts
tgt_dir=${data_dir}/asr-tts

. $(conda info --base)/etc/profile.d/conda.sh && conda deactivate && conda activate mult5

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Preparing training directories..."
    mkdir -p ${tgt_dir}

    # Simply copy over the ASR valid set
    ln -sfv ${asr_tsv_dir}/${asr_train_set}.tsv ${tgt_dir}/${tgt_asr_train_set_name}.tsv
    ln -sfv ${asr_tsv_dir}/${asr_train_set}.txt ${tgt_dir}/${tgt_asr_train_set_name}.txt
    # Simply copy over the TTS valid set
    ln -sfv ${tts_tsv_dir}/${tts_train_set}.tsv ${tgt_dir}/${tgt_tts_train_set_name}.tsv
    ln -sfv ${tts_tsv_dir}/${tts_train_set}.txt ${tgt_dir}/${tgt_tts_train_set_name}.txt

    cp ${dict_path} ${tgt_dir}/dict.txt
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Preparing validation directories..."

    # Simply copy over the ASR valid set
    ln -sfv ${asr_tsv_dir}/${asr_valid_set}.tsv ${tgt_dir}/${tgt_asr_valid_set_name}.tsv
    ln -sfv ${asr_tsv_dir}/${asr_valid_set}.txt ${tgt_dir}/${tgt_asr_valid_set_name}.txt
    # Simply copy over the TTS valid set
    ln -sfv ${tts_tsv_dir}/${tts_valid_set}.tsv ${tgt_dir}/${tgt_tts_valid_set_name}.tsv
    ln -sfv ${tts_tsv_dir}/${tts_valid_set}.txt ${tgt_dir}/${tgt_tts_valid_set_name}.txt
fi
