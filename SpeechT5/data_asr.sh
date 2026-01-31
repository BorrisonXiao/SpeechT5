#!/usr/bin/env bash

set -eou pipefail

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

data_dir=data
org_data_dir=/export/fs06/cxiao7/LibriSpeech
xvector_dir=/home/cxiao7/research/mult5/SpeechT5/SpeechT5/data/xvectors.zip
dict_path=data/downloads/dict.txt

stage=2
stop_stage=2

train_sets="train-clean-100"
dev_sets="dev-clean"
test_sets="test-clean test-other"

tgt_train_set_name="train-clean-100"
tgt_dev_set_name="dev-clean"

tsv_dir=${data_dir}/ASR/tsv

nshard=1
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Prepare tsv..."
    mkdir -p ${tsv_dir}/raw/valid
    mkdir -p ${tsv_dir}/raw/train
    mkdir -p ${tsv_dir}/raw/test

    # set -x
    # # Process the valid set
    # for split in ${dev_sets}; do
    #     # Create a proxy directory for the split
    #     ln -sfv ${org_data_dir}/${split} ${tsv_dir}/raw/valid
    # done
    # cp ${xvector_dir} ${tsv_dir}/raw/valid
    # python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/valid --dest ${tsv_dir}/valid --ext flac --valid-percent 0
    # # Rename the file for training
    # cp ${tsv_dir}/valid/train.tsv ${tsv_dir}/${tgt_dev_set_name}.tsv

    # # Process the test set
    # for split in ${test_sets}; do
    #     # Create a proxy directory for the split
    #     mkdir -p ${tsv_dir}/raw/test/${split}
    #     ln -sfv ${org_data_dir}/${split}/* ${tsv_dir}/raw/test/${split}
    #     python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/test/${split} --dest ${tsv_dir}/test/${split} --ext flac --valid-percent 0
    #     # Rename the file
    #     cp ${tsv_dir}/test/${split}/train.tsv ${tsv_dir}/${split}.tsv
    #     cp ${xvector_dir} ${tsv_dir}/raw/test/${split}
    # done

    # Process the training set
    for split in ${train_sets}; do
        # Create a proxy directory for the split
        ln -sfv ${org_data_dir}/${split} ${tsv_dir}/raw/train
    done
    python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/train --dest ${tsv_dir}/train --ext flac --valid-percent 0
    # Rename the file for training
    cp ${xvector_dir} ${tsv_dir}/raw/train
    cp ${tsv_dir}/train/train.tsv ${tsv_dir}/${tgt_train_set_name}.tsv

    # Add speaker embedding to the last column
    # for split in ${tgt_train_set_name} ${tgt_dev_set_name} ${test_sets}; do
    # for split in "${tgt_train_set_name}" ${test_sets}; do
    for split in ${tgt_train_set_name}; do
        python scripts/integrate_spkembs.py \
            -i ${tsv_dir}/${split}.tsv \
            --dset librispeech \
            --xvectors ${xvector_dir} \
            -o ${tsv_dir}/${split}_spk.tsv
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Prepare the ASR data for fine-tuning..."
    # for split in ${tgt_train_set_name} ${tgt_dev_set_name} ${test_sets}; do
    # for split in "speech_valid"; do
    # for split in ${test_sets}; do
    for split in ${tgt_train_set_name}; do
        # Generate the word-level labels
        python fairseq/examples/wav2vec/libri_labels.py ${tsv_dir}/${split}.tsv --output-dir ${tsv_dir} --output-name ${split}
        # Lowercase the labels due to the TTS data
        scripts/lowercase_text.py \
            -i ${tsv_dir}/${split}.wrd \
            -o ${tsv_dir}/${split}.lc.wrd
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Finalize the ASR data folder..."

    asr_data_dir=${data_dir}/ASR/asr
    mkdir -p ${asr_data_dir}

    # Link the text and speech pretrain data
    cp ${dict_path} ${PWD}/${asr_data_dir}/dict.txt
    ln -sfv ${PWD}/${tsv_dir}/${tgt_dev_set_name}_spk.tsv ${PWD}/${asr_data_dir}/${tgt_dev_set_name}.tsv
    ln -sfv ${PWD}/${tsv_dir}/${tgt_train_set_name}_spk.tsv ${PWD}/${asr_data_dir}/${tgt_train_set_name}.tsv
    ln -sfv ${PWD}/${tsv_dir}/${tgt_dev_set_name}.lc.wrd ${PWD}/${asr_data_dir}/${tgt_dev_set_name}.txt
    ln -sfv ${PWD}/${tsv_dir}/${tgt_train_set_name}.lc.wrd ${PWD}/${asr_data_dir}/${tgt_train_set_name}.txt
    for split in ${test_sets}; do
        ln -sfv ${PWD}/${tsv_dir}/${split}.lc.wrd ${PWD}/${asr_data_dir}/${split}.txt
        ln -sfv ${PWD}/${tsv_dir}/${split}.tsv ${PWD}/${asr_data_dir}/${split}.tsv
    done
fi
