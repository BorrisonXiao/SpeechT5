#!/usr/bin/env bash

set -eou pipefail

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

data_dir=data/libriTTS
org_data_dir=/home/ec2-user/data/raw/LibriTTS
xvector_dir=data/xvectors.zip
dict_path=data/text/bins/dict.txt

stage=0
stop_stage=2

# train_sets="train-clean-100 train-clean-360 train-other-500"
train_sets="train-clean-100"
dev_sets="dev-clean dev-other"
test_sets="test-clean test-other"
fs=16000

tsv_dir=${data_dir}/tsv

conda deactivate && conda deactivate
. $(conda info --base)/etc/profile.d/conda.sh && conda deactivate && conda activate mult5

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Prepare features..."
    mkdir -p ${tsv_dir}/raw/valid
    mkdir -p ${tsv_dir}/raw/train
    mkdir -p ${tsv_dir}/raw/test

    set -x
    # Process the valid set
    mkdir -p ${tsv_dir}/raw/valid
    python scripts/resample_wavs.py -i ${org_data_dir} -o ${tsv_dir}/raw/valid --splits ${dev_sets}
    python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/valid --dest ${tsv_dir}/valid --ext wav --valid-percent 0
    ln -sfv ${xvector_dir} ${tsv_dir}/raw/valid
    # Rename the file
    cp ${tsv_dir}/valid/train.tsv ${tsv_dir}/speech_valid.tsv

    # Process the test set
    for split in ${test_sets}; do
        # Create a proxy directory for the split
        mkdir -p ${tsv_dir}/raw/test/${split}
        python scripts/resample_wavs.py -i ${org_data_dir} -o ${tsv_dir}/raw/test/${split} --splits ${split}
        python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/test/${split} --dest ${tsv_dir}/test/${split} --ext wav --valid-percent 0
        # Rename the file
        cp ${tsv_dir}/test/${split}/train.tsv ${tsv_dir}/${split}.tsv
        ln -sfv ${xvector_dir} ${tsv_dir}/raw/test/${split}
    done

    # Process the training set
    python scripts/resample_wavs.py -i ${org_data_dir} -o ${tsv_dir}/raw/train --splits ${train_sets}
    python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/train --dest ${tsv_dir}/train --ext wav --valid-percent 0
    # Rename the file for training
    cp ${tsv_dir}/train/train.tsv ${tsv_dir}/speech_train.tsv
    ln -sfv ${xvector_dir} ${tsv_dir}/raw/train

    # Add speaker embedding to the last column
    for split in "speech_train" "speech_valid" ${test_sets}; do
        python scripts/integrate_spkembs_tts.py \
            -i ${tsv_dir}/${split}.tsv \
            --dset librispeech \
            --xvectors ${xvector_dir} \
            --delimiter "_" \
            -o ${tsv_dir}/${split}_spk.tsv
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Prepare the TTS data for fine-tuning..."
    for split in "speech_train" "speech_valid"  ${test_sets}; do
        # Generate the word-level labels
        python scripts/libri_tts_labels.py ${tsv_dir}/${split}.tsv --output-dir ${tsv_dir} --output-name ${split}
        # Lowercase the labels due to the pre-trained tokenizer
        # scripts/lowercase_text.py \
        #     -i ${tsv_dir}/${split}.wrd \
        #     -o ${tsv_dir}/${split}.lc.wrd
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Finalize the TTS data folder..."

    tts_data_dir=${data_dir}/tts
    mkdir -p ${tts_data_dir}

    # Link the TTS data
    cp ${dict_path} ${PWD}/${tts_data_dir}
    ln -sfv ${PWD}/${tsv_dir}/speech_valid_spk.tsv ${PWD}/${tts_data_dir}/speech_valid.tsv
    ln -sfv ${PWD}/${tsv_dir}/speech_train_spk.tsv ${PWD}/${tts_data_dir}/speech_train.tsv
    ln -sfv ${PWD}/${tsv_dir}/speech_valid.wrd ${PWD}/${tts_data_dir}/speech_valid.txt
    ln -sfv ${PWD}/${tsv_dir}/speech_train.wrd ${PWD}/${tts_data_dir}/speech_train.txt
    for split in ${test_sets}; do
        ln -sfv ${PWD}/${tsv_dir}/${split}.wrd ${PWD}/${tts_data_dir}/${split}.txt
        ln -sfv ${PWD}/${tsv_dir}/${split}_spk.tsv ${PWD}/${tts_data_dir}/${split}.tsv
    done
fi
