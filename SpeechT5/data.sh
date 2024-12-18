#!/usr/bin/env bash

set -eou pipefail

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

data_dir=data
ckpt_dir=models
n_cluster=500 # Default is 500 from the SpeechT5 paper
km_path=${data_dir}/kmeans_model.pt
lm_data_dir=${data_dir}/raw/librispeech-lm-corpus
spm_model=${ckpt_dir}/spm_char.model
train_split=0.95
org_data_dir=/export/corpora5/LibriSpeech

stage=6
stop_stage=6

train_sets="train-clean-100 train-clean-360 train-other-500"
dev_sets="dev-clean dev-other"

tsv_dir=${data_dir}/tsv
feat_dir=${data_dir}/hubert_features
text_dir=${data_dir}/text

nshard=1
lab_dir=${data_dir}/hubert_km_labels
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    ckpt_path=${ckpt_dir}/hubert_base_ls960.pt
    layer=6
    rank=0
    log "Stage 0: Prepare HuBERT features using the base model and layer ${layer}..."
    mkdir -p ${feat_dir}
    mkdir -p ${tsv_dir}/raw/valid
    for split in ${dev_sets}; do
        # Create a proxy directory for the split
        ln -sfv ${org_data_dir}/${split} ${tsv_dir}/raw/valid/${split}
    done
    python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/valid --dest ${tsv_dir}/valid --ext flac --valid-percent 0
    # Rename the file for training
    cp ${tsv_dir}/valid/train.tsv ${tsv_dir}/speech_valid.tsv

    for split in ${train_sets}; do
        # Create a proxy directory for the split
        ln -sfv ${org_data_dir}/${split} ${tsv_dir}/raw/train/${split}
    done
    python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/train --dest ${tsv_dir}/train --ext flac --valid-percent 0
    # Rename the file for training
    cp ${tsv_dir}/train/train.tsv ${tsv_dir}/speech_train.tsv

    # Add speaker embedding to the last column
    python scripts/integrate_spkembs.py \
        -i ${tsv_dir}/speech_valid.tsv \
        --dset librispeech \
        --xvectors ${data_dir}/xvectors.zip \
        -o ${tsv_dir}/speech_valid_spk.tsv
    python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} speech_valid ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Fit a k-means model to the HuBERT features..."
    python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py ${feat_dir} speech_train ${nshard} ${km_path} ${n_cluster} --percent 0.1
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Assign cluster IDs to the HuBERT features..."
    rank=0
    for split in "speech_valid"; do
    #  for split in "speech_train speech_valid"; do
        python fairseq/examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
        for rank in $(seq 0 $((nshard - 1))); do
            cat $lab_dir/${split}_${rank}_${nshard}.km
        done >$lab_dir/${split}.km
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Prepare the text data..."
    mkdir -p ${text_dir}
    # Step 1: Dump the lowercased text data to a single txt file
    # Combine all .txt files recursively into a single file
    find ${lm_data_dir}/corpus -type f -name "*.txt" -exec cat {} + >${text_dir}/tmp.txt

    # Convert the combined file to lowercase
    tr '[:upper:]' '[:lower:]' <${text_dir}/tmp.txt >${text_dir}/tmp2.txt

    # Remove blank lines from the file
    grep -v '^$' ${text_dir}/tmp2.txt >${text_dir}/text

    # Optional: Remove the intermediate file if not needed
    rm ${text_dir}/tmp.txt
    rm ${text_dir}/tmp2.txt
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Prepare the text data..."
    # Perform train/dev split of the text data
    python scripts/train_dev_split.py \
        -i ${text_dir}/text \
        -o ${text_dir}

    # Tokenize the text data
    for split in "text_train" "text_valid"; do
        python /home/cxiao7/research/mult5/SpeechT5/SpeechT5/fairseq/scripts/spm_encode.py \
            --model ${spm_model} \
            --output_format=piece \
            --inputs ${text_dir}/${split}.txt \
            --outputs ${text_dir}/${split}.token
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Prepare the text data using faieseq..."
    DEST_DIR=${text_dir}/bins
    DICT=/home/cxiao7/research/mult5/SpeechT5/SpeechT5/models/dict.txt
    fairseq-preprocess \
        --only-source \
        --trainpref ${text_dir}/text_train.txt \
        --validpref ${text_dir}/text_valid.txt \
        --destdir ${DEST_DIR} \
        --srcdict ${DICT} \
        --workers 20
fi


if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Finalize the pretrain data folder..."
    
    pretrain_data_dir=${data_dir}/pretrain
    mkdir -p ${pretrain_data_dir}

    # Link the text and speech pretrain data
    ln -sfv ${PWD}/${text_dir}/bins/* ${PWD}/${pretrain_data_dir}
    ln -sfv ${PWD}/${tsv_dir}/speech_valid_spk.tsv ${PWD}/${pretrain_data_dir}/speech_valid.tsv
    ln -sfv ${PWD}/${tsv_dir}/speech_train.tsv ${PWD}/${pretrain_data_dir}/speech_train.tsv
fi