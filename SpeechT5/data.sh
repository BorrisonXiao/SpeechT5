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
spm_dir=${ckpt_dir}/self_trained
vocab_size=3000
spm_model=${spm_dir}/spm_bpe_${vocab_size}.model
train_split=0.98
org_data_dir=/export/fs05/cxiao7/LibriSpeech
xvector_dir=/home/cxiao7/research/mult5/SpeechT5/SpeechT5/data/xvectors.zip
train_spm=true
dict_path=
max_token_len=1249

stage=4
stop_stage=4

train_sets="train-clean-100 train-clean-360 train-other-500"
dev_sets="dev-clean dev-other"

tsv_dir=${data_dir}/tsv
feat_dir=${data_dir}/debug_hubert_features
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
    mkdir -p ${tsv_dir}/raw/train

    # set -x
    # # Process the valid set
    # for split in ${dev_sets}; do
    #     # Create a proxy directory for the split
    #     ln -sfv ${org_data_dir}/${split} ${tsv_dir}/raw/valid
    # done
    # ln -sfv ${xvector_dir} ${tsv_dir}/raw/valid
    # python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/valid --dest ${tsv_dir}/valid --ext flac --valid-percent 0
    # # Rename the file for training
    # cp ${tsv_dir}/valid/train.tsv ${tsv_dir}/speech_valid.tsv

    # # Process the training set
    # for split in ${train_sets}; do
    #     # Create a proxy directory for the split
    #     ln -sfv ${org_data_dir}/${split} ${tsv_dir}/raw/train
    # done
    # ln -sfv ${org_data_dir} ${tsv_dir}/raw/train
    # python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/train --dest ${tsv_dir}/train --ext flac --valid-percent 0
    # # Rename the file for training
    # cp ${tsv_dir}/train/train.tsv ${tsv_dir}/speech_train.tsv

    # Add speaker embedding to the last column
    # for split in "speech_train" "speech_valid"; do
    for split in "speech_train"; do
        # python scripts/integrate_spkembs.py \
        #     -i ${tsv_dir}/${split}.tsv \
        #     --dset librispeech \
        #     --xvectors ${data_dir}/xvectors.zip \
        #     -o ${tsv_dir}/${split}_spk.tsv

        # Generate the hubert features
        python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Fit a k-means model to the HuBERT features..."
    python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py \
        ${feat_dir} \
        speech_train \
        ${nshard} \
        ${km_path} \
        ${n_cluster} \
        --percent 0.08
    # python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py data/hubert_features speech_train 1 data/kmeans_model.pt 500 --percent 0.05 --max_iter 1
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Assign cluster IDs to the HuBERT features..."
    rank=0
    set -x
    for split in "speech_train"; do
    # for split in "speech_train" "speech_valid"; do
        python fairseq/examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
        for rank in $(seq 0 $((nshard - 1))); do
            cat $lab_dir/${split}_${rank}_${nshard}.km
        done >$lab_dir/${split}.km
    done

    # Generate the dict.km.txt file for the k-means labels on the training split
    python scripts/generate_dict.py \
        -i ${lab_dir}/speech_train.km \
        -o ${lab_dir}/dict.km.txt
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
    # Perform train/dev split of the text data, also remove the empty lines
    # python scripts/train_dev_split.py \
    #     -i ${text_dir}/text \
    #     -o ${text_dir} \
    #     --merge-style "gaussian" \
    #     --sanitize \
    #     --dict ${ckpt_dir}/dict.txt \
    #     --ratio ${train_split} \
    #     --merge-params "400|50"

    # # # Train a SPM tokenizer on the text data if train_spm is set to true
    # if [ $train_spm = true ]; then
    #     mkdir -p ${spm_dir}
    #     python /home/cxiao7/research/mult5/SpeechT5/SpeechT5/fairseq/scripts/spm_train.py \
    #         --input=${text_dir}/text_train.txt \
    #         --model_prefix=${spm_model} \
    #         --train_extremely_large_corpus=true \
    #         --shuffle_input_sentence=true \
    #         --input_sentence_size=2000000 \
    #         --model_type=bpe \
    #         --vocab_size=${vocab_size}
    # fi

    # Tokenize the text data
    for split in "text_train" "text_valid"; do
        # python /home/cxiao7/research/mult5/SpeechT5/SpeechT5/fairseq/scripts/spm_encode.py \
        #     --model ${spm_model}.model \
        #     --output_format=piece \
        #     --inputs ${text_dir}/${split}.txt \
        #     --outputs ${text_dir}/${split}.token \
        #     --max-len ${max_token_len}

        # Plot the token length distribution
        stats_dir=${text_dir}/stats
        mkdir -p ${stats_dir}
        python scripts/plot_token_len.py \
            -i ${text_dir}/${split}.token \
            -o ${stats_dir}/${split}_token_len.png
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Prepare the text data using fairseq..."
    DEST_DIR=${text_dir}/bins
    # Set DICT to the specified dict path if it is not empty
    # Otherwise use the trained spm vocab file ${spm_model}.vocab
    if [ -z ${dict_path} ]; then
        cut -f1 ${spm_model}.vocab | tail -n +4 | sed "s/$/ 1/g" > ${spm_model}.fairseq.vocab
        DICT=${spm_model}.fairseq.vocab
    else
        DICT=${dict_path}
    fi
    fairseq-preprocess \
        --only-source \
        --trainpref ${text_dir}/text_train.token \
        --validpref ${text_dir}/text_valid.token \
        --destdir ${DEST_DIR} \
        --srcdict ${DICT} \
        --workers 20

    ln -sfv ${PWD}/${DEST_DIR}/train.bin ${DEST_DIR}/text_train.bin
    ln -sfv ${PWD}/${DEST_DIR}/valid.bin ${DEST_DIR}/text_valid.bin
    ln -sfv ${PWD}/${DEST_DIR}/train.idx ${DEST_DIR}/text_train.idx
    ln -sfv ${PWD}/${DEST_DIR}/valid.idx ${DEST_DIR}/text_valid.idx
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Finalize the pretrain data folder..."

    pretrain_data_dir=${data_dir}/pretrain
    mkdir -p ${pretrain_data_dir}

    # Link the text and speech pretrain data
    ln -sfv ${PWD}/${text_dir}/bins/text_train.* ${PWD}/${pretrain_data_dir}
    ln -sfv ${PWD}/${text_dir}/bins/text_valid.* ${PWD}/${pretrain_data_dir}
    ln -sfv ${PWD}/${tsv_dir}/speech_valid_spk.tsv ${PWD}/${pretrain_data_dir}/speech_valid.tsv
    ln -sfv ${PWD}/${tsv_dir}/speech_train_spk.tsv ${PWD}/${pretrain_data_dir}/speech_train.tsv
fi
