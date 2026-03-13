#!/bin/bash
# bash scripts/make_tsv_txt.sh /home/ec2-user/data/raw/LibriTTS_16k /home/ec2-user/mult5/SpeechT5/data/libriTTS_16k /home/ec2-user/data/raw/LibriTTS_16k /home/ec2-user/mult5/SpeechT5/data/libriTTS/spkrec-xvect
root=$1
dest=$2
wav_root=$3
spkemb_split=$4
if [ -z ${spkemb_split} ]; then
    spkemb_split=spkrec-xvect
fi
for split in dev-clean test-clean train-clean-100 train-clean-360; do
    echo "making ${split}.tsv and ${split}.txt ..."
    python scripts/libritts_manifest.py ${root} --dest ${dest} --split ${split} --wav-root ${wav_root} --spkemb-npy-dir ${spkemb_split}
done
