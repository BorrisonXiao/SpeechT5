import os
import glob
import numpy
import argparse
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchaudio.transforms as T

spk_model = {
    "speechbrain/spkrec-xvect-voxceleb": 512, 
    "speechbrain/spkrec-ecapa-voxceleb": 192,
}

def f2embed(wav_file, classifier, size_embed, resampler=None):
    signal, fs =torchaudio.load(wav_file)
    if fs != 16000 and fs is not None:
        assert fs == 24000, fs
        signal = resampler(signal)
        fs = 16000
    assert fs == 16000, fs
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    return embeddings

def process(args):
    wavlst = []
    for split in args.splits.split(","):
        wav_dir = os.path.join(args.libritts_root, split)
        wavlst_split = glob.glob(os.path.join(wav_dir, "*", "*", "*.wav"))
        print(f"{split} {len(wavlst_split)} utterances.")
        wavlst.extend(wavlst_split)
    spkemb_root = args.output_root
    if not os.path.exists(spkemb_root):
        print(f"Create speaker embedding directory: {spkemb_root}")
        os.mkdir(spkemb_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=args.speaker_embed, run_opts={"device": device}, savedir='/tmp')
    size_embed = spk_model[args.speaker_embed]
    resampler = T.Resample(24000, 16000)
    for utt_i in tqdm(wavlst, total=len(wavlst), desc="Extract"):
        utt_id = "-".join(utt_i.split("/")[-3:]).replace(".wav", "")
        utt_emb = f2embed(utt_i, classifier, size_embed, resampler)
        numpy.save(os.path.join(spkemb_root, f"{utt_id}.npy"), utt_emb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libritts-root", "-i", required=True, type=str, help="LibriTTS root directory.")
    parser.add_argument("--output-root", "-o", required=True, type=str, help="Output directory.")
    parser.add_argument("--speaker-embed", "-s", type=str, required=True, choices=["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb"],
                        help="Pretrained model for extracting speaker emebdding.")
    parser.add_argument("--splits", default="train-clean-100,train-clean-360,dev-clean,test-clean", type=str,
                        help="Split of train,dev,test seperate by comma.")
    args = parser.parse_args()
    print(f"Loading utterances from {args.libritts_root}/{args.splits}, "
        + f"Save speaker embedding 'npy' to {args.output_root}, "
        + f"Using speaker model {args.speaker_embed} with {spk_model[args.speaker_embed]} size.")
    process(args)

if __name__ == "__main__":
    """
    python scripts/prep_libritts_spkemb.py \
        -i /home/ec2-user/data/raw/LibriTTS \
        -o /home/ec2-user/mult5/SpeechT5/data/libriTTS/spkrec-xvect \
        -s speechbrain/spkrec-xvect-voxceleb
    """
    main()