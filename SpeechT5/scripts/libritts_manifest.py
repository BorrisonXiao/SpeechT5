import argparse
import os
from typing import Tuple

from scipy.io import wavfile
from torchaudio.datasets import LIBRITTS
from tqdm import tqdm


def load_libritts_item(
    fileid: str,
    path: str,
    ext_audio: str,
    ext_original_txt: str,
    ext_normalized_txt: str,
) -> Tuple[int, int, str, str, int, int, str]:
    speaker_id, chapter_id, segment_id, utterance_id = fileid.split("_")
    utterance_id = fileid

    normalized_text = utterance_id + ext_normalized_txt
    normalized_text = os.path.join(path, speaker_id, chapter_id, normalized_text)

    original_text = utterance_id + ext_original_txt
    original_text = os.path.join(path, speaker_id, chapter_id, original_text)

    file_audio = utterance_id + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
    sample_rate, wav = wavfile.read(file_audio)
    n_frames = wav.shape[0]

    # Load original text
    # with open(original_text) as ft:
    #     original_text = ft.readline()

    # Load normalized text
    with open(normalized_text, "r") as ft:
        normalized_text = ft.readline()

    return (
        n_frames,
        sample_rate,
        None,
        normalized_text,
        int(speaker_id),
        int(chapter_id),
        utterance_id,
    )


class LIBRITTS_16K(LIBRITTS):
    def __getitem__(self, n: int) -> Tuple[int, int, str, str, int, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, str, str, int, int, str):
            ``(waveform_length, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return load_libritts_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_original_txt,
            self._ext_normalized_txt,
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing wav files to index"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--split", required=True, type=str, help="dataset splits"
    )
    parser.add_argument(
        "--wav-root", default=None, type=str, metavar="DIR", help="saved waveform root directory for tsv"
    )
    parser.add_argument(
        "--spkemb-npy-dir", required=True, type=str, help="speaker embedding directory"
    )
    return parser

def main(args):
    dest_dir = args.dest
    wav_root = args.wav_root
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    dataset = LIBRITTS_16K(os.path.dirname(args.root), url=args.split, folder_in_archive=os.path.basename(args.root))
    tsv_f = open(os.path.join(dest_dir, f"{args.split}.tsv"), "w")
    txt_f = open(os.path.join(dest_dir, f"{args.split}.txt"), "w")
    print(wav_root, file=tsv_f)

    for n_frames, sr, ori_text, norm_text, spk_id, chap_id, utt_id in tqdm(dataset, desc="tsv/txt/wav"):
        assert sr == 16000, f"sampling rate {sr} != 16000"
        utt_file = os.path.join(args.split, f"{spk_id}", f"{chap_id}", f"{utt_id}.wav")
        spk_file = os.path.join(args.spkemb_npy_dir, f"{spk_id}-{chap_id}-{utt_id}.npy")
        assert os.path.exists(os.path.join(wav_root, utt_file))
        assert os.path.exists(os.path.join(wav_root, spk_file))

        print(f"{utt_file}\t{n_frames}\t{spk_file}", file=tsv_f)
        print(norm_text, file=txt_f)

    tsv_f.close()
    txt_f.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)