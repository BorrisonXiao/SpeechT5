#!/usr/bin/env python3
import os
import argparse
import soundfile as sf
from pathlib import Path
import math
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk long audio segments and update metadata + transcript files."
    )
    parser.add_argument(
        "-i", "--input", default="data/asr/test-clean.tsv",
        help="Path to input .tsv file"
    )
    parser.add_argument(
        "-t", "--transcript", default="data/asr/test-clean.txt",
        help="Path to transcript text file"
    )
    parser.add_argument(
        "-o", "--output-dir", default="data/asr",
        help="Output directory for new .tsv and .txt files",
    )
    parser.add_argument(
        "--fs", type=int, default=16000,
        help="Sampling rate (default: 16000)"
    )
    parser.add_argument(
        "--max-len", type=int, default=320000,
        help="Maximum number of samples per chunk",
    )
    parser.add_argument(
        "--wav-dir", default="data/asr/wav_chunks/test-clean",
        help="Output directory for new chunked wav files"
    )
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()

    # normalize to absolute paths for the output header
    args.wav_dir = os.path.abspath(args.wav_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.wav_dir, exist_ok=True)

    # Read input metadata TSV
    with open(args.input, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip() != ""]

    if not lines:
        raise ValueError("Input TSV is empty")

    orig_header = lines[0].strip()  # e.g., "LibriSpeech/test-clean"
    entries = [line.strip().split("\t") for line in lines[1:]]

    # Read transcripts (original, one-per-entry)
    with open(args.transcript, "r", encoding="utf-8") as f:
        transcripts = [line.rstrip("\n") for line in f.readlines()]

    if len(entries) != len(transcripts):
        raise ValueError(f"Mismatch: {len(entries)} entries vs {len(transcripts)} transcripts")

    # new header is the absolute path to where new audio files will be saved
    new_tsv_lines = [args.wav_dir]
    new_txt_lines = []

    dset_name = Path(args.input).stem
    out_tsv = os.path.join(args.output_dir, f"{dset_name}.chunked.tsv")
    out_txt = os.path.join(args.output_dir, f"{dset_name}.chunked.txt")

    print(f"Processing {len(entries)} audio files...")

    for idx, ((rel_path, length_str), text) in tqdm(enumerate(zip(entries, transcripts)), total=len(entries)):
        length_from_tsv = int(length_str)
        # initial num_chunks computed from tsv length (fast path; may be adjusted if we read audio)
        num_chunks = (length_from_tsv + args.max_len - 1) // args.max_len

        # Path to the original audio file (found by joining the original header + relative path)
        orig_audio_path = os.path.join(orig_header, rel_path)
        if not os.path.exists(orig_audio_path):
            raise FileNotFoundError(f"Missing audio file: {orig_audio_path}")

        # Determine the base new "line number" in the chunked transcript:
        # this is 1-indexed and equals the index (len(new_txt_lines)+1) where the first chunk's transcript
        # will be written for this utterance.
        base_new_line = len(new_txt_lines) + 1

        if num_chunks == 1:
            # No chunking required: symlink the original file into the new wav dir (avoid duplication)
            new_rel_path = rel_path  # keep directory structure under wav_dir
            dest_path = os.path.join(args.wav_dir, new_rel_path)
            ensure_parent_dir(dest_path)
            if not os.path.exists(dest_path):
                # create a symlink pointing to the original absolute path
                os.symlink(os.path.abspath(orig_audio_path), dest_path)

            new_tsv_lines.append(f"{new_rel_path}\t{length_from_tsv}")

            # The transcript for this (un-chunked) entry is the original text and occupies base_new_line
            new_txt_lines.append(text)
            continue

        # If we reach here: we expect multiple chunks; read the audio and write chunk files
        audio, sr = sf.read(orig_audio_path)
        if sr != args.fs:
            raise ValueError(f"Sample rate mismatch for {orig_audio_path}: expected {args.fs}, got {sr}")

        # number of frames (samples) in audio. audio may be 1-D or 2-D (frames x channels)
        total_samples = audio.shape[0] if hasattr(audio, "shape") else len(audio)
        # recompute number of chunks from actual audio length (safer)
        num_chunks = math.ceil(total_samples / args.max_len)

        base, ext = os.path.splitext(rel_path)
        for c in range(num_chunks):
            start = c * args.max_len
            end = min(start + args.max_len, total_samples)
            # slice frames; this works for 1D or 2D numpy arrays
            chunk_audio = audio[start:end]

            chunk_rel_path = f"{base}_{c:04d}{ext}"
            out_chunk_path = os.path.join(args.wav_dir, chunk_rel_path)
            ensure_parent_dir(out_chunk_path)
            sf.write(out_chunk_path, chunk_audio, sr)

            # length for tsv should be number of samples in chunk (frames)
            chunk_len = end - start
            new_tsv_lines.append(f"{chunk_rel_path}\t{chunk_len}")

            # IMPORTANT FIX:
            # - The first chunk stores the full transcript text (and occupies base_new_line in the new txt)
            # - Later chunks store a placeholder pointing to base_new_line (1-indexed!)
            if c == 0:
                new_txt_lines.append(text)
            else:
                # use the computed base_new_line (1-indexed) — THIS IS THE FIX
                new_txt_lines.append(f"<|{base_new_line}|>")

    # Write new tsv and txt files
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\n".join(new_tsv_lines) + "\n")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(new_txt_lines) + "\n")

    print(f"\n✅ Done! Written:")
    print(f"  TSV: {out_tsv}")
    print(f"  TXT: {out_txt}")
    print(f"  Audio chunks saved under: {args.wav_dir}")


if __name__ == "__main__":
    main()
