#!/usr/bin/env python3
import torchaudio
import torchaudio.transforms as T
import torch
from pathlib import Path
from tqdm import tqdm
import os
import sys
import argparse # New import for command-line arguments

# Default list of splits (can be overridden by the --splits argument)
DEFAULT_DEV_SETS = [
    "dev-clean",
    "dev-other",
    # Add other splits here if needed, e.g., "test-clean"
]

def resample_and_save_split(
    split_name: str, 
    orig_root: Path, 
    dest_root: Path, 
    target_sr: int
):
    """
    Finds all .wav files in a source split, resamples them to the target_sr, 
    and saves them to the destination directory while preserving the internal 
    folder structure. The original sample rate is dynamically detected per file.
    """
    
    # 1. Define source and destination paths based on the shell logic:
    # Source root for this split: ORG_DATA_DIR/dev-clean
    source_path = orig_root / split_name
    
    # Destination root: TSV_DIR/raw/valid/
    # The shell script targets 'valid' as the final destination for all splits
    # from ${dev_sets}.
    destination_path_root = dest_root

    if not source_path.is_dir():
        print(f"Error: Source directory not found: {source_path}", file=sys.stderr)
        return

    # 2. Find all WAV files and process
    # Use glob to find all .wav files recursively under the source directory
    all_wav_files = list(source_path.glob('**/*.wav'))

    if not all_wav_files:
        print(f"Warning: No .wav files found in {source_path}")
        return

    print(f"\nProcessing split: '{split_name}' with {len(all_wav_files)} files...")

    for audio_file_path in tqdm(all_wav_files, desc=f"Resampling {split_name}"):
        try:
            # 3. Determine the relative path to preserve directory structure
            # Example: relative_path will be spk_id/file_id.wav
            relative_path = audio_file_path.relative_to(source_path)
            
            # 4. Determine the full destination path
            target_file_path = destination_path_root / relative_path

            # Create destination folder if it doesn't exist
            target_file_path.parent.mkdir(parents=True, exist_ok=True)

            # 5. Load the audio and dynamically detect its sample rate
            # Load the audio (waveform is a Tensor, sr is the sample rate)
            waveform, sr = torchaudio.load(audio_file_path)
            
            # 6. Initialize the Resampler dynamically
            # If the file's sample rate (sr) is already the target rate, skip resampling
            if sr == target_sr:
                # Just save the original waveform
                resampled_waveform = waveform
            else:
                # Initialize the resampler using the detected source rate (sr)
                resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
                # Apply the resampling transform
                resampled_waveform = resampler(waveform)

            # 7. Save the (resampled or original) audio to the new path
            # The output file is always saved with the target sample rate.
            torchaudio.save(
                target_file_path, 
                resampled_waveform, 
                target_sr, 
                format="wav"
            )

        except Exception as e:
            tqdm.write(f"\n[ERROR] Failed to process {audio_file_path}: {e}")
            continue

    print(f"Finished processing split '{split_name}'. Resampled files saved to {destination_path_root}")


def parse_args():
    """
    Parses command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(
        description="Resamples audio files from specified splits and saves them "
                    "to a target directory structure, dynamically detecting source SR."
    )
    parser.add_argument(
        '-i', '--input-dir', 
        type=Path, 
        required=True, 
        help="Root directory containing original data (e.g., /data/LibriSpeech)."
    )
    parser.add_argument(
        '-o', '--output-dir', 
        type=Path, 
        required=True, 
        help="Root directory for saving resampled data (the tsv_dir equivalent)."
    )
    parser.add_argument(
        '-s', '--target-sr', 
        type=int, 
        default=16000,
        help="The desired sample rate for the output files (e.g., 16000)."
    )
    parser.add_argument(
        '--splits', 
        nargs='+', # Accepts one or more arguments
        default=DEFAULT_DEV_SETS, 
        help=f"List of data splits to process (e.g., dev-clean dev-other). Default is: {' '.join(DEFAULT_DEV_SETS)}"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Map parsed arguments to local variables for clarity
    org_data_dir = args.input_dir
    tsv_dir = args.output_dir
    target_sample_rate = args.target_sr
    dev_sets = args.splits

    if not org_data_dir.is_dir() or not org_data_dir.exists():
        print(f"Error: Original data root not found at {org_data_dir}.", file=sys.stderr)
        return

    if target_sample_rate <= 0:
        print("Error: Target sample rate must be a positive integer.", file=sys.stderr)
        return

    print(f"Starting Resampling Process (Source SR detected dynamically):")
    print(f"  Input Directory: {org_data_dir}")
    print(f"  Output Directory: {tsv_dir}")
    print(f"  Target SR: {target_sample_rate} Hz")
    print(f"  Splits to Process: {', '.join(dev_sets)}")
    print("-" * 30)

    for split in dev_sets:
        resample_and_save_split(
            split, 
            org_data_dir, 
            tsv_dir, 
            target_sample_rate
        )

    print("\nResampling script finished successfully.")

if __name__ == "__main__":
    main()