#!/usr/bin/env python
import os
import shutil
import random
import argparse
from pathlib import Path

def sample_libritts(input_dir, percent, output_dir):
    # Ensure input path is a Path object
    src_root = Path(input_dir)
    dest_root = Path(output_dir)
    
    # Get all speaker directories (the "103" level)
    # We filter to ensure we only pick directories
    all_speakers = [d for d in src_root.iterdir() if d.is_dir()]
    
    # Calculate how many speakers to sample
    sample_size = max(1, int(len(all_speakers) * (percent / 100)))
    sampled_speakers = random.sample(all_speakers, sample_size)
    
    print(f"Found {len(all_speakers)} total speakers.")
    print(f"Sampling {sample_size} speakers ({percent}%)...")

    # Create output directory if it doesn't exist
    dest_root.mkdir(parents=True, exist_ok=True)

    # Copy data
    for speaker_path in sampled_speakers:
        speaker_id = speaker_path.name
        target_path = dest_root / speaker_id
        
        print(f"Copying speaker {speaker_id}...")
        
        # shutil.copytree will copy the speaker folder and all subfolders (chapters)
        if target_path.exists():
            shutil.rmtree(target_path) # Clean up if it exists
        shutil.copytree(speaker_path, target_path)

    print(f"\nSuccessfully sampled data to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a percentage of speakers from LibriTTS.")
    parser.add_argument("-d", "--input-dir", default="/export/fs06/cxiao7/LibriTTS/train-clean-100", help="Path to the train-clean-100 directory")
    parser.add_argument("-p", "--percent", type=float, default=10, help="Percentage of speakers to sample (e.g., 10)")
    parser.add_argument("-o", "--output-dir", default="/export/fs06/cxiao7/LibriTTS-debug/train-debug-100-10", help="Path to the output directory")

    args = parser.parse_args()
    
    sample_libritts(args.input_dir, args.percent, args.output_dir)