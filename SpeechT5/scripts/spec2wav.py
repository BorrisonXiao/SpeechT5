#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torchaudio
from transformers import SpeechT5HifiGan, AutoTokenizer
from pathlib import Path


def load_hifigan_model(device):
    """Load pre-trained HiFi-GAN model from Hugging Face Hub"""
    model_name = "microsoft/speecht5_hifigan"  # Default model for 16kHz, 80-band mels
    model = SpeechT5HifiGan.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model


def process_mel_to_wav(model, mel_spectrogram, device):
    """Convert mel-spectrogram to waveform using HiFi-GAN"""
    with torch.no_grad():
        # Ensure correct shape and type
        if isinstance(mel_spectrogram, np.ndarray):
            mel_spectrogram = torch.from_numpy(mel_spectrogram)

        # Add batch dimension if missing
        if len(mel_spectrogram.shape) == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)

        mel_spectrogram = mel_spectrogram.to(device)
        waveform = model(mel_spectrogram).squeeze().cpu().numpy()
    return waveform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        default="/home/cxiao7/research/mult5/SpeechT5/SpeechT5/exp/inference_tts/v1.0-checkpoint_4_18000/test-clean",
        help="Input directory containing .npy files",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="/home/cxiao7/research/mult5/SpeechT5/SpeechT5/exp/inference_tts/v1.0-checkpoint_4_18000/test-clean_wav",
        help="Output directory for .wav files",
    )
    parser.add_argument(
        "--fs", default=16000, type=int, help="Sample rate (default: 16000)"
    )
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading HiFi-GAN model...")
    model = load_hifigan_model(device)

    # Process files
    input_path = Path(args.input_dir)
    npy_files = list(input_path.rglob("*.npy"))

    print(f"Found {len(npy_files)} .npy files")
    for npy_file in npy_files:
        try:
            # Load mel-spectrogram
            mel = np.load(npy_file)

            # Generate waveform
            waveform = process_mel_to_wav(model, mel, device)

            # Save output
            relative_path = npy_file.relative_to(input_path)
            output_path = Path(args.output_dir) / relative_path.with_suffix(".wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(
                str(output_path), torch.from_numpy(waveform).unsqueeze(0), args.fs
            )
            print(f"Generated: {output_path}")

        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}")


if __name__ == "__main__":
    main()
