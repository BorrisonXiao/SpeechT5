#!/usr/bin/env python3
import argparse
from pathlib import Path


def train_dev_split(
    data: Path,
    output_dir: Path,
    ratio: float = 0.95,
):
    """
    Split the raw text data into training and development sets.
    """
    texts = []
    with open(data, "r") as f:
        lines = f.readlines()
    for line in lines:
        texts.append(line)

    n = len(texts)
    n_train = int(n * ratio)
    n_dev = n - n_train

    train_texts = texts[:n_train]
    dev_texts = texts[n_train:]

    with open(output_dir / "text_train.txt", "w") as f:
        for text in train_texts:
            print(text, file=f)
    
    with open(output_dir / "text_valid.txt", "w") as f:
        for text in dev_texts:
            print(text, file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data", type=Path, required=True,
                        help="Path to the data tsv file.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True,
                        help="Path to the output tsv file.")
    parser.add_argument("-r", "--ratio", type=float, default=0.95,
                        help="Ratio of the training set.")

    args = parser.parse_args()

    train_dev_split(
        data=args.data,
        output_dir=args.output_dir,
        ratio=args.ratio,
    )


if __name__ == "__main__":
    main()
