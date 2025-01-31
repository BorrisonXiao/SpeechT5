#!/usr/bin/env python3
import argparse
from pathlib import Path
import copy
import numpy as np
from tqdm import tqdm


def clean_text(
    text: str,
    dictionary: dict = None,
):
    """
    Remove the characters that are not in the dictionary, except for the space.
    """
    if dictionary is None:
        return text
    text = "".join([c if c in dictionary or c == " " else "" for c in text])
    # Remove the extra spaces
    text = " ".join(text.split())
    return text


def train_dev_split(
    data: Path,
    output_dir: Path,
    ratio: float = 0.95,
    merge_style: str = None,
    merge_params: str = None,
    sanitize: bool = False,
    dict_path: Path = None,
):
    """
    Split the raw text data into training and development sets.
    """
    texts = []
    with open(data, "r") as f:
        lines = f.readlines()
    for line in lines:
        # Remove the empty lines
        line = line.strip()
        if line:
            texts.append(line)

    dictionary = None
    if dict_path is not None:
        # Load and parse the dictionary if given
        with open(dict_path, "r") as f:
            lines = f.readlines()
        dictionary = {}
        for line in lines:
            line = line.strip()
            if line:
                key, value = line.split()
                dictionary[key] = value

    # If merge_style is specified, merge the texts
    if merge_style is not None:
        _texts = copy.deepcopy(texts)
        texts = []
        if merge_style == "gaussian":
            mean, std = merge_params.split("|")
            mean = float(mean)
            std = float(std)
            merged_text = []
            max_len = int(np.random.normal(mean, std))

            for i in tqdm(range(len(_texts))):
                # Remove the double hyphens
                normed_text = _texts[i].replace("--", " ")
                if sanitize:
                    normed_text = clean_text(normed_text, dictionary)
                wc = len(normed_text.split())
                if len(merged_text) + wc >= max_len:
                    if len(merged_text) > 0:
                        texts.append(" ".join(merged_text))
                        merged_text = []
                    # Sample a max_len from the Gaussian distribution
                    max_len = int(np.random.normal(mean, std))
                    max_len = max(10, max_len)

                merged_text.extend(normed_text.split())

            # Add the last merged text
            if len(merged_text) > 0:
                texts.append(" ".join(merged_text))

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
    parser.add_argument("--merge-style", type=str, default=None,
                        choices=["gaussian"],
                        help="How to merge the input and output texts.")
    parser.add_argument("--merge-params", type=str, default=None,
                        help="Parameters for the merge style.")
    parser.add_argument("--sanitize", action="store_true",
                        help="Whether to sanitize the input and output texts.")
    parser.add_argument("--dict", type=Path, default=None,
                        help="Path to the dictionary file.")

    args = parser.parse_args()

    train_dev_split(
        data=args.data,
        output_dir=args.output_dir,
        ratio=args.ratio,
        merge_style=args.merge_style,
        merge_params=args.merge_params,
        sanitize=args.sanitize,
        dict_path=args.dict,
    )


if __name__ == "__main__":
    main()
