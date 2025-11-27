#!/usr/bin/env python3
import argparse
from pathlib import Path

"""
Generate the dict.km.txt file for the k-means labels on the training split
    python scripts/generate_dict.py \
        -i ${lab_dir}/speech_train.km \
        -o ${lab_dir}/dict.km.txt
The first column is the labels which occur in train.km, which contains the labels of the training split.
e.g. 127 127 127 127 148 148 391 391 391 304 304 391 ...
The second column is the count of each label
e.g.
10 100
23 300
35 500
...
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Path to the input file.")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Path to the output file.")

    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = f.readlines()
    
    label_count = {}
    for line in lines:
        labels = line.strip().split()
        for label in labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

    with open(args.output, "w") as f:
        for label, count in label_count.items():
            print(f"{label} {count}", file=f)


if __name__ == "__main__":
    main()
