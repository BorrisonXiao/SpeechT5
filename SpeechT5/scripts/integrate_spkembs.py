#!/usr/bin/env python3
import ast
import logging
import os
import sys
import argparse
from pathlib import Path
import zipfile
import csv

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from omegaconf import DictConfig


def integrate_spkembs(
    data: Path,
    dset: str,
    xvector_dir: Path,
    output: Path,
):
    """
    Integrate the third column, i.e. the speaker embeddings, into the data tsv file.
    Note that the speaker embeddings are stored in a zip file, where each file is named as the speaker id.
    """
    if dset not in ["librispeech"]:
        raise NotImplementedError(f"Dataset {dset} is not supported yet.")
    zf = zipfile.ZipFile(xvector_dir)
    spkid2offsets = {}
    for zinfo in zf.infolist():
        spkid = Path(zinfo.filename).stem
        offset = zinfo.header_offset + len(zinfo.FileHeader())
        length = zinfo.file_size
        spkid2offsets[spkid] = (offset, length)
        
    # Parse the tsv file
    with open(data, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        _data = list(reader)
        
    # For each row, get the speaker id and the corresponding xvector offset
    # The additional column looks like: xvectors.zip:<offset>:<length>, e.g. "xvectors.zip:164335:2176"
    res = []
    for i, row in enumerate(_data):
        if i == 0:
            res.append(row)
            continue
        spkid = "-".join(Path(row[0]).stem.split("-")[:-1])
        offset, length = spkid2offsets[spkid]
        res.append(row + [f"{Path(xvector_dir).name}:{offset}:{length}"])

    # Write the new tsv file
    with open(output, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data", type=Path, required=True,
                        help="Path to the data tsv file.")
    parser.add_argument("--dset", type=str, choices=["librispeech"],
                        default="librispeech", help="Dataset name.")
    parser.add_argument("--xvectors", type=Path, required=True,
                        help="Path to the xvector zip file.")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Path to the output tsv file.")

    args = parser.parse_args()

    integrate_spkembs(
        data=args.data,
        dset=args.dset,
        xvector_dir=args.xvectors,
        output=args.output,
    )


if __name__ == "__main__":
    main()
