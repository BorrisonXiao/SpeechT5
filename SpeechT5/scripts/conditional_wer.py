#!/usr/bin/env python3
import argparse
import os
from jiwer import wer


def parse_predictions(pred_file):
    """
    Parse a fairseq-style prediction file.
    Returns dicts keyed by sample id (int): refs, hyps
    """
    refs, hyps = {}, {}
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("T-"):
                sid = int(line.split("\t")[0][2:])
                # T- lines usually contain the reference text
                refs[sid] = line.split("\t", 1)[1] if "\t" in line else ""
            elif line.startswith("D-"):
                sid = int(line.split("\t")[0][2:])
                parts = line.split("\t")
                # D- lines usually contain the hypothesis (prediction) text
                hyps[sid] = parts[2] if len(parts) > 2 else ""
    # Only return refs and hyps as they are sufficient for WER calculation
    return refs, hyps


def parse_tsv(tsv_file, threshold):
    """
    Parse a TSV file to determine which T-xxxx IDs to exclude.
    
    The TSV file has a 1-line header and uses 1-based indexing for its data.
    T-xxxx IDs are 0-based.
    TSV Line Index (1-based) = T-xxxx ID (0-based) + 2
    T-xxxx ID = TSV Line Index - 2
    
    Returns a set of T-xxxx IDs (integers) to exclude.
    """
    exclude_sids = set()
    
    with open(tsv_file, "r", encoding="utf-8") as f:
        # Skip the header line
        f.readline()
        
        # Start data line index from 2 (since the header is line 1)
        # This 'i' variable is the 1-based line number of the data rows.
        for i, line in enumerate(f, start=2):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            try:
                # The duration is in the second column
                duration = int(parts[1])
            except ValueError:
                # Skip lines where the duration is not a valid integer
                continue

            # Check if duration exceeds the threshold
            if duration > threshold:
                # The T-xxxx ID is 0-based and corresponds to the index in the 
                # prediction file (e.g., T-0, T-1, ...).
                # T-xxxx ID = TSV Line Index - 2
                sid = i - 2
                exclude_sids.add(sid)
                
    return exclude_sids


def main():
    parser = argparse.ArgumentParser(description="Compute Conditional WER based on TSV duration threshold.")
    parser.add_argument("-i", "--input", required=True, help="Path to prediction file (fairseq format).")
    parser.add_argument("-t", "--tsv", required=True, help="Path to TSV file containing durations.")
    parser.add_argument(
        "-T", 
        "--threshold", 
        type=int, 
        default=320000, 
        help="Duration threshold (integer) above which utterances are excluded (default: 320000)."
    )
    args = parser.parse_args()

    # 1. Parse Predictions
    refs, hyps = parse_predictions(args.input)
    
    # 2. Determine Exclusions from TSV
    exclude_sids = parse_tsv(args.tsv, args.threshold)
    
    # 3. Filter Data
    conditional_refs = {}
    conditional_hyps = {}
    
    # Get all sample IDs from the prediction file that have both a reference and a hypothesis
    all_sids = sorted(list(set(refs.keys()) & set(hyps.keys())))
    
    # Filter the data
    included_count = 0
    excluded_count = 0
    for sid in all_sids:
        if sid not in exclude_sids:
            # Utterance is included
            conditional_refs[sid] = refs[sid]
            conditional_hyps[sid] = hyps[sid]
            included_count += 1
        else:
            # Utterance is excluded
            excluded_count += 1

    # 4. Compute Conditional WER
    if included_count == 0:
        conditional_wer = 0.0
        print(f"Warning: No utterances were included for WER calculation (Threshold: {args.threshold}).")
    else:
        # Sort by sid for consistent order
        sorted_sids = sorted(conditional_refs.keys())
        ref_texts = [conditional_refs[k] for k in sorted_sids]
        hyp_texts = [conditional_hyps[k] for k in sorted_sids]
        
        conditional_wer = wer(ref_texts, hyp_texts) * 100

    # 5. Output Results
    out_file = os.path.splitext(args.input)[0] + f".cond_wer_{args.threshold}.txt"
    
    print("-" * 50)
    print(f"Total Utterances Found: {len(all_sids)}")
    print(f"Duration Threshold: > {args.threshold}")
    print(f"Utterances Excluded: {excluded_count}")
    print(f"Utterances Included: {included_count}")
    print(f"Conditional WER: {conditional_wer:.2f}%")
    print(f"Results written to: {os.path.basename(out_file)}")
    print("-" * 50)

    # Write detailed output file
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Duration Threshold: > {args.threshold}\n")
        f.write(f"Utterances Excluded: {excluded_count}\n")
        f.write(f"Utterances Included: {included_count}\n\n")
        
        # Write included segments
        f.write("--- Included Segments ---\n")
        for k in sorted(conditional_refs.keys()):
            f.write(f"T-{k}\t{conditional_refs[k]}\n")
            f.write(f"D-{k}\t{conditional_hyps[k]}\n")
        
        f.write("\n")
        f.write(f"Conditional WER: {conditional_wer:.2f}\n")


if __name__ == "__main__":
    main()