#!/usr/bin/env python3
import argparse
import re
import os
from jiwer import wer


def parse_predictions(pred_file):
    """
    Parse a fairseq-style prediction file.
    Returns dicts keyed by sample id (int): refs, hyps, toks, probs
    """
    refs, hyps, toks, probs = {}, {}, {}, {}
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
            elif line.startswith("H-"):
                sid = int(line.split("\t")[0][2:])
                parts = line.split("\t")
                toks[sid] = parts[2] if len(parts) > 2 else ""
            elif line.startswith("P-"):
                sid = int(line.split("\t")[0][2:])
                probs[sid] = line.split("\t")[1] if "\t" in line else ""
    return refs, hyps, toks, probs


def merge_chunked_predictions(refs, hyps, toks, probs, transcript_file):
    """
    Merge predictions based on chunk indicators like <|2|>
    """
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript_lines = [l.strip() for l in f.readlines()]

    merged_refs = {}
    merged_hyps = {}
    merged_toks = {}
    merged_probs = {}

    for i, ref_text in enumerate(transcript_lines):
        line_id = i  # zero-based index for T-xxxx IDs
        # If this is an indicator line
        match = re.fullmatch(r"<\|(\d+)\|>", ref_text)
        if match:
            base_idx = int(match.group(1)) - 1  # convert 1-indexed to 0-indexed
            # Append this prediction to the base line
            if base_idx in merged_hyps and line_id in hyps:
                merged_hyps[base_idx] += " " + hyps[line_id]
                merged_toks[base_idx] += " " + toks.get(line_id, "")
                merged_probs[base_idx] += " " + probs.get(line_id, "")
        else:
            # This is a real transcript (base segment)
            merged_refs[line_id] = ref_text
            merged_hyps[line_id] = hyps.get(line_id, "")
            merged_toks[line_id] = toks.get(line_id, "")
            merged_probs[line_id] = probs.get(line_id, "")

    return merged_refs, merged_hyps, merged_toks, merged_probs


def main():
    parser = argparse.ArgumentParser(description="Compute WER for chunked and unmerged predictions.")
    parser.add_argument("-i", "--input", required=True, help="Path to prediction file.")
    parser.add_argument(
        "-t", "--transcript", required=True, help="Path to transcript txt file."
    )
    args = parser.parse_args()

    # --- Calculation for unmerged WER (Original WER) ---
    refs, hyps, toks, probs = parse_predictions(args.input)
    
    # Get all sample IDs from both refs and hyps to ensure we cover all lines
    all_sids = sorted(list(set(refs.keys()) | set(hyps.keys())))
    
    # Prepare lists for unmerged WER calculation
    # We use all available keys, including those with placeholder tokens
    original_ref_texts = [refs.get(k, "") for k in all_sids]
    original_hyp_texts = [hyps.get(k, "") for k in all_sids]
    
    # Calculate Original WER (including placeholder tokens as normal text)
    original_wer = wer(original_ref_texts, original_hyp_texts) * 100
    
    # --- Calculation for merged WER ---
    merged_refs, merged_hyps, merged_toks, merged_probs = merge_chunked_predictions(
        refs, hyps, toks, probs, args.transcript
    )

    # Prepare lists for merged WER calculation
    ref_texts = [merged_refs[k] for k in sorted(merged_refs.keys())]
    hyp_texts = [merged_hyps[k] for k in sorted(merged_hyps.keys())]

    # Calculate Merged WER
    total_wer = wer(ref_texts, hyp_texts) * 100

    # Write output file
    out_file = os.path.splitext(args.input)[0] + ".wer"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Original WER (all lines): {original_wer:.2f}\n\n")
        
        for k in sorted(merged_refs.keys()):
            f.write(f"T-{k}\t{merged_refs[k]}\n")
            f.write(f"D-{k}\t{merged_hyps[k]}\n\n")
            
        f.write(f"Merged WER: {total_wer:.2f}\n")

    print(f"Original WER (all lines): {original_wer:.2f}")
    print(f"Generate {os.path.basename(out_file)}: Merged WER: {total_wer:.2f}")


if __name__ == "__main__":
    main()