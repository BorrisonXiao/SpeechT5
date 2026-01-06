#!/usr/bin/env python3
"""
Script to join multiple TSV and TXT files, create symlinks to audio files, 
and write new TSV and TXT files with synchronized entries.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Set, Tuple, Optional
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Join multiple TSV and TXT files, create symlinks to audio files, "
                    "and write new TSV and TXT files with synchronized entries."
    )
    parser.add_argument(
        "-i", "--input-tsvs",
        nargs="+",
        required=True,
        help="List of input TSV files to join"
    )
    parser.add_argument(
        "-t", "--input-txts",
        nargs="+",
        required=True,
        help="List of input TXT files (transcriptions) in the same order as TSV files"
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Directory where symlinks to raw audio files will be created"
    )
    parser.add_argument(
        "-o", "--output-prefix",
        required=True,
        help="Output file prefix (without extension) for both TSV and TXT files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually doing it"
    )
    
    return parser.parse_args()


def read_tsv_file(tsv_path: str) -> Tuple[str, List[str]]:
    """
    Read a TSV file and return the header (first line) and entries.
    
    Args:
        tsv_path: Path to the TSV file
        
    Returns:
        Tuple of (header_line, list_of_entries)
    """
    with open(tsv_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return "", []
    
    # First line is the header (directory path)
    header = lines[0].strip()
    # Rest are entries
    entries = [line.strip() for line in lines[1:]]
    
    return header, entries


def read_txt_file(txt_path: str) -> List[str]:
    """
    Read a TXT file and return all lines.
    
    Args:
        txt_path: Path to the TXT file
        
    Returns:
        List of text lines
    """
    with open(txt_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    
    return lines


def create_symlink(original_path: str, raw_dir: Path, dry_run: bool = False) -> Path:
    """
    Create a symlink from the original file to the raw directory.
    
    Args:
        original_path: Path to the original audio file
        raw_dir: Directory where symlinks will be created
        dry_run: If True, only print what would be done
        
    Returns:
        Path to the created symlink
    """
    # Convert to Path object for easier manipulation
    original = Path(original_path)
    
    if not original.exists():
        raise FileNotFoundError(f"Original file not found: {original_path}")
    
    # Create the symlink path
    # Preserve the directory structure within raw_dir
    symlink_path = raw_dir / original.name
    
    # Ensure the target directory exists
    if not dry_run:
        raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if symlink already exists
    if symlink_path.exists():
        if symlink_path.is_symlink():
            # Verify it points to the correct file
            if symlink_path.resolve() == original.resolve():
                return symlink_path
            else:
                # Remove incorrect symlink
                if not dry_run:
                    symlink_path.unlink()
        else:
            # It's not a symlink, remove it
            if not dry_run:
                symlink_path.unlink()
    
    # Create the symlink
    if not dry_run:
        # Use absolute path for the symlink target
        symlink_path.symlink_to(original.resolve())
    
    return symlink_path


def process_files(
    input_tsvs: List[str], 
    input_txts: List[str], 
    raw_dir: str, 
    dry_run: bool = False
) -> Tuple[str, List[str], List[str]]:
    """
    Process all input TSV and TXT files, create symlinks, and collect synchronized entries.
    
    Args:
        input_tsvs: List of input TSV file paths
        input_txts: List of input TXT file paths
        raw_dir: Directory where symlinks will be created
        dry_run: If True, only print what would be done
        
    Returns:
        Tuple of (new_header, list_of_processed_tsv_entries, list_of_processed_text_lines)
    """
    all_tsv_entries = []
    all_text_lines = []
    processed_files = set()  # To avoid duplicates
    raw_dir_path = Path(raw_dir)
    
    if len(input_tsvs) != len(input_txts):
        raise ValueError(f"Number of TSV files ({len(input_tsvs)}) must match number of TXT files ({len(input_txts)})")
    
    print(f"Processing {len(input_tsvs)} TSV/TXT file pairs...")
    print(f"Raw directory: {raw_dir_path}")
    
    for tsv_file, txt_file in tqdm(zip(input_tsvs, input_txts), 
                                    total=len(input_tsvs), 
                                    desc="Processing file pairs"):
        print(f"\nProcessing pair:")
        print(f"  TSV: {tsv_file}")
        print(f"  TXT: {txt_file}")
        
        try:
            # Read TSV file
            header, tsv_entries = read_tsv_file(tsv_file)
            if not tsv_entries:
                print(f"Warning: No entries found in {tsv_file}")
                continue
            
            # Read TXT file
            txt_lines = read_txt_file(txt_file)
            
            # Check if counts match
            if len(tsv_entries) != len(txt_lines):
                print(f"Warning: Number of entries in TSV ({len(tsv_entries)}) "
                      f"doesn't match TXT ({len(txt_lines)}) for {tsv_file}")
                # Use the smaller count to avoid index errors
                min_count = min(len(tsv_entries), len(txt_lines))
                tsv_entries = tsv_entries[:min_count]
                txt_lines = txt_lines[:min_count]
                print(f"  Using first {min_count} entries from each file")
            
            # Process each entry pair
            for tsv_entry, txt_line in tqdm(zip(tsv_entries, txt_lines), 
                                            total=len(tsv_entries), 
                                            desc="Processing entries", 
                                            leave=False):
                if not tsv_entry.strip():  # Skip empty TSV entries
                    continue
                
                # Split the TSV entry into parts
                parts = tsv_entry.split('\t')
                
                # Construct the full path to the audio file
                # The first column in TSV is relative to the header directory
                audio_filename = parts[0]
                original_audio_path = Path(header) / audio_filename
                
                # Check if we've already processed this file
                if str(original_audio_path) in processed_files:
                    print(f"Skipping duplicate: {original_audio_path}")
                    continue
                
                # Create symlink
                try:
                    symlink_path = create_symlink(str(original_audio_path), raw_dir_path, dry_run)
                    
                    # Update the TSV entry with the new path
                    # Use just the filename (not full path) for the new TSV
                    new_audio_path = symlink_path.name
                    
                    # Reconstruct the TSV entry with the new audio path
                    if len(parts) == 3:
                        new_tsv_entry = f"{new_audio_path}\t{parts[1]}\t{parts[2]}"
                    elif len(parts) == 2:
                        new_tsv_entry = f"{new_audio_path}\t{parts[1]}"
                    else:
                        # Just in case there's only one column
                        new_tsv_entry = f"{new_audio_path}"
                    
                    all_tsv_entries.append(new_tsv_entry)
                    all_text_lines.append(txt_line)
                    processed_files.add(str(original_audio_path))
                    
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing {original_audio_path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error reading {tsv_file} or {txt_file}: {e}")
            continue
    
    # Create new header - use the raw_dir as specified
    new_header = str(raw_dir_path)
    
    print(f"\nProcessed {len(all_tsv_entries)} unique audio files with synchronized text")
    return new_header, all_tsv_entries, all_text_lines


def write_output_files(
    output_prefix: str, 
    header: str, 
    tsv_entries: List[str], 
    text_lines: List[str], 
    dry_run: bool = False
):
    """
    Write the processed data to new TSV and TXT files.
    
    Args:
        output_prefix: Output file prefix (without extension)
        header: Header line for the TSV file
        tsv_entries: List of TSV entry lines
        text_lines: List of text lines
        dry_run: If True, only print what would be written
    """
    # Create output file paths
    output_tsv_path = f"{output_prefix}.tsv"
    output_txt_path = f"{output_prefix}.txt"
    
    if dry_run:
        print(f"\n[DRY RUN] Would write output to:")
        print(f"  TSV: {output_tsv_path}")
        print(f"  TXT: {output_txt_path}")
        print(f"TSV Header: {header}")
        print(f"Number of entries: {len(tsv_entries)}")
        if tsv_entries:
            print("First 3 TSV entries:")
            for i, entry in enumerate(tsv_entries[:3]):
                print(f"  {i+1}. {entry}")
        if text_lines:
            print("First 3 text lines:")
            for i, line in enumerate(text_lines[:3]):
                print(f"  {i+1}. {line}")
        return
    
    # Ensure the output directory exists
    output_dir = Path(output_tsv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write TSV file
    with open(output_tsv_path, 'w') as f:
        # Write header
        f.write(f"{header}\n")
        # Write all entries
        for entry in tsv_entries:
            f.write(f"{entry}\n")
    
    # Write TXT file
    with open(output_txt_path, 'w') as f:
        for line in text_lines:
            f.write(f"{line}\n")
    
    print(f"\nOutput written:")
    print(f"  TSV: {output_tsv_path}")
    print(f"  TXT: {output_txt_path}")
    print(f"Total synchronized entries: {len(tsv_entries)}")


def main():
    args = parse_arguments()
    
    print("=" * 60)
    print("TSV/TXT Joiner Script")
    print("=" * 60)
    
    if args.dry_run:
        print("\nDRY RUN MODE - No files will be modified\n")
    
    # Process the TSV and TXT files
    new_header, processed_tsv_entries, processed_text_lines = process_files(
        args.input_tsvs, 
        args.input_txts,
        args.raw_dir, 
        args.dry_run
    )
    
    # Write the output TSV and TXT files
    write_output_files(
        args.output_prefix, 
        new_header, 
        processed_tsv_entries, 
        processed_text_lines, 
        args.dry_run
    )
    
    if not args.dry_run:
        print("\nDone!")
    else:
        print("\nDry run completed. Use without --dry-run to execute.")


if __name__ == "__main__":
    main()