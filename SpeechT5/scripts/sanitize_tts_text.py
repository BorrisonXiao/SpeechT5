#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

def remove_unclosed_quotes(text: str) -> str:
    """Remove unclosed leading and trailing quotation marks."""
    # Pattern to match unclosed quotes at start/end
    # ^['"]|['"]$ - matches quote at beginning OR quote at end
    # But we need to be more careful about matching pairs
    
    # Remove unclosed leading quote
    if text and text[0] in ['"', "'"]:
        # Check if there's a matching closing quote
        closing_quote_pos = text.find(text[0], 1)
        if closing_quote_pos == -1:  # No matching closing quote found
            text = text[1:]
        # If there is a matching closing quote, leave it as is
    
    # Remove unclosed trailing quote (after processing the beginning)
    if text and text[-1] in ['"', "'"]:
        # Check if there's a matching opening quote before this position
        opening_quote_pos = text.rfind(text[-1], 0, len(text)-1)
        if opening_quote_pos == -1:  # No matching opening quote found
            text = text[:-1]
    
    return text

def sanitize_text(
    data: Path,
    output: Path,
    encoding: str = "utf-8"
) -> None:
    """Sanitize file contents by removing unclosed quotation marks"""
    try:
        with data.open("r", encoding=encoding) as infile, \
             output.open("w", encoding=encoding) as outfile:

            for line in infile:
                # Remove unclosed quotes and preserve the line
                processed = remove_unclosed_quotes(line.strip())
                outfile.write(processed + "\n")
                
        print(f"Successfully created: {output}")
        
    except Exception as e:
        print(f"Error processing {data}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Sanitize text file by removing unclosed quotation marks"
    )
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Input file to process")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output file for sanitized text")
    parser.add_argument("--encoding", type=str, default="utf-8",
                        help="File encoding (default: utf-8)")

    args = parser.parse_args()
    
    sanitize_text(
        data=args.input,
        output=args.output,
        encoding=args.encoding
    )

if __name__ == "__main__":
    main()