#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

def remove_unclosed_quotes(text: str) -> str:
    """Remove unclosed leading and trailing quotation marks."""
    text = text.strip()
    if not text:
        return text

    # Remove unclosed leading quote
    if text[0] in ('"', "'"):
        if text.find(text[0], 1) == -1:
            text = text[1:]
    
    if not text: return text

    # Remove unclosed trailing quote
    if text[-1] in ('"', "'"):
        if text.rfind(text[-1], 0, len(text) - 1) == -1:
            text = text[:-1]
            
    return text

def normalize_content(text: str) -> str:
    """Keep only English characters (A-Z, a-z) and lowercase everything."""
    # This regex keeps only a-z and A-Z, then we lowercase the result
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def sanitize_text(
    input_path: Path,
    output_path: Path,
    normalize: bool = False,
    encoding: str = "utf-8"
) -> None:
    """Sanitize file contents by removing unclosed quotes and optionally normalizing."""
    try:
        with input_path.open("r", encoding=encoding) as infile, \
             output_path.open("w", encoding=encoding) as outfile:

            for line in infile:
                # 1. Handle unclosed quotes
                processed = remove_unclosed_quotes(line)
                
                # 2. Optional Normalization
                if normalize:
                    processed = normalize_content(processed)
                
                # Write non-empty lines (or adjust if you want to keep blank lines)
                outfile.write(processed.strip() + "\n")
                
        print(f"Successfully created: {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Sanitize text file by removing unclosed quotation marks and normalizing text."
    )
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Input file to process")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output file for sanitized text")
    parser.add_argument("--normalize-text", action="store_true",
                        help="Keep only English characters and lowercase everything")
    parser.add_argument("--encoding", type=str, default="utf-8",
                        help="File encoding (default: utf-8)")

    args = parser.parse_args()
    
    sanitize_text(
        input_path=args.input,
        output_path=args.output,
        normalize=args.normalize_text,
        encoding=args.encoding
    )

if __name__ == "__main__":
    main()