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

def normalize_content(text: str, keep_puncs: list = None) -> str:
    """
    Keep only English characters (A-Z, a-z) and lowercase everything.
    Optionally keep specific punctuation marks.
    """
    # Escape punctuation marks to ensure they are safe for Regex
    escaped_puncs = "".join([re.escape(p) for p in keep_puncs]) if keep_puncs else ""
    
    # Create pattern: match anything NOT (^) a-z, A-Z, whitespace, or allowed puncs
    pattern = rf"[^a-zA-Z\s{escaped_puncs}]"
    
    text = re.sub(pattern, '', text)
    return text.lower()

def sanitize_text(
    input_path: Path,
    output_path: Path,
    normalize: bool = False,
    keep_puncs: list = None,
    encoding: str = "utf-8"
) -> None:
    """Sanitize file contents based on quote logic and normalization rules."""
    try:
        with input_path.open("r", encoding=encoding) as infile, \
             output_path.open("w", encoding=encoding) as outfile:

            for line in infile:
                # 1. Handle unclosed quotes
                processed = remove_unclosed_quotes(line)
                
                # 2. Optional Normalization
                if normalize:
                    processed = normalize_content(processed, keep_puncs)
                
                outfile.write(processed.strip() + "\n")
                
        print(f"Successfully processed: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Sanitize text file with quote removal and optional normalization."
    )
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Input file")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output file")
    parser.add_argument("--normalize-text", action="store_true",
                        help="Keep only English characters and lowercase")
    parser.add_argument("--keep-puncs", nargs="+", default=[",", ".", "!", "?"],
                        help="Punctuation marks to keep (e.g., . , ! ?)")
    parser.add_argument("--encoding", type=str, default="utf-8",
                        help="File encoding (default: utf-8)")

    args = parser.parse_args()
    
    sanitize_text(
        input_path=args.input,
        output_path=args.output,
        normalize=args.normalize_text,
        keep_puncs=args.keep_puncs,
        encoding=args.encoding
    )

if __name__ == "__main__":
    main()