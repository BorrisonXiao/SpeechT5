#!/usr/bin/env python3
import argparse
from pathlib import Path

def lowercase(
    data: Path,
    output: Path,
    sanitize: bool = False,
    encoding: str = "utf-8"
) -> None:
    """Convert file contents to lowercase and write to .lc file"""
    try:
        with data.open("r", encoding=encoding) as infile, \
             output.open("w", encoding=encoding) as outfile:

            for line in infile:
                processed = line.lower()
                if sanitize:
                    processed = " ".join(processed.split()).strip() + "\n"
                outfile.write(processed)
                
        print(f"Successfully created: {output}")
        
    except Exception as e:
        print(f"Error processing {data}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Convert text file to lowercase version"
    )
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Input file to process")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output directory for .lc file")
    parser.add_argument("--encoding", type=str, default="utf-8",
                        help="File encoding (default: utf-8)")

    args = parser.parse_args()
    
    lowercase(
        data=args.input,
        output=args.output,
        encoding=args.encoding
    )

if __name__ == "__main__":
    main()
