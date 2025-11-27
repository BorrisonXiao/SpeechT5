# import numpy as np
# import os

# bin_path = "/home/ec2-user/mult5/SpeechT5/data/text/bins/train.bin"
# idx_path = "/home/ec2-user/mult5/SpeechT5/data/text/bins/train.idx"

# bin_size = os.path.getsize(bin_path)
# sizes = np.fromfile(idx_path, dtype=np.int32, count=-1, offset=16)  # Adjust offset per header

# element_size = 4  # e.g., 4 for float32; adjust per your dtype
# cumulative = 0
# for i, size in enumerate(sizes):
#     start = cumulative * element_size
#     end = start + size * element_size
#     if end > bin_size:
#         print(f"Error at index {i}: {end} > {bin_size}")
#         break
#     cumulative += size
# else:
#     print("Index is valid.")

#!/usr/bin/env python3
import os
import argparse
import torch
from fairseq.data import IndexedDataset, data_utils
from fairseq.data import indexed_dataset

def check_dataset(data_prefix, dict_path=None, fix_lua_indexing=False, max_samples=5):
    """
    Validate a fairseq dataset (.idx and .bin files)
    
    Args:
        data_prefix: Path prefix to dataset files (e.g., '/path/to/train' for train.idx/bin)
        dict_path: Optional path to dictionary file for decoding
        fix_lua_indexing: Whether to adjust for Lua-based indexing (default: False)
        max_samples: Number of samples to print for visual inspection
    """
    # Check if dataset files exist
    idx_file = data_prefix + '.idx'
    bin_file = data_prefix + '.bin'
    
    if not os.path.exists(idx_file):
        raise FileNotFoundError(f"Index file not found: {idx_file}")
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"Data file not found: {bin_file}")

    # Load dictionary if provided
    dictionary = None
    if dict_path:
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
        from fairseq.data import Dictionary
        dictionary = Dictionary.load(dict_path)
        print(f"Loaded dictionary with {len(dictionary)} tokens")

    # Initialize dataset
    # dataset = IndexedDataset(
    #     data_prefix,
    #     fix_lua_indexing=fix_lua_indexing
    # )
    dataset = indexed_dataset.make_dataset(data_prefix, impl="mmap")

    print(f"Dataset loaded: {len(dataset)} samples")
    print("=" * 50)

    # Iterate through samples
    error_count = 0
    for i in range(len(dataset)):
        try:
            # Get sample
            sample = dataset[i]
            
            # Basic checks
            if not isinstance(sample, torch.Tensor):
                raise TypeError(f"Sample {i} is not a torch.Tensor")
            
            if sample.dim() != 1:
                raise ValueError(f"Sample {i} has wrong dimensionality: {sample.dim()}")
            
            if sample.size(0) == 0:
                raise ValueError(f"Sample {i} is empty")

            # Check token validity if dictionary is available
            if dictionary is not None:
                if (sample >= len(dictionary)).any():
                    raise ValueError(f"Sample {i} contains invalid token indices")
                if (sample < 0).any():
                    raise ValueError(f"Sample {i} contains negative indices")

            # Display first few samples
            if i < max_samples:
                print(f"Sample {i}:")
                print(f"  Shape: {sample.shape}")
                print(f"  Data type: {sample.dtype}")
                print(f"  Min/Max index: {sample.min()}/{sample.max()}")
                
                if dictionary is not None:
                    tokens = ' '.join(dictionary.symbols[token] for token in sample)
                    print(f"  Decoded: {tokens}")
                print()

        except Exception as e:
            error_count += 1
            print(f"❌ Error in sample {i}: {str(e)}")
            if error_count >= 10:  # Stop after 10 errors
                print("Too many errors, stopping...")
                break

    # Summary
    print("=" * 50)
    if error_count == 0:
        print("✅ All samples passed basic validation")
    else:
        print(f"❌ Found {error_count} errors in dataset")
    
    # Additional dataset info
    print(f"\nDataset summary:")
    print(f"Total samples: {len(dataset)}")
    if dictionary:
        print(f"Vocabulary size: {len(dictionary)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate fairseq dataset')
    parser.add_argument('--data-prefix', default="/home/ec2-user/mult5/SpeechT5/data/text/bins/train",
                        help='Path prefix to dataset files (e.g., /path/to/train)')
    parser.add_argument('--dict-path', 
    default="/home/ec2-user/mult5/SpeechT5/data/text/bins/dict.txt",
                        help='Path to dictionary file (optional)')
    parser.add_argument('--fix-lua-indexing', action='store_true',
                        help='Adjust for Lua-based indexing')
    parser.add_argument('--max-samples', type=int, default=5,
                        help='Number of samples to display')
    
    args = parser.parse_args()
    
    check_dataset(
        data_prefix=args.data_prefix,
        dict_path=args.dict_path,
        fix_lua_indexing=args.fix_lua_indexing,
        max_samples=args.max_samples
    )