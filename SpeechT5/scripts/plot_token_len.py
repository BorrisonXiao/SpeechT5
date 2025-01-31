import argparse
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

def read_token_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip().split() for line in lines]

def plot_token_length_distribution(token_sequences, output_path=None):
    token_lengths = [len(tokens) for tokens in tqdm(token_sequences)]
    
    total_sents = len(token_sequences)
    plt.hist(token_lengths, bins=range(1, max(token_lengths) + 2), edgecolor='black')
    plt.title(f'Token Length Distribution ({total_sents})')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot token length distribution from a .token file.')
    parser.add_argument('-i', '--file_path', type=Path, required=True, help='Path to the .token file.')
    parser.add_argument('-o', '--output_path', type=Path, help='Path to save the plot.')
    args = parser.parse_args()

    token_sequences = read_token_file(args.file_path)
    plot_token_length_distribution(token_sequences, args.output_path)

if __name__ == '__main__':
    main()