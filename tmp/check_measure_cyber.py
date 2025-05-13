import numpy as np
import os

# Load base matrix
base = np.load('base.npy')
n = base.shape[0]

# Folder path containing .npy files (excluding base.npy)
folder_path = './'  # Adjust this if needed
thresh = 0.8
largest_n = n

for filename in os.listdir(folder_path):
    if filename.endswith('.npy') and filename != 'base.npy':
        file_path = os.path.join(folder_path, filename)
        matrix = np.load(file_path)

        # Compute absolute difference
        diff = np.abs(matrix - base)

        # Get top largest_n largest changes
        flat_indices = np.argpartition(diff.flatten(), -largest_n)[-largest_n:]
        top_mask = np.zeros_like(diff, dtype=int)
        top_mask[np.unravel_index(flat_indices, diff.shape)] = 1

        # Sum top-n binary indicators across columns
        column_counts = np.sum(top_mask, axis=0)

        if np.max(column_counts) >= thresh * largest_n:
            print(f'{filename}: Measurement anomaly')
        else:
            print(f'{filename}: Cyber anomaly')

        # Print top 5 columns with largest absolute diff sum
        column_sums = np.sum(diff, axis=0)
        top_indices = np.argsort(column_sums)[-5:][::-1]
        print(f'Top 5 root cause in {filename}:')
        for rank, idx in enumerate(top_indices, 1):
            print(f'  {rank}. Column {idx}, Sum = {column_sums[idx]:.4f}')
