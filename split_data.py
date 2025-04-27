import pandas as pd
import os

def split_csv(input_file, output_dir, chunk_size=5000):
    """
    Split a large CSV file into smaller chunks
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save chunks
        chunk_size (int): Number of rows per chunk
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read CSV in chunks
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        output_path = os.path.join(output_dir, f'train_part_{i+1}.csv')
        chunk.to_csv(output_path, index=False)
        print(f'Saved chunk {i+1} to {output_path}')

if __name__ == '__main__':
    # Split the training data
    split_csv('train.csv', 'train_chunks') 