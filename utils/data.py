import numpy as np
from sklearn.decomposition import PCA


def load_dataset(data_path, seq_length, seq_step, num_signals):
    """
    Process pre-normalized time series data into sequences with PCA dimensionality reduction"

    Parameters:
        seq_length (int): Length of the sliding window sequences
        seq_step (int): Step size between sequence starts
        num_signals (int): Number of PCA components to keep
        data_path (str): Path to .npy file containing data
    """

    # Load the dataset
    dataset = np.load(data_path)

    # Verify input dimensions
    if len(dataset.shape) != 2:
        raise ValueError("Dataset should be 2D array of shape (time_steps, features)")
        
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=num_signals, svd_solver='full')
    reduced_data = pca.fit_transform(dataset)
    
    # Create sliding window sequences
    sequences = []
    num_total_steps = reduced_data.shape[0]
    
    for start_idx in range(0, num_total_steps - seq_length + 1, seq_step):
        end_idx = start_idx + seq_length
        sequence = reduced_data[start_idx:end_idx]
        sequences.append(sequence)
        
    return np.array(sequences)
