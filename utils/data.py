import argparse
import json
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

    Returns:
        np.ndarray: Processed sequences
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


def load_labeled_dataset(data_path, seq_length, seq_step, num_signals):
    """
    Process pre-normalized time series data into sequences with PCA dimensionality reduction

    Parameters:
        seq_length (int): Length of the sliding window sequences
        seq_step (int): Step size between sequence starts
        num_signals (int): Number of PCA components to keep
        data_path (str): Path to .npy file containing data

    Returns:
        sequences (np.ndarray): Sequences of sliding windows
        sequence_labels (np.ndarray): Labels for each sequence
    """
    # Load the dataset
    dataset = np.load(data_path)

    # Verify input dimensions
    if len(dataset.shape) != 2:
        raise ValueError("Dataset should be 2D array of shape (time_steps, features)")
    
    samples = dataset[:, :-1]
    labels = dataset[:, -1]
        
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=num_signals, svd_solver='full')
    reduced_data = pca.fit_transform(samples)
    
    # Create sliding window sequences and corresponding labels
    sequences = []
    sequence_labels = []
    num_total_steps = reduced_data.shape[0]
    
    for start_idx in range(0, num_total_steps - seq_length + 1, seq_step):
        end_idx = start_idx + seq_length
        
        # Get sequence and corresponding labels
        sequence = reduced_data[start_idx:end_idx]
        label_window = labels[start_idx:end_idx]
        
        # Use majority voting for sequence label (or last label)
        # label = np.mean(label_window) > 0.5  # For probabilistic labels
        label = label_window[-1]  # Use last label in window
        
        sequences.append(sequence)
        sequence_labels.append(label)
        
    return np.array(sequences), np.array(sequence_labels)
    


def load_settings():
    """
    Load settings from the provided file.
    """
    parser = argparse.ArgumentParser(description="Run GAN and Autoencoder training.")
    parser.add_argument("--settings", type=str, required=True, help="Path to the settings file (JSON)")
    args = parser.parse_args()
    file_path = args.settings

    with open(file_path, "r") as f:
        if file_path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported file format.")
