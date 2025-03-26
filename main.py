import json
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.gan import GAN
from models.autoencoder import Autoencoder
from utils.data import load_dataset


np.random.seed(0)
torch.manual_seed(9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


if __name__ == "__main__":
    # Load settings from the provided file
    settings = load_settings()

    # Extract parameters
    data_path = settings.get("data_path", None)
    seq_length = settings["seq_length"]
    seq_step = settings["seq_step"]
    num_signals = settings["num_signals"]
    latent_dim = settings["latent_dim"]
    hidden_units = settings["hidden_units"]
    num_layers = settings["num_layers"]
    batch_size = settings["batch_size"]
    
    # Load the data -------------------------------------------------------------------------------
    data_path = None
    sequences = load_dataset(
        data_path=data_path,
        seq_length=seq_length,
        seq_step=seq_step,
        num_signals=num_signals
    )

    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(sequences_tensor)

    batch_size = 64
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Initialize and train the GAN model ----------------------------------------------------------
    gan = GAN(
        latent_dim=latent_dim,
        hidden_units=hidden_units,
        num_layers=num_layers,
        input_dim=num_signals,
        output_dim=num_signals,
        device=device
    ).to(device)
    gan.fit()

    # Initialize and train the autoencoder model --------------------------------------------------
    autoencoder = Autoencoder()
    autoencoder.fit()
