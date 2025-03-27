import json
import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.gan import GAN
from models.autoencoder import Autoencoder
from utils.data import load_dataset, load_settings


np.random.seed(0)
torch.manual_seed(9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load settings from the provided file --------------------------------------------------------
    print("-"*100 + "\n" + "Loading settings")
    settings = load_settings()

    # Extract parameters
    data_path = settings["data_path"]
    num_epochs = settings["num_epochs"]
    seq_length = settings["seq_length"]
    seq_step = settings["seq_step"]
    num_signals = settings["num_signals"]
    latent_dim = settings["latent_dim"]
    hidden_units = settings["hidden_units"]
    num_layers = settings["num_layers"]
    batch_size = settings["batch_size"]
    learning_rate_gan = settings["learning_rate_gan"]
    learning_rate_autoencoder = settings["learning_rate_autoencoder"]

    print("Settings:")
    print("Data path:", data_path)
    
    # Load the data -------------------------------------------------------------------------------
    print("-"*100 + "\n" + "Loading data")
    sequences = load_dataset(
        data_path=data_path,
        seq_length=seq_length,
        seq_step=seq_step,
        num_signals=num_signals
    )

    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(sequences_tensor)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Initialize and train the GAN model ----------------------------------------------------------
    print("-"*100 + "\n" + "Training GAN")

    base_dir = "experiments/gan"
    run_name = (
        f"latent_{latent_dim}_hidden_{hidden_units}_"
        f"layers_{num_layers}_sign_{num_signals}_"
        f"epochs_{num_epochs}_lr_{learning_rate_gan}"
    )
    save_dir = os.path.join(base_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    gan = GAN(
        latent_dim=latent_dim,
        hidden_units=hidden_units,
        num_layers=num_layers,
        input_dim=num_signals,
        output_dim=num_signals,
        save_dir=save_dir,
        device=device
    ).to(device)
    gan.fit(train_loader, seq_length, num_epochs=num_epochs, d_lr=learning_rate_gan, g_lr=learning_rate_gan)

    # # Initialize and train the autoencoder model --------------------------------------------------
    # print("-"*100 + "\n" + "Training Autoencoder")
    # autoencoder = Autoencoder()
    # autoencoder.fit()
