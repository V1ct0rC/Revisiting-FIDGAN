import json
import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.gan import AnomalyDetectionGAN
from models.autoencoder import Autoencoder
from utils.data import load_dataset, load_settings, load_labeled_dataset


np.random.seed(0)
torch.manual_seed(9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load settings from the provided file --------------------------------------------------------
    print("-"*100 + "\n" + "Loading settings")
    settings = load_settings()

    # Extract parameters
    train_data_path = settings["train_data_path"]
    valid_data_path = settings["valid_data_path"]

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
    
    # # Load the data -------------------------------------------------------------------------------
    # print("-"*100 + "\n" + "Loading data")
    # sequences_train = load_dataset(
    #     data_path=train_data_path,
    #     seq_length=seq_length,
    #     seq_step=seq_step,
    #     num_signals=num_signals
    # )

    # sequences_train_tensor = torch.tensor(sequences_train, dtype=torch.float32)
    # train_dataset = TensorDataset(sequences_train_tensor)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=True
    # )

    # # Initialize and train the GAN model ----------------------------------------------------------
    # print("-"*100 + "\n" + "Training GAN")

    # base_dir_gan = "experiments/gan"
    # run_name_gan = (
    #     f"latent_{latent_dim}_hidden_{hidden_units}_"
    #     f"layers_{num_layers}_sign_{num_signals}_"
    #     f"epochs_{num_epochs}_lr_{learning_rate_gan}"
    # )
    # save_dir_gan = os.path.join(base_dir_gan, run_name_gan)
    # os.makedirs(save_dir_gan, exist_ok=True)

    # gan = AnomalyDetectionGAN(
    #     latent_dim=latent_dim, hidden_units=hidden_units, num_layers=num_layers,
    #     input_dim=num_signals, output_dim=num_signals, save_dir=save_dir_gan,
    #     device=device
    # ).to(device)
    # gan.fit(
    #     train_loader, seq_length, 
    #     num_epochs=num_epochs, d_lr=learning_rate_gan, g_lr=learning_rate_gan
    # )

    # Load the validation data --------------------------------------------------------------------
    print("-"*100 + "\n" + "Loading validation data")

    sequences_valid, labels_valid = load_labeled_dataset(
        data_path=valid_data_path,
        seq_length=seq_length,
        seq_step=seq_step,
        num_signals=num_signals
    )

    sequences_valid_tensor = torch.tensor(sequences_valid, dtype=torch.float32)
    labels_valid_tensor = torch.tensor(labels_valid, dtype=torch.float32)
    valid_dataset = TensorDataset(sequences_valid_tensor, labels_valid_tensor)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Evaluate the GAN models ---------------------------------------------------------------------
    print("-"*100 + "\n" + "Evaluating GAN")

    base_dir_gan = "experiments/gan"
    run_name_gan = (
        f"latent_{latent_dim}_hidden_{hidden_units}_"
        f"layers_{num_layers}_sign_{num_signals}_"
        f"epochs_{num_epochs}_lr_{learning_rate_gan}"
    )
    save_dir_gan = os.path.join(base_dir_gan, run_name_gan)
    
    auc_scores = []
    for epoch in range(num_epochs):
        gan = AnomalyDetectionGAN(
            latent_dim=latent_dim, hidden_units=hidden_units, num_layers=num_layers,
            input_dim=num_signals, output_dim=num_signals, save_dir=save_dir_gan,
            device=device, load_model_index=epoch
        ).to(device)
        auc = gan.evaluate(valid_loader)
        auc_scores.append(auc)

    best_epoch = np.argmax(auc_scores)

    # # Initialize and train the autoencoder model --------------------------------------------------
    # print("-"*100 + "\n" + "Training Autoencoder")
    # autoencoder = Autoencoder()
    # autoencoder.fit()
