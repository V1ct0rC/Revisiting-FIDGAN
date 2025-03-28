import json
import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.fidgan import FIDGAN
from models.gan import AnomalyDetectionGAN
from models.autoencoder import AnomalyDetectionAutoencoder
from utils.data import load_dataset, load_settings, load_labeled_dataset


np.random.seed(0)
torch.manual_seed(9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load settings from the provided file --------------------------------------------------------
    print("-"*100 + "\n" + "Loading settings")
    settings = load_settings()

    # Extract parameters
    dataset = settings["dataset"]

    train_data_path = settings["train_data_path"]
    valid_data_path = settings["valid_data_path"]

    num_epochs = settings["num_epochs_gan"]
    num_epochs_autoencoder = settings["num_epochs_autoencoder"]

    seq_length = settings["seq_length"]
    seq_step = settings["seq_step"]
    num_signals = settings["num_signals"]
    latent_dim = settings["latent_dim"]
    hidden_units = settings["hidden_units"]
    num_layers = settings["num_layers"]
    batch_size = settings["batch_size"]

    learning_rate_gan = settings["learning_rate_gan"]
    learning_rate_autoencoder = settings["learning_rate_autoencoder"]

    base_dir_exp = "experiments"
    run_name = (
        f"{dataset}_"
        f"latent_{latent_dim}_hidden_{hidden_units}_"
        f"layers_{num_layers}_sign_{num_signals}_"
        f"epochs_{num_epochs}_lr_{learning_rate_gan}"
    )
    save_dir_run = os.path.join(base_dir_exp, run_name)
    os.makedirs(save_dir_run, exist_ok=True)
    
    # Load the data -------------------------------------------------------------------------------
    print("-"*100 + "\n" + "Loading data")

    sequences_train = load_dataset(
        data_path=train_data_path,
        seq_length=seq_length,
        seq_step=seq_step,
        num_signals=num_signals
    )

    sequences_train_tensor = torch.tensor(sequences_train, dtype=torch.float32)
    train_dataset = TensorDataset(sequences_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

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

    # Initialize and train the GAN model ----------------------------------------------------------
    print("-"*100 + "\n" + "Training GAN")

    gan = AnomalyDetectionGAN(
        latent_dim=latent_dim, hidden_units=hidden_units, num_layers=num_layers,
        input_dim=num_signals, output_dim=num_signals, save_dir=save_dir_run,
        device=device
    ).to(device)
    gan.fit(
        train_loader, seq_length, 
        num_epochs=num_epochs, d_lr=learning_rate_gan, g_lr=learning_rate_gan
    )    

    # Evaluate the GAN models ---------------------------------------------------------------------
    print("-"*100 + "\n" + "Evaluating GAN")
    
    auc_scores_only_Ld = []
    for epoch in range(num_epochs):
        gan = AnomalyDetectionGAN(
            latent_dim=latent_dim, hidden_units=hidden_units, num_layers=num_layers,
            input_dim=num_signals, output_dim=num_signals, save_dir=save_dir_run,
            device=device, load_model_index=epoch
        ).to(device)
        auc = gan.evaluate(valid_loader)
        auc_scores_only_Ld.append(auc)

    best_gan_epopch = np.argmax(auc_scores_only_Ld)
    print(f"AUC scores: {auc_scores_only_Ld}")
    print(f"Best GAN epoch: {best_gan_epopch}")

    # Initialize and train the autoencoder model --------------------------------------------------
    print("-"*100 + "\n" + "Training Autoencoder")

    autoencoder = AnomalyDetectionAutoencoder(
        input_dim=num_signals, hidden_units=hidden_units, num_layers=num_layers,
        latent_dim=latent_dim, save_dir=save_dir_run, device=device, load_decoder_index=best_gan_epopch
    ).to(device)
    autoencoder.fit(
        train_loader, num_epochs=num_epochs_autoencoder, lr=learning_rate_autoencoder
    )

    # Evaluate the autoencoder model --------------------------------------------------------------
    print("-"*100 + "\n" + "Evaluating Autoencoder")

    auc_scores_only_Lr = []
    for epoch in range(num_epochs_autoencoder):
        autoencoder = AnomalyDetectionAutoencoder(
            input_dim=num_signals, hidden_units=hidden_units, num_layers=num_layers,
            latent_dim=latent_dim, save_dir=save_dir_run, device=device, 
            load_decoder_index=best_gan_epopch, load_encoder_index=epoch
        ).to(device)
        auc = autoencoder.evaluate(valid_loader)
        auc_scores_only_Lr.append(auc)

    best_autoencoder_epoch = np.argmax(auc_scores_only_Lr)
    print(f"AUC scores: {auc_scores_only_Lr}")
    print(f"Best Autoencoder epoch: {best_autoencoder_epoch}")

    # # Initialize FIDGAN model ---------------------------------------------------------------------
    # print("-"*100 + "\n" + "Setting up FIDGAN")

    # fidgan = FIDGAN(
    #     input_dim_encoder=num_signals, hidden_units=hidden_units, num_layers=num_layers,
    #     latent_dim=latent_dim, output_dim_generator=num_signals, device=device,
    #     save_dir=save_dir_run, load_generator_index=best_gan_epopch, load_discriminator_index=best_gan_epopch,
    #     load_encoder_index=best_autoencoder_epoch
    # ).to(device)

    # # Evaluate the FIDGAN model --------------------------------------------------------------------
    # print("-"*100 + "\n" + "Evaluating FIDGAN")

    # fidgan_auc = fidgan.evaluate(valid_loader)
    # print(f"FIDGAN AUC: {fidgan_auc}")
