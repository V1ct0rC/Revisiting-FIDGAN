import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import auc, roc_auc_score, roc_curve
from matplotlib import pyplot as plt

from models.gan import Generator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_units, num_layers, latent_dim, device=device):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_units, latent_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_units)
        latent_seq = self.linear(lstm_out)  # Shape: (batch_size, seq_len, latent_dim)
        return latent_seq


class AnomalyDetectionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_units, num_layers, latent_dim, save_dir, device=device, load_decoder_index=None, load_encoder_index=None):
        super(AnomalyDetectionAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_units, num_layers, latent_dim, device)
        self.decoder = Generator(latent_dim, hidden_units, num_layers, input_dim, device)
        self.device = device
        self.save_dir = save_dir

        os.makedirs(f"{save_dir}/autoencoder", exist_ok=True)
        
        if load_decoder_index is not None:
            self.decoder.load_state_dict(torch.load(f"{save_dir}/gan/generator_epoch_{load_decoder_index}.pth"))

        if load_encoder_index is not None:
            self.encoder.load_state_dict(torch.load(f"{save_dir}/autoencoder/encoder_epoch_{load_encoder_index}.pth"))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def fit(self, data_loader, num_epochs=300, optimizer=None, lr=0.01, criterion=None):
        """
        Train the autoencoder (encoder only) with frozen decoder
        
        Parameters:
            train_loader (DataLoader): Loader with input sequences
            num_epochs (int): Number of training epochs
            lr (float): Learning rate for encoder
        """
        # Freeze decoder completely
        for param in self.decoder.parameters():
            param.requires_grad = False

        # Initialize optimizer and loss
        if optimizer is None:
            optimizer = optim.Adam(self.encoder.parameters(), lr=lr)

        if criterion is None:
            criterion = nn.MSELoss()

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, real_data in enumerate(data_loader):
                real_data = real_data[0].to(self.device)
                
                # Forward pass
                reconstructions = self(real_data)
                loss = criterion(reconstructions, real_data)
                loss = torch.sqrt(loss)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Save encoder weights
            torch.save(self.encoder.state_dict(), f"{self.save_dir}/autoencoder/encoder_epoch_{epoch}.pth")

            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(data_loader):.4f}")

    def evaluate(self, test_loader):
        """
        Evaluate the model on test data and return AUC score

        Parameters:
            test_loader (DataLoader): Loader providing tuples of (sequences, labels) where labels are 1 (normal) or 0 (anomaly)

        Returns:
            float: AUC score
        """
        self.eval()
        all_scores = []
        all_labels = []
        criterion = nn.MSELoss(reduction='none')
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                
                # Get reconstructions
                reconstructions = self(inputs)
                
                # Calculate per-sample RMSE (sequence and feature dimensions)
                loss = torch.mean((reconstructions - inputs)**2, dim=(1, 2))
                loss = torch.sqrt(loss)

                all_scores.extend(loss.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Handle case with only one class present
        if len(np.unique(all_labels)) < 2:
            raise ValueError("Labels must contain both normal (1) and anomalous (0) samples")
        
        # Calculate AUC (higher MSE = more anomalous)
        fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=0)
        # auc_score = roc_auc_score(all_labels, all_scores)
        auc_score = auc(fpr, tpr)
        
        return auc_score