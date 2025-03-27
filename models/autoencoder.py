import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, roc_curve
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
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.device = device

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # Get final hidden state from last layer
        return self.linear(last_hidden)


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
        # Expand latent vector to match decoder's expected sequence length
        z_seq = z.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, latent_dim)
        return self.decoder(z_seq)
    
    def fit(self, data_loader, num_epochs=300, optimizer=None, lr=0.001, criterion=None):
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
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        auc_score = roc_auc_score(all_labels, all_scores)
        
        # Plot and save ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Autoencoder')
        plt.legend(loc="lower right")
        
        # Create save directory
        save_auc_dir = os.path.join(self.save_dir, "metrics", "AUC")
        os.makedirs(save_auc_dir, exist_ok=True)
        plt.savefig(os.path.join(save_auc_dir, "Lr_roc_curve.png"))
        plt.close()
        
        return auc_score