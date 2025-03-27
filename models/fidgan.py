import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

from models.gan import Generator, Discriminator
from models.autoencoder import Encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FIDGAN(nn.Module):
    def __init__(self, input_dim_encoder, hidden_units, num_layers, latent_dim, output_dim_generator, save_dir, device=device, load_generator_index=None, load_discriminator_index=None, load_encoder_index=None):
        super(FIDGAN, self).__init__()
        self.generator = Generator(latent_dim=latent_dim, hidden_units=hidden_units, num_layers=num_layers, output_dim=output_dim_generator, device=device)
        self.discriminator = Discriminator(input_dim=output_dim_generator, hidden_units=hidden_units, num_layers=num_layers, device=device)
        self.encoder = Encoder(input_dim=input_dim_encoder, hidden_units=hidden_units, num_layers=num_layers, latent_dim=latent_dim, device=device)
        self.device = device

        if load_generator_index is not None:
            self.generator.load_state_dict(torch.load(f"{save_dir}/gan/generator_epoch_{load_generator_index}.pth"))
        
        if load_discriminator_index is not None:
            self.discriminator.load_state_dict(torch.load(f"{save_dir}/gan/discriminator_epoch_{load_discriminator_index}.pth"))

        if load_encoder_index is not None:
            self.encoder.load_state_dict(torch.load(f"{save_dir}/autoencoder/encoder_epoch_{load_encoder_index}.pth"))


    def adScore(self, discrimination_loss, reconstruction_loss, tau=1):
        """
        Compute the anomaly detection score based on the reconstruction and discrimination losses.
        """
        return (tau * discrimination_loss) + ((1 - tau) * reconstruction_loss)
    
    def forward(self, x):
        """
        Forward pass through the FIDGAN.

        Parameters:
            x (torch.Tensor): Input data of shape (batch_size, seq_len, input_dim)
        
        Returns:
            tuple: (reconstructed_data, discriminator_output)
        """
        # Encode input sequence to latent vector
        z = self.encoder(x)  # (batch_size, latent_dim)
        
        # Expand latent vector to match sequence length
        batch_size, seq_len = x.size(0), x.size(1)
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, latent_dim)
        
        # Generate reconstructed sequence
        x_recon = self.generator(z_seq)  # (batch_size, seq_len, output_dim)
        
        # Get discriminator predictions for reconstructed data
        d_out = self.discriminator(x)  # (batch_size, 1)
        
        return x_recon, d_out
    
    def evaluate(self, test_loader, tau=0.92):
        """
        Evaluate the model on test data and return AUC score
        
        Parameters:
            test_loader (DataLoader): Loader providing tuples of (sequences, labels)
                                    where labels are 1 (normal) or 0 (anomaly)
            tau (float): Weighting parameter for anomaly score combination
        
        Returns:
            float: AUC score
        """
        self.eval()
        all_scores = []
        all_labels = []

        discrimination_loss = nn.BCELoss(reduction='none')
        reconstruction_loss = nn.MSELoss(reduction='none')
        
        with torch.inference_mode():
            for batch in test_loader:
                data, labels = batch
                data = data.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward pass
                x_recon, d_out = self(data)
                
                # Calculate losses
                recon_loss = reconstruction_loss(x_recon, data).mean(dim=(1,2))
                disc_loss = discrimination_loss(d_out.squeeze(), labels)
                
                # Combine losses
                scores = self.adScore(disc_loss, recon_loss, tau)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate AUC (higher score = more anomalous)
        return roc_auc_score(all_labels, all_scores)