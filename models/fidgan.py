import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import auc, roc_auc_score, roc_curve

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
        self.save_dir = save_dir

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
        z = self.encoder(x)
        
        # Generate reconstructed sequence
        x_recon = self.generator(z)  # (batch_size, seq_len, output_dim)
        
        # Get discriminator predictions for reconstructed data
        d_out = self.discriminator(x)  # (batch_size, 1)
        
        return x_recon, d_out
    
    def _scale(self, arr):
        if np.max(arr) - np.min(arr) == 0:
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    def _reduction(self, fpr, tpr):
        """
        Reduce the number of points in the ROC curve to avoid overplotting.

        parameters:
            fpr (np.ndarray): False positive rates
            tpr (np.ndarray): True positive rates

        returns:
            list: Reduced false positive rates
            list: Reduced true positive rates
        """
        fpr2 = []
        tpr2 = []
        fpr2.append(fpr[0])
        tpr2.append(tpr[0])
        for i in range(1, fpr.size):
            d = pow(( pow(fpr2[-1] - fpr[i], 2) + pow(tpr2[-1] - tpr[i], 2) ), 0.5)
            if( d > 0.07 ):
                fpr2.append(fpr[i])
                tpr2.append(tpr[i])
        
        return fpr2, tpr2

    def evaluate(self, test_loader, tau=0.99):
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

        all_disc_scores = []
        all_recon_scores = []

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
                recon_loss = reconstruction_loss(x_recon, data).mean(dim=(1, 2))
                disc_loss = discrimination_loss(d_out.squeeze(), labels)
                
                # Combine losses
                # scores = self.adScore(disc_loss, recon_loss, tau)
                
                all_disc_scores.extend(disc_loss.cpu().numpy())
                all_recon_scores.extend(recon_loss.cpu().numpy())

                # all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_disc_scores = self._scale(np.array(all_disc_scores))
        all_recon_scores = self._scale(np.array(all_recon_scores))

        all_scores = self.adScore(all_disc_scores, all_recon_scores, tau)

        # Calculate AUC for the discriminator scores only
        fpr_ld, tpr_ld, _ = roc_curve(all_labels, all_disc_scores, pos_label=0)
        auc_score_ld = auc(fpr_ld, tpr_ld)

        fpr_ld2, tpr_ld2 = self._reduction(fpr_ld, tpr_ld)
        
        # Plot and save ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_ld2, tpr_ld2, lw=2, label=f'ROC curve (AUC = {auc_score_ld:.2f})', linestyle=':', marker='o', color='r', markersize = 6)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Discriminator Loss')
        plt.legend(loc="lower right")
        
        # Ensure directory exists
        save_auc_dir = os.path.join(self.save_dir, "metrics", "AUC")
        os.makedirs(save_auc_dir, exist_ok=True)
        plt.savefig(os.path.join(save_auc_dir, "Ld_roc_curve.png"))
        plt.close()

        # Calculate AUC for the reconstruction scores only
        fpr_lr, tpr_lr, _ = roc_curve(all_labels, all_recon_scores, pos_label=0)
        auc_score_lr = auc(fpr_lr, tpr_lr)

        fpr_lr2, tpr_lr2 = self._reduction(fpr_lr, tpr_lr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_lr2, tpr_lr2, lw=2, label=f'ROC curve (AUC = {auc_score_lr:.2f})', linestyle='-.', marker='s', color='g', markersize = 7)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Reconstruction Loss')
        plt.legend(loc="lower right")

        # Ensure directory exists
        save_auc_dir = os.path.join(self.save_dir, "metrics", "AUC")
        os.makedirs(save_auc_dir, exist_ok=True)
        plt.savefig(os.path.join(save_auc_dir, "Lr_roc_curve.png"))
        plt.close()
        
        # Calculate AUC (higher score = more anomalous)
        fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=0)
        # auc_score = roc_auc_score(all_labels, all_scores)
        auc_score = auc(fpr, tpr)

        fpr2, tpr2 = self._reduction(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr2, tpr2, lw=2, label=f'ROC curve (AUC = {auc_score:.2f})', linestyle='--', marker='^', color='b', markersize = 6)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - Combined Loss (tau={tau})')
        plt.legend(loc="lower right")

        # Ensure directory exists
        save_auc_dir = os.path.join(self.save_dir, "metrics", "AUC")
        os.makedirs(save_auc_dir, exist_ok=True)
        plt.savefig(os.path.join(save_auc_dir, "roc_curve.png"))
        plt.close()


        return auc_score