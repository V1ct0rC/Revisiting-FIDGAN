import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_units, num_layers, output_dim, device=device):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.device = device
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.linear = nn.Linear(hidden_units, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, z):
        batch_size, seq_len = z.size(0), z.size(1)

        # Initialize hidden state TODO: Check if this is necessary
        # h = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device)
        # c = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device)

        lstm_out, _ = self.lstm(z)
        output = self.linear(lstm_out)
        return self.tanh(output)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_units, num_layers, device=device):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Use the last time step's output
        output = self.linear(last_out)
        return self.sigmoid(output)
    
    def predict(self, x):
        """
        Predict using the discriminator.
        """
        return self.forward(x)


class AnomalyDetectionGAN(nn.Module):
    def __init__(self, latent_dim, hidden_units, num_layers, input_dim, output_dim, save_dir, device=device, load_model_index=None):
        super(AnomalyDetectionGAN, self).__init__()
        self.device = device
        self.save_dir = save_dir

        # Initialize generator and discriminator
        self.generator = Generator(latent_dim, hidden_units, num_layers, output_dim, device=self.device)
        self.discriminator = Discriminator(input_dim, hidden_units, num_layers, device=self.device)

        os.makedirs(f"{save_dir}/gan", exist_ok=True)

        # Load pre-trained model if specified
        if load_model_index is not None:
            self.generator.load_state_dict(torch.load(f"{save_dir}/gan/generator_epoch_{load_model_index}.pth"))
            self.discriminator.load_state_dict(torch.load(f"{save_dir}/gan/discriminator_epoch_{load_model_index}.pth"))

    def forward(self, x, z, seq_len):
        fake_x = self.generator(z, seq_len)
        fake_pred = self.discriminator(fake_x)
        real_pred = self.discriminator(x)
        return fake_x, fake_pred, real_pred
    
    def fit(self, data_loader, seq_len, num_epochs=100, d_optimizer=None, g_optimizer=None, d_lr=0.001, g_lr=0.001, criterion=None):
        """
        Fit the GAN model.

        Parameters:
            data_loader (torch.utils.data.DataLoader): Data loader
            seq_len (int): Sequence length
            num_epochs (int): Number of epochs
            d_optimizer (torch.optim.Optimizer): Discriminator optimizer
            g_optimizer (torch.optim.Optimizer): Generator optimizer
            d_lr (float): Discriminator learning rate
            g_lr (float): Generator learning rate
            criterion (torch.nn.Module): Loss function
        """
        if d_optimizer is None:
            d_optimizer = optim.SGD(self.discriminator.parameters(), lr=d_lr)
        if g_optimizer is None:    
            g_optimizer = optim.Adam(self.generator.parameters(), lr=g_lr)

        if criterion is None:
            criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            for batch_idx, real_data in enumerate(data_loader):
                real_data = real_data[0].to(self.device)
                batch_size = real_data.size(0)
 
                # Train Discriminator -----------------------------------------------------------------
                self.discriminator.train()

                # Generate noise
                z = torch.randn(batch_size, seq_len, self.generator.latent_dim).to(self.device)

                # Get discriminator outputs
                fake_data = self.generator(z)
                real_pred = self.discriminator(real_data)
                fake_pred = self.discriminator(fake_data.detach())

                # Calculate losses
                real_loss = criterion(real_pred, torch.ones_like(real_pred))  # Discriminator should predict 1 for real data
                fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))  # Discriminator should predict 0 for fake data
                d_loss = (real_loss + fake_loss) / 2

                d_optimizer.zero_grad()
                # Backpropagation
                d_loss.backward()
                d_optimizer.step()

                # Train Discriminator -----------------------------------------------------------------
                self.generator.train()

                # Generate new fake data
                fake_data = self.generator(z)
                fake_pred = self.discriminator(fake_data)

                # Calculate generator loss (flipped labels)
                g_loss = criterion(fake_pred, torch.ones_like(fake_pred))

                g_optimizer.zero_grad()
                # Backpropagation
                g_loss.backward()
                g_optimizer.step()

            # Print progress
            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"Disc Loss: {d_loss.item():.6f} "
                f"Gen Loss: {g_loss.item():.6f}"
            )

            torch.save(self.generator.state_dict(), f"{self.save_dir}/gan/generator_epoch_{epoch}.pth")
            torch.save(self.discriminator.state_dict(), f"{self.save_dir}/gan/discriminator_epoch_{epoch}.pth")
    
    def predict(self, x):
        """
        Predict using the discriminator.

        Parameters:
            x (torch.Tensor): Input

        Returns:
            torch.Tensor: Discriminator
        """
        return self.discriminator.predict(x)

    def generate(self, z):
        """
        Generate data using the generator.

        Parameters:
            z (torch.Tensor): Input noise

        Returns:
            torch.Tensor: Generator output
        """
        return self.generator(z)
    
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
        
        with torch.no_grad():
            for batch in test_loader:
                # Assume test_loader returns (data, labels)
                data, labels = batch
                data = data[0].to(self.device) if isinstance(data, list) else data.to(self.device)
                
                # Get discriminator predictions (probability of being real)
                preds = self.discriminator(data)
                
                # Convert to anomaly scores (1 - probability of being real)
                scores = 1 - preds.cpu().numpy().flatten()
                all_scores.extend(scores)
                all_labels.extend(labels.numpy().flatten())
        
        # Handle case with only one class present
        if len(np.unique(all_labels)) < 2:
            raise ValueError("Labels must contain both normal (0) and anomalous (1) samples")
        
        return roc_auc_score(all_labels, all_scores)


if __name__ == "__main__":
    seq_length = 30

    generator = Generator(latent_dim=15, hidden_units=100, num_layers=3, output_dim=seq_length, device=device).to(device)
    discriminator = Discriminator(input_dim=seq_length, hidden_units=100, num_layers=3, device=device).to(device)

    z = torch.randn(32, seq_length, 15).to(device)
    gen_out = generator(z)
    dis_out = discriminator(gen_out)
    
    print("Latent: ", z.size())
    print("Generator output: ", gen_out.size())
    print("Discriminator output: ", dis_out.size())