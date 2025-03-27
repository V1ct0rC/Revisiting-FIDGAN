import os
import torch
import torch.nn as nn
import torch.optim as optim

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


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_units, num_layers, latent_dim, save_dir, device=device, load_model_index=None):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_units, num_layers, latent_dim, device)
        self.decoder = Generator(latent_dim, hidden_units, num_layers, input_dim, device)
        self.device = device
        self.save_dir = save_dir

        os.makedirs(f"{save_dir}/autoencoder", exist_ok=True)
        
        if load_model_index is not None:
            self.decoder.load_state_dict(torch.load(f"{save_dir}/gan/generator_epoch_{load_model_index}.pth"))

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

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Save encoder weights
            torch.save(self.encoder.state_dict(), f"{self.save_dir}/autoencoder/encoder_epoch_{epoch+1}.pth")

            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(data_loader):.4f}")