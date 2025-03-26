import torch
import torch.nn as nn

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
    def __init__(self, input_dim, hidden_units, num_layers, latent_dim, device=device):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_units, num_layers, latent_dim, device)
        self.decoder = Generator(latent_dim, hidden_units, num_layers, input_dim, device)

        # TODO: Load generator weights into decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def fit(self):
        # TODO: Implement training loop
        pass