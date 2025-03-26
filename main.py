import torch
import numpy as np

from models.gan import GAN
from models.autoencoder import Autoencoder

np.random.seed(0)
torch.manual_seed(9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Load the data


    # Initialize and train the GAN model
    gan = GAN()
    gan.fit()

    # Initialize and train the autoencoder model
    autoencoder = Autoencoder()
    autoencoder.fit()

    