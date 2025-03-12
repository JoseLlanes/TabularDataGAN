import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


class EncoderDecoderGenerator(nn.Module):
    def __init__(self, in_out_dim, num_layers=3, initial_neurons=256, latent_dim=None):
        """
        - in_out_dim: Input and output dimension (same for autoencoder).
        - num_layers: Number of hidden layers (excluding input, output, and latent layers).
        - initial_neurons: Number of neurons in the first hidden layer (shrinks progressively).
        - latent_dim: Dimension of the latent space (computed if None).
        """
        super(EncoderDecoderGenerator, self).__init__()

        if latent_dim is None:
            latent_dim = initial_neurons // (2 ** (num_layers - 1))
        
        # ###############
        # ### Encoder ###
        # ###############
        encoder_layers = []
        for i in range(num_layers):
            if i == 0:
                encoder_layers.extend(self.custom_layer(in_out_dim, initial_neurons))
                neurons = initial_neurons
            else:
                encoder_layers.extend(self.custom_layer(neurons, neurons // 2))
                neurons //= 2

        # Latent dimension
        encoder_layers.extend(self.custom_layer(neurons, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ###############
        # ### Decoder ###
        # ###############
        decoder_layers = []
        neurons = latent_dim
        for _ in range(num_layers):
            decoder_layers.extend(self.custom_layer(neurons, neurons * 2))
            neurons *= 2

        # Final layer
        decoder_layers.append(nn.Linear(neurons, in_out_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    @staticmethod
    def custom_layer(input_dim, output_dim, dropout_rate=0.1):
        return [
            nn.Linear(input_dim, output_dim), 
            nn.BatchNorm1d(output_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate)
        ]

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output


class CNN1DEncoderDecoder(nn.Module):
    def __init__(self, in_out_channels, num_layers=3, initial_filters=64, kernel_size=3, latent_dim=None):
        """
        Params:
        - in_out_channels: Number of input and output channels.
        - num_layers: Number of convolutional layers.
        - initial_filters: Number of filters in the first convolutional layer.
        - kernel_size: Size of the convolutional kernel.
        - latent_dim: Size of the latent representation.
        """
        super(CNN1DEncoderDecoder, self).__init__()
        
        if latent_dim is None:
            latent_dim = initial_filters // (2 ** (num_layers - 1))
        
        # ###############
        # ### Encoder ###
        # ###############
        encoder_layers = []
        filters = initial_filters
        for i in range(num_layers):
            encoder_layers.append(nn.Conv1d(in_out_channels if i == 0 else filters, 2 * filters, kernel_size, stride=2, padding=1))
            encoder_layers.append(nn.BatchNorm1d(2 * filters))
            encoder_layers.append(nn.ReLU())
            filters *= 2
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent layer
        self.latent = nn.Conv1d(filters, latent_dim, kernel_size=1)
        
        # ###############
        # ### Decoder ###
        # ###############
        decoder_layers = []
        for i in range(num_layers):
            decoder_layers.append(nn.ConvTranspose1d(latent_dim if i == 0 else filters, filters // 2, kernel_size, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.BatchNorm1d(filters // 2))
            decoder_layers.append(nn.ReLU())
            filters //= 2
        
        # Final layer
        decoder_layers.append(nn.Conv1d(filters, in_out_channels, kernel_size=1))
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.latent(encoded)
        output = self.decoder(latent)
        return output
