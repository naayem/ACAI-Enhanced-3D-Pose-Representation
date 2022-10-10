import torch
from torch import nn
import numpy as np

class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.latent_dim = kwargs['latent_dim']
        self.img_shape = kwargs['img_shape']
        self.layers_dim = kwargs['layers_dim']
        self.nb_layers = kwargs['nb_layers']
        self.dropout = kwargs['dropout']

        self.initialize_models()

    def initialize_models(self):
        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        input_size = np.prod(self.img_shape)
        activation = nn.LeakyReLU(0.2, inplace=True)
        dropout = nn.Dropout(self.dropout)
        sub_network = []

        sub_network.extend([nn.Linear(input_size, self.layers_dim), nn.BatchNorm1d(self.layers_dim), activation, dropout])
        
        for _ in range(self.nb_layers-3):
            sub_network.extend([nn.Linear(self.layers_dim, self.layers_dim), nn.BatchNorm1d(self.layers_dim), activation, dropout])

        sub_network.extend([nn.Linear(self.layers_dim, self.latent_dim)])

        self.encoder = nn.Sequential(*sub_network)
        return
    
    def build_decoder(self):
        input_size = np.prod(self.img_shape)
        activation = nn.LeakyReLU(0.2, inplace=True)
        dropout = nn.Dropout(self.dropout)
        sub_network = []

        sub_network.extend([nn.Linear(self.latent_dim, self.layers_dim), nn.BatchNorm1d(self.layers_dim), activation, dropout])
        
        for _ in range(self.nb_layers-3):
            sub_network.extend([nn.Linear(self.layers_dim, self.layers_dim), nn.BatchNorm1d(self.layers_dim), activation, dropout])

        sub_network.extend([nn.Linear(self.layers_dim, input_size)])

        self.decoder = nn.Sequential(*sub_network)
        return

    def forward_encoder(self, img):
        input_size = np.prod(self.img_shape)
        img_flat = img.view(-1, input_size).float() ## How to make this always work
        code = self.encoder(img_flat)
        return code

    def forward_decoder(self, code):
        reconstructed_flat = self.decoder(code)
        reconstructed =  reconstructed_flat.view(-1, *self.img_shape)
        return reconstructed
    
    def forward(self, input, part=''):
        internals = dict()

        if(part == 'decoder'):
            internals["code"] = input
        else:
            code = self.forward_encoder(input)
            internals["code"] = code

        internals["reconstructed"] = self.forward_decoder(internals["code"])
        return internals

class CRITIC(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.latent_dim = kwargs['latent_dim']
        self.img_shape = kwargs['img_shape']
        self.layers_dim = kwargs['layers_dim']
        self.nb_layers = kwargs['nb_layers']
        self.dropout = kwargs['dropout']

        self.build_discriminator()
    
    def build_discriminator(self):
        input_size = np.prod(self.img_shape)
        activation = nn.LeakyReLU(0.2, inplace=True)
        dropout = nn.Dropout(self.dropout)
        sub_network = []

        sub_network.extend([nn.Linear(input_size, self.layers_dim), nn.BatchNorm1d(self.layers_dim), activation, dropout])
        
        for _ in range(self.nb_layers-3):
            sub_network.extend([nn.Linear(self.layers_dim, self.layers_dim), nn.BatchNorm1d(self.layers_dim), activation, dropout])

        sub_network.extend([nn.Linear(self.layers_dim, self.latent_dim)])

        self.discriminator = nn.Sequential(*sub_network)
        return

    def forward_discriminator(self, img):
        input_size = np.prod(self.img_shape)
        img_flat = img.view(-1, input_size).float() ## How to make this always work
        interp_coef = torch.mean(self.discriminator(img_flat), axis=1)
        return interp_coef
        
    def forward(self, img):
        return self.forward_discriminator(img)

class AES(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.latent_dim = kwargs['latent_dim']
        self.img_shape = kwargs['img_shape']
        self.layers_dim = kwargs['layers_dim']
        self.nb_layers = kwargs['nb_layers']
        self.dropout = kwargs['dropout']

        self.initialize_models()

    def initialize_models(self):
        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        input_size = np.prod(self.img_shape)
        activation = nn.LeakyReLU(0.2, inplace=True)
        dropout = nn.Dropout(self.dropout)
        sub_network = []

        sub_network.extend([nn.Linear(input_size, self.layers_dim), nn.BatchNorm1d(self.layers_dim), activation, dropout])
        
        for _ in range(self.nb_layers-3):
            sub_network.extend([nn.Linear(self.layers_dim, self.layers_dim), nn.BatchNorm1d(self.layers_dim), activation, dropout])

        sub_network.extend([nn.Linear(self.layers_dim, self.latent_dim)])

        self.encoder = nn.Sequential(*sub_network)
        return
    
    def build_decoder(self):
        input_size = np.prod(self.img_shape)
        activation = nn.LeakyReLU(0.2, inplace=True)
        dropout = nn.Dropout(self.dropout)
        sub_network = []

        sub_network.extend([nn.Linear(self.latent_dim, self.layers_dim), nn.BatchNorm1d(self.layers_dim), activation, dropout])
        
        for _ in range(self.nb_layers-3):
            sub_network.extend([nn.Linear(self.layers_dim, self.layers_dim), nn.BatchNorm1d(self.layers_dim), activation, dropout])

        sub_network.extend([nn.Linear(self.layers_dim, input_size)])

        self.decoder = nn.Sequential(*sub_network)
        return

    def forward_encoder(self, img):
        input_size = np.prod(self.img_shape)
        img_flat = img.view(-1, input_size).float() ## How to make this always work
        code = self.encoder(img_flat)
        return code

    def forward_decoder(self, code):
        reconstructed_flat = self.decoder(code)
        reconstructed =  reconstructed_flat.view(-1, *self.img_shape)
        return reconstructed
    
    def forward(self, input, part=''):
        internals = dict()

        if(part == 'decoder'):
            internals["code"] = input
        else:
            code = self.forward_encoder(input)
            code = code / code.norm(dim=1, keepdim=True) # z on the unit sphere
            internals["code"] = code

        internals["reconstructed"] = self.forward_decoder(internals["code"])
        return internals