import torch
from torch import nn
import numpy as np

            
class Autoencoder_conv(nn.Module):
    def __init__(self, scales, depth, latent, colors, batch_norm=False):
        super().__init__()
             
        self.encoder = self._make_network(scales, depth, latent, colors, part='encoder', bn=batch_norm)
        self.decoder = self._make_network(scales, depth, latent, colors, part='decoder', bn=batch_norm)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    @staticmethod
    def _make_network(scales, depth, latent, colors, part=None, bn = True):
        """
        input:
        part - encoder/decoder, str
        
        Following the structure of reimplementation by authors paper at
        kylemcdonald/ACAI (PyTorch).ipynb
        """
        activation = nn.LeakyReLU(0.01) 
        
        sub_network = []
        
        if part == 'encoder':
            sub_network += [nn.Conv2d(colors, depth, 1, padding=1)]
            
            input_channels = depth
            transformation = nn.AvgPool2d(2)
            
        elif part == 'decoder':
            
            input_channels = latent
            transformation = nn.Upsample(scale_factor=2)
        
        # joint part
        for scale in range(scales):
            k = depth * np.power(2,scale)
                        
            if bn:
                sub_network.extend([nn.Conv2d(input_channels, k, 3, padding=1), nn.BatchNorm2d(k), activation, 
                                    transformation])
                
            else:
                sub_network.extend([nn.Conv2d(input_channels, k, 3, padding=1), activation,
                    transformation])
            
            input_channels = k
        
        if part == 'encoder':
            k = depth << scales
            sub_network.extend([nn.Conv2d(input_channels, k, 3, padding=1), activation, nn.Conv2d(k, latent, 3, padding=1)])
        
        elif part == 'decoder':
            sub_network.extend([nn.Conv2d(input_channels, depth, 3, padding=1), activation, nn.Conv2d(depth, colors, 3, padding=1)])
        
        
        # Same initialization as in paper
        slope = 0.2
        for layer in sub_network:
            if hasattr(layer, 'weight'):
                layer.weight.data.normal_(std=(1/((1 + slope**2) * np.prod(layer.weight.data.shape[:-1])))**2)

            if hasattr(layer, 'bias'):
                layer.bias.data.zero_()
        
        return nn.Sequential(*sub_network)    
    
    def track_gradient(self):
        def get_grads(block):
            cache = []
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    cache.append(torch.cat([layer.weight.grad.reshape(-1,1), layer.bias.grad.reshape(-1,1)]))
            return torch.cat(cache)
        
        return {'encoder_grads': torch.norm(get_grads(self.encoder)),
                'decoder_grads': torch.norm(get_grads(self.decoder))}



class AE_FC(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        #input_size is the size of the flatten 3d pose
        #input_dim are the dim of 3d pose
        self.input_size = kwargs['input_size']
        self.input_dim = kwargs['input_dim']
        self.fc_layer = kwargs['fc_layer'] 

        self.encoder = nn.Sequential(
            nn.Linear(kwargs['input_size'], self.fc_layer),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.fc_layer, kwargs['latent_size']),
            nn.BatchNorm1d(kwargs['latent_size']),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(kwargs['latent_size'], self.fc_layer),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.fc_layer, kwargs['input_size']),
            nn.BatchNorm1d(kwargs['input_size']),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.internals = {
            "reconstructed" : 0, 
            "code" : 0
        }

    
    def forward_encoder(self, features):
        features_flat = features.view(-1, self.input_size).float() ## How to make this always work
        code = self.encoder(features_flat)
        return code

    def forward_decoder(self, code):
        reconstructed_flat = self.decoder(code)
        reconstructed =  reconstructed_flat.view(-1, *self.input_dim)
        return reconstructed

    
    def forward(self, features):
        code = self.forward_encoder(features)
        self.internals["code"] = code
        self.internals["reconstructed"] = self.forward_decoder(code)
        return self.internals

### out = net(x) #reconstructiom
###

### encoder step: z1 = net.forward_encoder(x1); z2 = net.forward_encoder(x2)
### interpolation step: z = alpha * z1 + (1-alpha) * z2
### decoder step: out = net.forward_decoder(out)