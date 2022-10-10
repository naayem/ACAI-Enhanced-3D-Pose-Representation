import torch
import torch.nn as nn
import torch.nn.functional as F

from .joints_loss import JointsLoss
from ..functional.interpolation_s import random_sinterpolation

class AcaiAeLoss(nn.Module):
    def __init__(self, criterion, ae_reg_coef = 0.5):
        super(AcaiAeLoss, self).__init__()
        self.joints_loss = JointsLoss(criterion)
        self.ae_reg_coef = ae_reg_coef

    def forward(self, x, reconstruction, reconstruction_interpolated, trainer):
        alpha_reconstruction = trainer.models.critic(reconstruction_interpolated) # d_w(X^HAT_ALPHA)
        zeros = torch.zeros(alpha_reconstruction.size(), device=trainer.device0)

        # Term1: Reconstruction loss
        # Term2: Trying to fool the Critic via Lowering it's predicted values on interpolated samples
        reconstruction_loss = self.joints_loss(reconstruction, x) # ||X-X^HAT||
        critic_fooling_loss = F.mse_loss(alpha_reconstruction, zeros) # ||d_w(X^HAT_ALPHA)||^2

        ae_loss = reconstruction_loss + self.ae_reg_coef * critic_fooling_loss # ||X-X^HAT|| + LAMBDA * ||d_w(X^HAT_ALPHA)||^2

        return ae_loss, reconstruction_loss, critic_fooling_loss, self.ae_reg_coef

class AcaiCriticLoss(nn.Module):
    def __init__(self, stabilizer_coef = 0.2):
        super(AcaiCriticLoss, self).__init__()
        self.stabilizer_coef = stabilizer_coef

    def forward(self, x, reconstruction, alpha, reconstruction_interpolated, trainer):
        alpha_reconstruction = trainer.models.critic(reconstruction_interpolated) # d_w(X^HAT_ALPHA)
        zeros = torch.zeros(alpha_reconstruction.size(), device=trainer.device0)
        # Term1: Critic is trying to guess actual alpha
        # Term2: Critic is trying to assing "high realistic score" to samples which are linear interpolations (in data spcae)
        #        of original images and their reconstructions. Thus we are trying to encode the information about real samples
        #        to help Critic to distinguish between original and interpolated samples. (REGULARIZATION, optional)
        #        In case if our AE is perfect, it is just the critic(X) -> 0, w.r.t. Critic parameters
        alpha_guessing_loss = F.mse_loss(alpha_reconstruction, alpha.flatten()) # ||d_w(X^HAT_ALPHA) -  ALPHA||^2
        realistic_loss = F.mse_loss(trainer.models.critic(self.stabilizer_coef * x + (1 - self.stabilizer_coef) * reconstruction), zeros) # ||d_w( GAMMA*X + (1-GAMMA)*X^HAT )||^2

        critic_loss = alpha_guessing_loss + realistic_loss # ||d_w(X^HAT_ALPHA) -  ALPHA||^2 + ||d_w( GAMMA*X + (1-GAMMA)*X^HAT )||^2

        return critic_loss, alpha_guessing_loss, realistic_loss

class AcaiOutputs(nn.Module):
    def __init__(self):
        super(AcaiOutputs, self).__init__()

    def forward(self, x, trainer):
        batch_size = x.size(0)

        # Randomzie interpolated coefficient alpha and divide by two to get interval [0,0.5]
        alpha = 0.5 * torch.rand((batch_size),1).to(trainer.gpus[0])

        # Constructs non-interpolated latent space and decoded input
        ae_outputs = trainer.models.ae(x)
        latent_code = ae_outputs['code'] # Z
        reconstruction = ae_outputs['reconstructed'] # X^HAT

        # Here we shift all objects in batch by 1
        shifted_index = torch.arange(0, batch_size)-1

        # Decode interpolated latent code and calculate Critic's predictions
        interpolated_code = alpha * latent_code + (1 - alpha) * latent_code[shifted_index] #interpolate i and N-i elements # Z_ALPHA 
        reconstruction_interpolated = trainer.models.ae(interpolated_code, part='decoder')['reconstructed'] # X^HAT_ALPHA

        return reconstruction, alpha, reconstruction_interpolated

class AcaiOutputsRandom(nn.Module):
    def __init__(self):
        super(AcaiOutputsRandom, self).__init__()

    def forward(self, x, y, trainer):

        # Constructs non-interpolated latent space and decoded input
        ae_outputs = trainer.models.ae(x)
        reconstruction = ae_outputs['reconstructed'] # X^HAT

        batch_interp1 = x
        batch_interp2 = y
        z1, _ = trainer.models.ae(batch_interp1).values()
        z2, _ = trainer.models.ae(batch_interp2).values()

        interpolated_code, alpha = random_sinterpolation(z1, z2)

        # Decode interpolated latent code
        reconstruction_interpolated = trainer.models.ae(interpolated_code, part='decoder')['reconstructed'] # X^HAT_ALPHA

        return reconstruction, alpha, reconstruction_interpolated