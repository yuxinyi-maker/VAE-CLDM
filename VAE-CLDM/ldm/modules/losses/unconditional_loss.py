"""
Unconditional VAE loss function
Specifically for unconditional VAE training, simplified loss computation

Core functions:
1. Reconstruction loss(L1)
2. KL regularization loss
3. Perceptual loss
4. Remove all condition handling and discriminator logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnconditionalVAELoss(nn.Module):
    """
    Unconditional VAE loss function
    VAE,Remove all condition handling and discriminator logic
    
    Loss components:
    1. Reconstruction loss(L1):ensure reconstruction quality
    2. KL loss:regularize latent space
    3. Perceptual loss:improve perceptual quality
    """
    
    def __init__(self, kl_weight=1e-6, pixel_weight=1.0, perceptual_weight=1.0,
                 disc_start=10000, disc_weight=0.0, disc_num_layers=3,
                 disc_in_channels=1, use_actnorm=True, disc_loss="hinge"):
        super().__init__()
        
        self.kl_weight = kl_weight
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = 0.0  # Unconditional VAE does not use a discriminator
        self.disc_start = disc_start
        
        # Pixel loss
        self.pixel_loss = nn.L1Loss()
        
        # Perceptual loss(simplified version)
        self.perceptual_loss = self._simple_perceptual_loss
        
        # Unconditional VAE does not need a discriminator
        self.discriminator = None
    
    def _simple_perceptual_loss(self, inputs, reconstructions):
        """Perceptual loss"""
        return torch.abs(inputs - reconstructions).mean()
    
    def forward(self, inputs, reconstructions, posterior=None, optimizer_idx=0,
                global_step=0, split="train", **kwargs):
        """
        Forward pass
        
        Args:
            inputs: input data
            reconstructions: reconstruction data
            posterior: posterior distribution(optional)
            optimizer_idx: optimizer index(0=generator,1=discriminator)
            global_step: global step
            split: training/validation split
        """
        
        # VAE,generator
        return self._generator_loss(inputs, reconstructions, posterior, global_step, split)
    
    def _generator_loss(self, inputs, reconstructions, posterior, global_step, split):
        """generator - VAEsimplified version"""
        # Reconstruction loss
        rec_loss = self.pixel_loss(reconstructions, inputs)
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
        else:
            p_loss = torch.tensor(0.0, device=inputs.device)
        
        # KL loss
        kl_loss = torch.tensor(0.0, device=inputs.device)
        if posterior is not None:
            try:
                kl_loss = posterior.kl()
                kl_loss = torch.clamp(kl_loss, min=1e-8)
            except Exception as e:
                print(f"KL loss: {e}")
                kl_loss = torch.tensor(1e-4, device=inputs.device)
        
        # Unconditional VAE does not use a discriminator
        g_loss = torch.tensor(0.0, device=inputs.device)
        
        # Total loss
        total_loss = (self.pixel_weight * rec_loss + 
                     self.perceptual_weight * p_loss + 
                     self.kl_weight * kl_loss)
        
        # Logging
        log = {
            f"{split}/total_loss": total_loss.detach(),
            f"{split}/rec_loss": rec_loss.detach(),
            f"{split}/kl_loss": kl_loss.detach(),
        }
        
        if self.perceptual_weight > 0:
            log[f"{split}/perceptual_loss"] = p_loss.detach()
        
        return total_loss, log


# backward compatible
LPIPSWithDiscriminator3D_Unconditional = UnconditionalVAELoss
