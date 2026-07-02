

import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryVAELoss(nn.Module):
   
    def __init__(self, kl_weight=1e-4, rec_weight=1.0, free_bits=0.1):
        
        super().__init__()
        self.kl_weight = kl_weight
        self.rec_weight = rec_weight
        self.free_bits = free_bits
        
    def forward(self, inputs, reconstructions, posterior, global_step=0, split="train"):
        
        inputs_binary = (inputs + 1) / 2  # [-1,1] -> [0,1]
        
        # Constrain reconstruction data range for numerical stability
        reconstructions = torch.clamp(reconstructions, -1, 1)
        rec_binary = (reconstructions + 1) / 2  # [-1,1] -> [0,1]
        
        # Binary cross-entropy loss - best suited for 0/1 binary data
        bce_loss = F.binary_cross_entropy(
            rec_binary, 
            inputs_binary, 
            reduction='mean'
        )
        
        # ==================== KL loss computation ====================
        kl_loss = posterior.kl()
        kl_loss = torch.mean(kl_loss)
        
        # Free-bits mechanism: ensure KL loss does not fall below minimum, preventing KL collapse
        if self.free_bits > 0:
            kl_loss = torch.max(kl_loss, torch.tensor(self.free_bits, device=kl_loss.device))
        
        # Mild KL loss upper bound to prevent gradient explosion
        kl_loss = torch.clamp(kl_loss, max=10.0)
        
        # ==================== KL weight warmup ====================
        # Progressively increase KL weight so the model learns reconstruction first, then latent space distribution
        if global_step < 1000:
            # Linear warmup: gradually increase from 0 to target KL weight over the first 1000 steps
            current_kl_weight = self.kl_weight * (global_step / 1000)
        else:
            current_kl_weight = self.kl_weight
            
        # ==================== Total loss computation ====================
        total_loss = self.rec_weight * bce_loss + current_kl_weight * kl_loss
        
        # ==================== Loss logging ====================
        log_dict = {
            f"{split}/total_loss": total_loss.detach(),
            f"{split}/bce_loss": bce_loss.detach(),        # binary cross-entropy loss
            f"{split}/kl_loss": kl_loss.detach(),          # KL divergence loss
            f"{split}/kl_weight": torch.tensor(current_kl_weight),  # current KL weight
            f"{split}/latent_mean": posterior.mean.mean().detach(), # latent space mean
            f"{split}/latent_std": posterior.std.mean().detach(),   # latent space std
        }
        
        return total_loss, log_dict