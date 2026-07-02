
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import numpy as np

# Import custom modules
from ldm.modules.diffusionmodules.model_new import Encoder, Decoder
from ldm.modules.distributions.distributions_new import DiagonalGaussianDistribution
from ldm.util_new import instantiate_from_config

import torch.nn as nn
from einops import rearrange


class AutoencoderKL(pl.LightningModule):
    
    def __init__(self, ddconfig, lossconfig, embed_dim, ckpt_path=None, 
                 ignore_keys=[], data_key="image", colorize_nlabels=None,
                 monitor=None, is_3d=True, learning_rate=2e-5, 
                 kl_weight=1e-6, latent_clip_value=3.0):
        super().__init__()
        
        # ==================== Basic Config ====================
        self.data_key = data_key
        self.is_3d = is_3d
        self.learning_rate = learning_rate
        
        # ==================== Loss Weight Config ====================
        self.kl_weight = kl_weight
        self.latent_clip_value = latent_clip_value
        
        # ==================== Network Component Initialization ====================
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        
        # Validation step output collection
        self.validation_step_outputs = []
        
        # Compute latent space parameters
        self.embed_dim = embed_dim
        self.z_channels = ddconfig.get("z_channels", embed_dim)
        
        # Instantiate distribution
        self.quant_conv = nn.Conv3d(2*self.z_channels, 2*self.embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(self.embed_dim, self.z_channels, 1)
        
        # Load checkpoint
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        """Initialize from checkpoint"""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def encode(self, x):
        """Encode to latent space"""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        """Decode from latent space"""
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def forward(self, input, sample_posterior=True):
        """Forward pass"""
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    
    def get_input(self, batch, k):
        """Get input data"""
        x = batch[k]
        
        if self.is_3d:
            # 3D data processing: (B, D, H, W) -> (B, 1, D, H, W)
            if len(x.shape) == 4:
                x = x.unsqueeze(1)  # Add channel dimension
            x = x.to(memory_format=torch.contiguous_format).float()
        else:
            # 2D data processing: (B, H, W) -> (B, 1, H, W)
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        
        return x
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        inputs = self.get_input(batch, self.data_key)
        reconstructions, posterior = self.forward(inputs)
        
        # Compute loss
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, split="train")
        
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        return aeloss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        try:
            inputs = self.get_input(batch, self.data_key)
            reconstructions, posterior = self.forward(inputs)
            
            # Compute loss
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, split="val")
            
            # Compute reconstruction loss (MSE)
            rec_loss = F.mse_loss(reconstructions, inputs, reduction='mean')
            
            # Log all losses
            self.log("val_rec_loss", rec_loss, prog_bar=True, logger=True, on_epoch=True)
            self.log("val_aeloss", aeloss, prog_bar=True, logger=True, on_epoch=True)
            
            # Safely log other metrics
            for key, value in log_dict_ae.items():
                self.log(f"val_{key}", value, logger=True, on_epoch=True)
            
            return log_dict_ae
            
        except Exception as e:
            print(f"Validation step error: {e}")
            # Return default loss value
            return {"val_loss": 0.0}
    
    def configure_optimizers(self):
        """Configure optimizers"""
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.encoder.parameters())+
                               list(self.decoder.parameters())+
                               list(self.quant_conv.parameters())+
                               list(self.post_quant_conv.parameters()),
                               lr=lr, betas=(0.5, 0.9))
        return opt
