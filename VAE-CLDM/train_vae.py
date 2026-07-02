
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ldm.modules.diffusionmodules.model_new import Encoder, Decoder
from ldm.modules.distributions.distributions_new import DiagonalGaussianDistribution
from ldm.data.numerical_rock3d_dataset import NumericalRocks3DDataset


class ImprovedVAE(pl.LightningModule):
    
    
    def __init__(self, 
                 # VAE architecture parameters
                 ddconfig,
                 embed_dim=4,
                 # Training parameters
                 learning_rate=1e-4,
                 kl_weight=5e-8,
                 porosity_weight=2.0,
                 contrast_weight=1.0,
                 # Feature switches
                 use_porosity_constraint=True,
                 use_contrast_loss=True,
                 use_multiscale_porosity=True,
                 contrast_start_epoch=0,  
                 high_porosity_threshold=0.15,
                 high_porosity_focus_gamma=3.0,
                 high_porosity_boost_weight=4.0):
        super().__init__()
        
        self.save_hyperparameters()
        
        # ==================== VAE Components ====================
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        # Latent space parameters
        self.embed_dim = embed_dim
        self.z_channels = ddconfig.get("z_channels", embed_dim)
        
        # Quantization convolution layers
        self.quant_conv = nn.Conv3d(2*self.z_channels, 2*self.embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(self.embed_dim, self.z_channels, 1)
        
        # ==================== Training Parameters ====================
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.porosity_weight = porosity_weight
        self.contrast_weight = contrast_weight
        self.use_porosity_constraint = use_porosity_constraint
        self.use_contrast_loss = use_contrast_loss
        self.use_multiscale_porosity = use_multiscale_porosity
        self.contrast_start_epoch = contrast_start_epoch  
        self.high_porosity_threshold = high_porosity_threshold
        self.high_porosity_focus_gamma = high_porosity_focus_gamma
        self.high_porosity_boost_weight = high_porosity_boost_weight
        
        # ==================== Porosity Cache ====================
        self.porosity_cache = {
            'low': [],      # 0.06-0.10
            'medium': [],   # 0.10-0.14
            'high': []      # 0.14-0.20
        }
        self.latent_cache = {
            'low': [],
            'medium': [],
            'high': []
        }
        
        # Validation outputs
        self.validation_step_outputs = []
        
    
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
    
    def forward(self, x):
        """Forward pass"""
        posterior = self.encode(x)
        z = posterior.sample()
        dec = self.decode(z)
        return dec, posterior
    
    def compute_porosity(self, x, region=None):
        """Compute porosity"""
        # Detect range and normalize
        if x.min() < 0:
            x_normalized = (x + 1.0) / 2.0
        else:
            x_normalized = x
        
        # Region cropping
        if region is not None:
            d_s, d_e, h_s, h_e, w_s, w_e = region
            x_region = x_normalized[:, :, d_s:d_e, h_s:h_e, w_s:w_e]
        else:
            x_region = x_normalized
        
        # Binarize and compute
        binary = (x_region < 0.5).float()
        porosity = binary.mean(dim=[1, 2, 3, 4])
        
        return porosity
    
    def compute_multiscale_porosity(self, x):
        """Compute multi-scale porosity"""
        B, C, D, H, W = x.shape
        
        # Global porosity
        global_porosity = self.compute_porosity(x)
        
        # Local porosity (8 sub-blocks)
        local_porosities = []
        for d_start in [0, D//2]:
            for h_start in [0, H//2]:
                for w_start in [0, W//2]:
                    region = (
                        d_start, d_start + D//2,
                        h_start, h_start + H//2,
                        w_start, w_start + W//2
                    )
                    local_por = self.compute_porosity(x, region)
                    local_porosities.append(local_por)
        
        local_porosities = torch.stack(local_porosities, dim=1)  # [B, 8]
        
        return global_porosity, local_porosities
    
    def update_porosity_cache(self, porosities, latents):
        """Update porosity cache"""
        for i in range(len(porosities)):
            por = porosities[i].item()
            lat = latents[i].detach()
            
            # Classify
            if por < 0.10:
                category = 'low'
            elif por < 0.14:
                category = 'medium'
            else:
                category = 'high'
            
            # Add and limit size
            self.porosity_cache[category].append(por)
            self.latent_cache[category].append(lat)
            
            if len(self.porosity_cache[category]) > 32:
                self.porosity_cache[category].pop(0)
                self.latent_cache[category].pop(0)
    
    def compute_contrast_loss(self):
       
        min_samples = 2
        if (len(self.latent_cache['low']) < min_samples or
            len(self.latent_cache['medium']) < min_samples or
            len(self.latent_cache['high']) < min_samples):
            return torch.tensor(0.0, device=self.device)
        
     
        n_samples = min(
            min(len(self.latent_cache['low']), len(self.latent_cache['medium']), len(self.latent_cache['high'])),
            4  # max 4
        )
        
        # Random sampling
        low_latents = torch.stack([self.latent_cache['low'][i] 
                                   for i in np.random.choice(len(self.latent_cache['low']), n_samples, replace=False)])
        medium_latents = torch.stack([self.latent_cache['medium'][i]
                                     for i in np.random.choice(len(self.latent_cache['medium']), n_samples, replace=False)])
        high_latents = torch.stack([self.latent_cache['high'][i]
                                   for i in np.random.choice(len(self.latent_cache['high']), n_samples, replace=False)])
        
   
        low_vs_medium = F.mse_loss(low_latents.mean(dim=0), medium_latents.mean(dim=0))
        low_vs_high = F.mse_loss(low_latents.mean(dim=0), high_latents.mean(dim=0))
        medium_vs_high = F.mse_loss(medium_latents.mean(dim=0), high_latents.mean(dim=0))
        
       
        intra_low = torch.mean(torch.stack([F.mse_loss(low_latents[i], low_latents.mean(dim=0)) for i in range(n_samples)]))
        intra_medium = torch.mean(torch.stack([F.mse_loss(medium_latents[i], medium_latents.mean(dim=0)) for i in range(n_samples)]))
        intra_high = torch.mean(torch.stack([F.mse_loss(high_latents[i], high_latents.mean(dim=0)) for i in range(n_samples)]))
        
      
        inter_distance = (low_vs_medium + low_vs_high + medium_vs_high) / 3.0
        intra_distance = (intra_low + intra_medium + intra_high) / 3.0
        
     
        inter_distance = torch.clamp(inter_distance, min=1e-6, max=10.0)  
        intra_distance = torch.clamp(intra_distance, min=1e-6, max=10.0)
        
     
        ratio = inter_distance / (intra_distance + 1e-8)
        ratio = torch.clamp(ratio, min=1e-6, max=100.0)
        
        contrast_loss = -torch.log(ratio + 1e-8)
        
      
        contrast_loss = torch.clamp(contrast_loss, min=-10.0, max=10.0)
        
        return contrast_loss
    
    def compute_losses(self, x, reconstructions, posterior, latents, prefix='train'):
  
        batch_size = x.size(0)
        device = x.device

        # === Global porosity statistics ===
        orig_porosity = self.compute_porosity(x)
        recon_porosity = self.compute_porosity(reconstructions)
        focus_strength = torch.relu(orig_porosity - self.high_porosity_threshold)
        denom = max(1.0 - self.high_porosity_threshold, 1e-6)
        focus_strength = focus_strength / denom

        # 1. Reconstruction loss
        rec_error = F.mse_loss(reconstructions, x, reduction='none')
        rec_per_sample = rec_error.view(batch_size, -1).mean(dim=1)
        if self.high_porosity_focus_gamma > 0:
            rec_weights = 1.0 + self.high_porosity_focus_gamma * focus_strength
            rec_loss = (rec_per_sample * rec_weights).mean()
        else:
            rec_loss = rec_per_sample.mean()

        # 2. KL loss
        kl_loss = posterior.kl().mean()

        # 3. Porosity loss (optional multi-scale)
        porosity_loss = 0.0
        multiscale_loss = 0.0
        if self.use_porosity_constraint:
            if self.use_multiscale_porosity:
                _, orig_local = self.compute_multiscale_porosity(x)
                _, recon_local = self.compute_multiscale_porosity(reconstructions)
                porosity_loss = F.mse_loss(recon_porosity, orig_porosity)
                multiscale_loss = F.mse_loss(recon_local, orig_local)
            else:
                porosity_loss = F.mse_loss(recon_porosity, orig_porosity)

        # 4. High porosity deficit penalty
        high_porosity_boost = torch.zeros(1, device=device).squeeze()
        if self.high_porosity_boost_weight > 0:
            deficit = torch.relu(orig_porosity - recon_porosity)
            high_porosity_boost = (deficit * focus_strength).mean()

        # 5. Contrastive loss
        contrast_loss = 0.0
        if self.use_contrast_loss and prefix == 'train':
            if self.current_epoch >= self.contrast_start_epoch:
                contrast_loss = self.compute_contrast_loss()

        # 6. Total loss
        total_loss = (
            rec_loss
            + self.kl_weight * kl_loss
            + self.porosity_weight * porosity_loss
            + 0.5 * multiscale_loss
            + self.contrast_weight * contrast_loss
            + self.high_porosity_boost_weight * high_porosity_boost
        )

      
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\nWarning: NaN/Inf detected!")
            print(f"  rec_loss: {rec_loss.item()}")
            print(f"  kl_loss: {kl_loss.item()}")
            print(f"  porosity_loss: {porosity_loss if isinstance(porosity_loss, float) else porosity_loss.item()}")
            print(f"  multiscale_loss: {multiscale_loss if isinstance(multiscale_loss, float) else multiscale_loss.item()}")
            print(f"  contrast_loss: {contrast_loss if isinstance(contrast_loss, float) else contrast_loss.item()}")
            print(f"  high_porosity_boost: {high_porosity_boost.item() if isinstance(high_porosity_boost, torch.Tensor) else high_porosity_boost}")
            total_loss = rec_loss.clone()

        # Logging
        log_dict = {
            f'{prefix}/total_loss': total_loss.detach(),
            f'{prefix}/rec_loss': rec_loss.detach(),
            f'{prefix}/kl_loss': kl_loss.detach(),
            f'{prefix}/porosity_loss': porosity_loss.detach() if isinstance(porosity_loss, torch.Tensor) else porosity_loss,
            f'{prefix}/multiscale_loss': multiscale_loss.detach() if isinstance(multiscale_loss, torch.Tensor) else multiscale_loss,
            f'{prefix}/contrast_loss': contrast_loss.detach() if isinstance(contrast_loss, torch.Tensor) else contrast_loss,
            f'{prefix}/high_porosity_boost': high_porosity_boost.detach() if isinstance(high_porosity_boost, torch.Tensor) else high_porosity_boost,
            f'{prefix}/high_focus_strength': focus_strength.mean().detach(),
        }

        # Porosity details
        if self.use_porosity_constraint:
            log_dict.update({
                f'{prefix}/original_porosity': orig_porosity.mean().detach(),
                f'{prefix}/reconstructed_porosity': recon_porosity.mean().detach(),
                f'{prefix}/porosity_error': torch.abs(recon_porosity - orig_porosity).mean().detach(),
            })

        return total_loss, log_dict
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x = batch['image']
        
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        # Map to [-1, 1]
        x = x * 2.0 - 1.0
        
        # Forward pass
        posterior = self.encode(x)
        z = posterior.sample()
        reconstructions = self.decode(z)
        
        # Update cache
        with torch.no_grad():
            porosities = self.compute_porosity(x)
            self.update_porosity_cache(porosities, z)
        
        # Compute losses
        total_loss, log_dict = self.compute_losses(x, reconstructions, posterior, z, prefix='train')
        
        # Log
        for key, value in log_dict.items():
            self.log(key, value, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        # Print
        if batch_idx % 100 == 0:
            print(f"\n  Step {self.global_step}:")
            print(f"     Total loss: {total_loss.item():.6f}")
            print(f"     Rec loss: {log_dict['train/rec_loss']:.6f}")
            print(f"     Contrast loss: {log_dict['train/contrast_loss']:.6f}")
            print(f"     High porosity focus weight: {float(log_dict['train/high_focus_strength']):.4f}")
            print(f"     High porosity boost: {float(log_dict['train/high_porosity_boost']):.6f}")
            if self.use_porosity_constraint:
                print(f"     Porosity error: {log_dict['train/porosity_error']:.4f}")
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x = batch['image']
        
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        x = x * 2.0 - 1.0
        
        posterior = self.encode(x)
        z = posterior.sample()
        reconstructions = self.decode(z)
        
        total_loss, log_dict = self.compute_losses(x, reconstructions, posterior, z, prefix='val')
        
        for key, value in log_dict.items():
            self.log(key, value, prog_bar=True, logger=True, on_epoch=True)
        
        self.validation_step_outputs.append(log_dict)
        
        return log_dict
    
    def on_validation_epoch_end(self):
        """Validation epoch end"""
        if not self.validation_step_outputs:
            return
        
        # Compute averages
        avg_metrics = {}
        for key in self.validation_step_outputs[0].keys():
            values = [x[key] for x in self.validation_step_outputs if key in x]
            if values:
              
                if isinstance(values[0], torch.Tensor):
                    avg_metrics[key] = torch.stack(values).mean()
                else:
                    avg_metrics[key] = sum(values) / len(values)
        
        # Print
        print("\n" + "="*80)
        print(f"Epoch {self.current_epoch} Validation Summary:")
        print("="*80)
        print(f"  Total loss: {avg_metrics.get('val/total_loss', 0):.6f}")
        print(f"  Rec loss: {avg_metrics.get('val/rec_loss', 0):.6f}")
        
        if self.use_porosity_constraint:
            porosity_error = avg_metrics.get('val/porosity_error', 0)
            print(f"  Porosity error: {porosity_error:.4f}")
            
            if porosity_error < 0.02:
                print("  Porosity reconstruction: Excellent")
            elif porosity_error < 0.05:
                print("  Porosity reconstruction: Good")
            elif porosity_error < 0.10:
                print("  Porosity reconstruction: Fair")
            else:
                print("  Porosity reconstruction: Poor")
        
        print("="*80 + "\n")
        
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


class DecoderSensitivityCallback(Callback):
    """Decoder sensitivity monitoring"""
    
    def __init__(self, test_every_n_epochs=10):
        super().__init__()
        self.test_every_n_epochs = test_every_n_epochs
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Test at end of validation epoch"""
        if (trainer.current_epoch + 1) % self.test_every_n_epochs != 0:
            return
        
        print("\n" + "=="*40)
        print(f"Epoch {trainer.current_epoch} Decoder Sensitivity Test")
        print("=="*40)
        
        device = pl_module.device
        
        # Test latent
        torch.manual_seed(42)
        base_latent = torch.randn(1, 4, 8, 8, 8, device=device)
        
        scale_factors = [0.5, 1.0, 1.5, 2.0]
        porosities = []
        
        with torch.no_grad():
            for scale in scale_factors:
                scaled_latent = base_latent * scale
                decoded = pl_module.decode(scaled_latent)
                
                # Compute porosity
                decoded_np = decoded.cpu().numpy()[0, 0]
                normalized = (decoded_np + 1.0) / 2.0
                binary = (normalized < 0.5).astype(np.uint8)
                porosity = binary.mean()
                
                porosities.append(porosity)
                print(f"  Latent scale {scale:.1f}x -> Porosity {porosity:.4f}")
        
        # Sensitivity
        por_range = max(porosities) - min(porosities)
        print(f"\n  Porosity variation range: {por_range:.4f} ({por_range*100:.1f} percentage points)")
        
        if por_range < 0.05:
            print(f"  Sensitivity: Very low (<5%)")
        elif por_range < 0.10:
            print(f"  Sensitivity: Low (5-10%)")
        elif por_range < 0.20:
            print(f"  Sensitivity: Moderate (10-20%)")
        else:
            print(f"  Sensitivity: Good (>20%)")
        
        print("=="*40 + "\n")


def main():
    """Main training function"""
    
    config = {
        # Training parameters
        "epochs": 180,
        "batch_size": 2, 
        "accumulate_grad_batches": 4,  
        "learning_rate": 1e-4,
        "num_workers": 4,
        
        # VAE parameters
        "kl_weight": 1.2e-8,
        "porosity_weight": 2.0,
        "contrast_weight": 1.1,  
        "use_porosity_constraint": True,
        "use_contrast_loss": True,
        "use_multiscale_porosity": False,  
        "contrast_start_epoch": 5,  
        "high_porosity_threshold": 0.15,
        "high_porosity_focus_gamma": 4.0,
        "high_porosity_boost_weight": 6.0,
        
        # Data paths
        "train_data_dir": "/hy-tmp/VAEDDPM/train_data_limestone",
        "val_data_dir": "/hy-tmp/VAEDDPM/val_data_limestone",
        
        # Checkpoint resume
        "resume_from_checkpoint": "/hy-tmp/VAEDDPM/outputs/unconditionalVAE/checkpoints/10.ckpt",
        
        # Output paths
        "output_dir": "/hy-tmp/VAEDDPM/outputs/unconditionalVAE/checkpoints",
        "log_dir": "/hy-tmp/VAEDDPM/logs/vae_logs",
        
        # VAE architecture
        "ddconfig": {
            'double_z': True,
            'z_channels': 4,
            'resolution': 64,
            'in_channels': 1,
            'out_ch': 1,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0,
            'tanh_out': True,
            'spatial_dims': 3,
        },
        "embed_dim": 4,
    }
    
  
    
    # Create directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Load data
    print("Loading datasets...")
    
    train_dataset = NumericalRocks3DDataset(
        data_root=config['train_data_dir'],
        target_shape=(64, 64, 64),
    )
    
    val_dataset = NumericalRocks3DDataset(
        data_root=config['val_data_dir'],
        target_shape=(64, 64, 64),
    )
    
    print(f"Training set: {len(train_dataset)}")
    print(f"Validation set: {len(val_dataset)}\n")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    print("Creating improved VAE...")
    
    model = ImprovedVAE(
        ddconfig=config['ddconfig'],
        embed_dim=config['embed_dim'],
        learning_rate=config['learning_rate'],
        kl_weight=config['kl_weight'],
        porosity_weight=config['porosity_weight'],
        contrast_weight=config['contrast_weight'],
        use_porosity_constraint=config['use_porosity_constraint'],
        use_contrast_loss=config['use_contrast_loss'],
        use_multiscale_porosity=config['use_multiscale_porosity'],
        contrast_start_epoch=config.get('contrast_start_epoch', 0), 
        high_porosity_threshold=config['high_porosity_threshold'],
        high_porosity_focus_gamma=config['high_porosity_focus_gamma'],
        high_porosity_boost_weight=config['high_porosity_boost_weight'],
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['output_dir'], 'checkpoints'),
        filename='vae-{epoch:03d}-{val/total_loss:.4f}',
        save_top_k=3,
        monitor='val/total_loss',
        mode='min',
        save_last=True,
        every_n_epochs=10,  
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    sensitivity_callback = DecoderSensitivityCallback(test_every_n_epochs=10)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config['log_dir'],
        name='vae_training'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, sensitivity_callback],
        log_every_n_steps=10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
        precision='16-mixed',
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),  
    )
    
    # Start training
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=config['resume_from_checkpoint']
    )
    
  
    print("Complete!")
  
if __name__ == "__main__":
    main()

