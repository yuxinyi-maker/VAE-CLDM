
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add path
sys.path.append('/hy-tmp/VAEDDPM')

from ldm.models.autoencoder_unconditional import AutoencoderKL
from ldm.models.diffusion.numerical_conditional_diffusion import NumericalConditionalDiffusion
from ldm.modules.diffusionmodules.unet_numerical_conditional import ConditionalUNet3DNumerical
from ldm.data.numerical_rock3d_dataset import NumericalRocks3DDataModule


class NumericalConditionalDiffusionModel(pl.LightningModule):
    """
    Numerical conditional diffusion model
    Supports three conditions: porosity, pore size mean, pore size std
    Porosity is computed in real-time, pore size parameters are parsed from filenames
    """
    
    def __init__(self, 
                 vae_ckpt_path: str,
                 learning_rate: float = 5e-5,
                 condition_weight: float = 1.0,
                 output_dir: str = "./outputs"):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Initialize VAE (frozen)
        self.vae = AutoencoderKL(
            ddconfig={
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
                'spatial_dims': 3
            },
            lossconfig={
                'target': 'ldm.modules.losses.unconditional_loss.UnconditionalVAELoss',
                'params': {
                    'kl_weight': 1.0e-06,
                    'pixel_weight': 1.0,
                    'perceptual_weight': 1.0,
                    'disc_start': 10000,
                    'disc_weight': 0.05,
                    'disc_num_layers': 2,
                    'disc_in_channels': 1,
                    'use_actnorm': True,
                    'disc_loss': "hinge"
                }
            },
            embed_dim=4,
            monitor='val/rec_loss',
            ckpt_path=vae_ckpt_path,
            ignore_keys=[],
            colorize_nlabels=None
        )
        
        # Load checkpoint
        if vae_ckpt_path and os.path.exists(vae_ckpt_path):
            print(f"Loading VAE checkpoint: {vae_ckpt_path}")
            checkpoint = torch.load(vae_ckpt_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.vae.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.vae.load_state_dict(checkpoint, strict=False)
        
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print("VAE frozen")
        
        # Initialize diffusion model
        self.diffusion = NumericalConditionalDiffusion(
            first_stage_config={
                'target': 'ldm.models.autoencoder_unconditional.AutoencoderKL',
                'params': {
                    'ddconfig': {
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
                        'spatial_dims': 3
                    },
                    'lossconfig': {
                        'target': 'ldm.modules.losses.unconditional_loss.UnconditionalVAELoss',
                        'params': {
                            'kl_weight': 1.0e-06,
                            'pixel_weight': 1.0,
                            'perceptual_weight': 1.0,
                            'disc_start': 10000,
                            'disc_weight': 0.05,
                            'disc_num_layers': 2,
                            'disc_in_channels': 1,
                            'use_actnorm': True,
                            'disc_loss': "hinge"
                        }
                    },
                    'embed_dim': 4,
                    'monitor': 'val/rec_loss',
                    'ckpt_path': vae_ckpt_path,
                    'ignore_keys': [],
                    'colorize_nlabels': None
                }
            },
            cond_stage_config=None,
            num_timesteps_cond=1,
            cond_stage_key=None,
            cond_stage_trainable=False,
            conditioning_key=None,
            scale_factor=0.18215,
            # Three numerical conditions
            use_porosity_condition=True,
            porosity_dim=1,
            use_pore_size_mean_condition=True,
            pore_size_mean_dim=1,
            use_pore_size_std_condition=True,
            pore_size_std_dim=1,
            condition_fusion_dim=128,
            unconditional_mode=False,
            
            condition_weight=condition_weight  # Passed from __init__ parameter (default 30.0)
        )
        # Inject frozen VAE into diffusion module for latent space training
        self.diffusion.vae = self.vae
        self.diffusion.scale_factor = 0.18215
        print("Diffusion model initialized")
        
      
        
        print("Dynamic condition weight system is implemented in the diffusion module")
        
      
        self.grad_norm_threshold = 8.0
        self.grad_norm_patience = 15  
        self.adaptive_lr_decay = 0.5
        self.min_adaptive_lr = 3e-9
        self.min_condition_lr = None
    
    def training_step(self, batch, batch_idx):
        
        # Encode image to latent space
        with torch.no_grad():
            x = self.vae.get_input(batch, 'image')
            x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            posterior = self.vae.encode(x)
            z = posterior.sample()
        
          
        if torch.rand(1).item() < 0.0:  
            batch_cfg = {
                'image': batch['image'],  
                'porosity': torch.zeros_like(batch['porosity']),
                'pore_size_mean': torch.zeros_like(batch['pore_size_mean']),
                'pore_size_std': torch.zeros_like(batch['pore_size_std']),
                'filename': batch.get('filename', '')
            }
            
            # Use unconditional training
            loss, log_dict = self.diffusion.shared_step(batch_cfg, batch_idx, z=z)
            self.log('train/cfg_mode', 0.0, prog_bar=False, logger=True)  # Log unconditional mode
        else:
            # Normal conditional training
            loss, log_dict = self.diffusion.shared_step(batch, batch_idx, z=z)
            self.log('train/cfg_mode', 1.0, prog_bar=False, logger=True)  # Log conditional mode
        
      
        loss = loss * 0.18  
      
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        
        # Log loss
        self.log('train/loss', loss, prog_bar=True, logger=True)
        self.log('train/mse_loss', log_dict.get('mse_loss', 0), prog_bar=False, logger=True)
        self.log('train/condition_loss', log_dict.get('condition_loss', 0), prog_bar=False, logger=True)
        self.log('train/structure_loss', log_dict.get('structure_loss', 0), prog_bar=False, logger=True)
        self.log('train/condition_weight', log_dict.get('condition_weight', 0), prog_bar=False, logger=True)
        self.log('train/structure_weight', log_dict.get('structure_weight', 0), prog_bar=False, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Encode image to latent space
        with torch.no_grad():
            x = self.vae.get_input(batch, 'image')
            x = x * 2.0 - 1.0  
            posterior = self.vae.encode(x)
            z = posterior.sample()
        
        loss, log_dict = self.diffusion.shared_step(batch, batch_idx, z=z)

        if torch.isnan(loss) or torch.isinf(loss):
           
            return None
        
        # Log loss
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val/mse_loss', log_dict.get('mse_loss', 0), prog_bar=False, logger=True)
        self.log('val/condition_loss', log_dict.get('condition_loss', 0), prog_bar=False, logger=True)
        self.log('val/structure_loss', log_dict.get('structure_loss', 0), prog_bar=False, logger=True)
        self.log('val/condition_weight', log_dict.get('condition_weight', 0), prog_bar=False, logger=True)
        self.log('val/structure_weight', log_dict.get('structure_weight', 0), prog_bar=False, logger=True)
        
        return loss
    
    
    def configure_optimizers(self):
        
        base_lr = self.hparams.learning_rate
        
        # Separate condition embedding parameters from other parameters
        condition_embed_params = []
        other_params = []
        
        print("\n" + "="*80)
        print("Configuring optimizer - Scheme 3 (moderate version)")
        print("="*80)
        
        for name, param in self.diffusion.named_parameters():
  
            is_condition_embed = any(
                cond_name in name for cond_name in [
                    'porosity_proj',            
                    'porosity_embed',           
                    'pore_size_combined_embed', 
                    'film_gamma_net',           
                    'film_beta_net'            
                ]
            )
            
            if is_condition_embed:
                condition_embed_params.append(param)
              
            else:
                other_params.append(param)
        
     
        condition_lr_multiplier = 5.0  
        self.condition_lr_multiplier = condition_lr_multiplier
        self.min_condition_lr = self.min_adaptive_lr * self.condition_lr_multiplier
        
      
        param_groups = [
            {
                'params': other_params, 
                'lr': base_lr,
                'weight_decay': 0.01,
                'name': 'main_network'
            },
            {
                'params': condition_embed_params, 
                'lr': base_lr * condition_lr_multiplier,  
                'weight_decay': 0.0,   
                'name': 'condition_embeddings'
            }
        ]
        
     
        optimizer = torch.optim.AdamW(
            param_groups, 
            betas=(0.85, 0.999), 
            eps=1e-6,  
            amsgrad=True
        )
        

        actual_grad_clip = "to be read from trainer"
        
       
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=300,  
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
    
    def on_before_optimizer_step(self, optimizer):
        
        # Separate different types of parameters
        output_params = []
        critical_params = []
        
        for name, param in self.diffusion.model.named_parameters():
            if param.grad is None:
                continue
            
            if 'out.2.weight' in name or 'out.2.bias' in name:
                output_params.append(param)
            
            elif ('input_blocks.0.0' in name or 
                  ('output_blocks' in name and 'skip_connection' in name and 
                   int(name.split('.')[2]) >= 9)):  
                critical_params.append(param)
        
       
        clip_records = []
        
        if output_params:
            output_norm = torch.nn.utils.clip_grad_norm_(output_params, max_norm=0.1)
            if output_norm > 0.1:
                clip_records.append(f"output layer: {output_norm:.2f}->0.1")
        
        if critical_params:
            critical_norm = torch.nn.utils.clip_grad_norm_(critical_params, max_norm=0.5)
            if critical_norm > 0.5:
                clip_records.append(f"critical layer: {critical_norm:.2f}->0.5")
        
      
        total_norm = 0.0
        for p in self.diffusion.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        
        # Log gradient norm
        self.log('train/grad_norm', grad_norm, on_step=True, on_epoch=False, prog_bar=False)

       
        if not hasattr(self, 'grad_norm_history'):
            self.grad_norm_history = []
 
        self.grad_norm_history.append(grad_norm)
        patience = getattr(self, 'grad_norm_patience', 10)
        threshold = getattr(self, 'grad_norm_threshold', 8.0)

        if len(self.grad_norm_history) > patience:
            self.grad_norm_history.pop(0)

        if (
            len(self.grad_norm_history) == patience
            and all(gn > threshold for gn in self.grad_norm_history)
            and self.global_step > patience
        ):
            decay = getattr(self, 'adaptive_lr_decay', 0.5)
            base_min_lr = getattr(self, 'min_adaptive_lr', 0.0)
            condition_min_lr = getattr(self, 'min_condition_lr', base_min_lr)

            print(f"\nAdaptive LR fallback: high gradient {patience} consecutive times, reducing LR by x{decay}")
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                new_lr = current_lr * decay
                group_name = param_group.get('name', '')

                if group_name == 'condition_embeddings':
                    new_lr = max(new_lr, condition_min_lr)
                    if new_lr == condition_min_lr and current_lr == condition_min_lr:
                        print(f"   Condition embedding LR has reached minimum (>={condition_min_lr:.2e})")
                else:
                    new_lr = max(new_lr, base_min_lr)
                    if new_lr == base_min_lr and current_lr == base_min_lr:
                        print(f"   Backbone network LR has reached minimum (>={base_min_lr:.2e})")

                param_group['lr'] = new_lr

            self.grad_norm_history = []  
        
    
    def configure_callbacks(self):
        """Configure callbacks"""
        callbacks = []
        
        # Model checkpoint - save every 10 epochs
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.hparams.output_dir,
            filename='numerical_diffusion-{epoch:03d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=-1,  # Save all checkpoints
            save_last=True,
            every_n_epochs=10,  # Save every 10 epochs
            save_on_train_epoch_end=False,  # Ensure saving after validation
            enable_version_counter=False  
        )
        callbacks.append(checkpoint_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        return callbacks


def main(model_config=None):
    
    modelConfig = {
        "state": "train",  # "train" or "eval"
        "epoch": 300,      # Training epochs
        "batch_size": 4,   
        "learning_rate": 3e-8,  
        "device": "cuda:0",  # Device
        "grad_clip": 1.0,  
        "accumulate_grad_batches": 12,  
        "precision": 32,  
        
        # Data paths
        "train_data_path": "/hy-tmp/VAEDDPM/train_data_limestone",
        "val_data_path": "/hy-tmp/VAEDDPM/val_data_limestone",  # Temporarily using same data
        "vae_ckpt_path": "/hy-tmp/VAEDDPM/outputs/unconditionalVAE/checkpoints/improved/checkpoints/last.ckpt",

        
        # Output paths 
        "output_dir": "/hy-tmp/VAEDDPM/outputs/conditionaldiffusion/improved",  
        "log_dir": "/hy-tmp/VAEDDPM/logs/diffusion_logs/improved",
        
        
        "num_workers": 4,
        "condition_weight": 80.0,  
        
        # Generation parameters (used in eval mode)
        "num_samples": 4,
        "target_pore_size_mean": 1.0,
        "target_pore_size_std": 0.3,
        
        # Checkpoint
        "resume_ckpt": "/hy-tmp/VAEDDPM/outputs/conditionaldiffusion/improved/last.ckpt",  
        "test_load_weight": None,  # Weights to load for testing
    }
    
    # If custom config is provided, update
    if model_config is not None:
        modelConfig.update(model_config)
    
   
    
    # Create output directories
    os.makedirs(modelConfig['output_dir'], exist_ok=True)
    os.makedirs(modelConfig['log_dir'], exist_ok=True)
    
    if modelConfig["state"] == "train":
        # Training mode
        train_model(modelConfig)
    else:
        # Evaluation/generation mode
        eval_model(modelConfig)


def train_model(modelConfig):
    """Train model"""

    
  
    data_module = NumericalRocks3DDataModule(
        train_data_root=modelConfig['train_data_path'],
        val_data_root=modelConfig['val_data_path'],
        batch_size=modelConfig['batch_size'],
        num_workers=modelConfig['num_workers'],
        target_shape=(64, 64, 64),
      
        condition_ranges={
            'pore_size_mean': [0.0, 20.0],  # Pore size mean range
            'pore_size_std': [0.0, 20.0]    # Pore size std range
        },
        auto_generate_labels=True
    )
    print("Pore size normalization range set: [0, 20] -> [0, 1]")
    
    # 模型
    model = NumericalConditionalDiffusionModel(
        vae_ckpt_path=modelConfig['vae_ckpt_path'],
        learning_rate=modelConfig['learning_rate'],
        condition_weight=modelConfig['condition_weight'],
        output_dir=modelConfig['output_dir']
    )
    
    # 训练器
    trainer = pl.Trainer(
        max_epochs=modelConfig['epoch'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=modelConfig['precision'],
        gradient_clip_val=modelConfig['grad_clip'],
        accumulate_grad_batches=modelConfig['accumulate_grad_batches'],
        logger=TensorBoardLogger(modelConfig['log_dir'], name='numerical_diffusion'),
        callbacks=model.configure_callbacks(),
        log_every_n_steps=50,
        check_val_every_n_epoch=1  
    )
    
    # 开始训练
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader(), ckpt_path=modelConfig['resume_ckpt'])
    


if __name__ == '__main__':
    main()

