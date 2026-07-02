
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union
import pytorch_lightning as pl
from ldm.modules.diffusionmodules.unet_numerical_conditional import ConditionalUNet3DNumerical


class NumericalConditionalDiffusion(pl.LightningModule):
    """
    Numerical conditional diffusion model - based on MCDDPM
    Supports three numerical conditions: porosity, pore size mean, pore size std
    Porosity is computed in real-time, pore size parameters are parsed from filenames
    """
    
    def __init__(self, 
                 first_stage_config: Dict,
                 cond_stage_config: Optional[Dict] = None,
                 num_timesteps_cond: int = 1,
                 cond_stage_key: str = None,
                 cond_stage_trainable: bool = False,
                 concat_mode: bool = True,
                 cond_stage_forward: Optional[callable] = None,
                 conditioning_key: Optional[str] = None,
                 scale_factor: float = 0.18215,
                 scale_by_std: bool = False,
                 *args, **kwargs):
        
        # Numerical condition parameters (based on MCDDPM, using three conditions)
        self.use_porosity_condition = kwargs.pop('use_porosity_condition', True)
        self.porosity_dim = kwargs.pop('porosity_dim', 1)
        self.use_pore_size_mean_condition = kwargs.pop('use_pore_size_mean_condition', True)
        self.pore_size_mean_dim = kwargs.pop('pore_size_mean_dim', 1)
        self.use_pore_size_std_condition = kwargs.pop('use_pore_size_std_condition', True)
        self.pore_size_std_dim = kwargs.pop('pore_size_std_dim', 1)
        self.condition_fusion_dim = kwargs.pop('condition_fusion_dim', 128)
        self.unconditional_mode = kwargs.pop('unconditional_mode', False)
        
        # Call parent class initialization
        super().__init__()
        
        # Basic parameters
        self.first_stage_config = first_stage_config
        self.condition_weight = kwargs.pop('condition_weight', 1.0)
        self.learning_rate = kwargs.pop('learning_rate', 1e-4)
        self.parameterization = kwargs.pop('parameterization', 'eps')
        self.num_timesteps = kwargs.pop('num_timesteps', 1000)
        self.cond_stage_config = cond_stage_config
        self.num_timesteps_cond = num_timesteps_cond
        self.cond_stage_key = cond_stage_key
        self.cond_stage_trainable = cond_stage_trainable
        self.concat_mode = concat_mode
        self.cond_stage_forward = cond_stage_forward
        self.conditioning_key = conditioning_key
        self.scale_factor = scale_factor
        self.scale_by_std = scale_by_std
        
        
        self.porosity_range_weights = {
            (0.06, 0.08): 8.0,   
            (0.08, 0.10): 4.0,  
            (0.10, 0.12): 0.8,   
            (0.12, 0.14): 0.4,   
            (0.14, 0.16): 0.4,   
            (0.16, 0.18): 2.5,   
        }
        print("Data distribution adaptive weighting enabled:")
        for (min_p, max_p), weight in self.porosity_range_weights.items():
            print(f"   Porosity {min_p:.2f}-{max_p:.2f}: weight x{weight:.1f}")
        
       
        # Support multi-stage training strategy, prioritize different conditions at different epochs
        self.use_dynamic_condition_weights = True
        self.current_train_epoch = 0
        print("Dynamic condition weight system enabled (5-stage training strategy)")
        
      
        self.register_schedule(num_timesteps=self.num_timesteps)
        
       
        from ldm.modules.losses.structure_similarity_loss import CombinedStructureLoss
        self.structure_loss = CombinedStructureLoss(
            ssim_weight=0.5,          # SSIM weight
            gradient_weight=0.3,      # Gradient weight
            connectivity_weight=0.2,  # Connectivity weight
            frequency_weight=0.1,     # Frequency domain weight
            use_ssim=True,
            use_gradient=True,
            use_connectivity=True,
            use_frequency=False       
        )
        self.structure_loss_weight = 0.3  # Structure loss total weight
        self.use_structure_loss = False  
        print("Structure similarity loss temporarily disabled (Epoch 104 NaN emergency fix)")
        
       
        self.model = ConditionalUNet3DNumerical(
            in_channels=4,  # VAE latent space channels
            out_channels=4,  # VAE latent space channels
            model_channels=128,  
            attention_resolutions=[],  
            dropout=0.0,
            channel_mult=[1, 2, 4, 4],
            conv_resample=True,
            dims=3,
            num_classes=None,
            use_checkpoint=True,
            num_heads=4,
            num_head_channels=64,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            # Numerical condition parameters
            use_porosity_condition=self.use_porosity_condition,
            porosity_dim=self.porosity_dim,
            use_pore_size_mean_condition=self.use_pore_size_mean_condition,
            pore_size_mean_dim=self.pore_size_mean_dim,
            use_pore_size_std_condition=self.use_pore_size_std_condition,
            pore_size_std_dim=self.pore_size_std_dim,
        )
        
        print(f"Numerical conditional diffusion model initialized")
        print(f"   Porosity condition: {'enabled' if self.use_porosity_condition else 'disabled'}")
        print(f"   Pore size mean condition: {'enabled' if self.use_pore_size_mean_condition else 'disabled'}")
        print(f"   Pore size std condition: {'enabled' if self.use_pore_size_std_condition else 'disabled'}")
    
    def register_schedule(self, num_timesteps=1000, beta_schedule='linear', 
                         linear_start=1e-4, linear_end=2e-2):
        """
        Register noise schedule parameters
        """
        # Generate beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(linear_start, linear_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == 'cosine':
            # Cosine schedule
            timesteps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
            alphas_cumprod = torch.cos((timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # Register as buffer (not optimized, but saved to checkpoint)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Used for q_sample
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Used for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # Used for log
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
    
    def get_condition_inputs(self, batch: Dict) -> Dict:
        """
        Extract numerical condition inputs from batch
        Based on MCDDPM condition processing
        """
        conditions = {}
        
        # Extract porosity condition
        if self.use_porosity_condition and 'porosity' in batch:
            porosity = batch['porosity']
            if isinstance(porosity, torch.Tensor):
                conditions['porosity_condition'] = porosity
            else:
                conditions['porosity_condition'] = torch.tensor([porosity], dtype=torch.float32)
        
        # Extract pore size mean condition
        if self.use_pore_size_mean_condition and 'pore_size_mean' in batch:
            pore_size_mean = batch['pore_size_mean']
            if isinstance(pore_size_mean, torch.Tensor):
                conditions['pore_size_mean_condition'] = pore_size_mean
            else:
                conditions['pore_size_mean_condition'] = torch.tensor([pore_size_mean], dtype=torch.float32)
        
        # Extract pore size std condition
        if self.use_pore_size_std_condition and 'pore_size_std' in batch:
            pore_size_std = batch['pore_size_std']
            if isinstance(pore_size_std, torch.Tensor):
                conditions['pore_size_std_condition'] = pore_size_std
            else:
                conditions['pore_size_std_condition'] = torch.tensor([pore_size_std], dtype=torch.float32)
        
        return conditions
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: add noise
        Based on DDPM noise schedule
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Use precomputed noise schedule parameters
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device)
        
        # Ensure dimension match [B] -> [B, 1, 1, 1, 1]
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def apply_model(self, x_noisy: torch.Tensor, t: torch.Tensor, 
                   cond: Optional[Dict] = None, **kwargs) -> torch.Tensor:
       
        if cond is None:
            cond = {}
        
      
        # If input is raw batch format, conversion needed
        if 'porosity' in cond and 'porosity_condition' not in cond:
            # Raw format, needs conversion
            condition_inputs = self.get_condition_inputs(cond)
        else:
            # Already processed format, use directly
            condition_inputs = cond
        
        # Call numerical conditional UNet
        return self.model(
            x_noisy, 
            t, 
            porosity_condition=condition_inputs.get('porosity_condition'),
            pore_size_mean_condition=condition_inputs.get('pore_size_mean_condition'),
            pore_size_std_condition=condition_inputs.get('pore_size_std_condition'),
            **kwargs
        )
    
   
    
    def sample(self, cond: Optional[Dict] = None, batch_size: int = 16, 
              return_intermediates: bool = False, x_T: Optional[torch.Tensor] = None,
              ddim_use_original_steps: bool = False, eta: float = 0.0,
              verbose: bool = True, quantize_denoised: bool = False,
              mask: Optional[torch.Tensor] = None, x0: Optional[torch.Tensor] = None,
              temperature: float = 1.0, noise_dropout: float = 0.0,
              score_corrector: Optional[callable] = None,
              corrector_kwargs: Optional[Dict] = None) -> Union[torch.Tensor, tuple]:
       
        if cond is None:
            cond = {}
        
        # Ensure condition input format is correct
        condition_inputs = self.get_condition_inputs(cond)
        
        # Call parent class sampling method
        return super().sample(
            cond=cond,
            batch_size=batch_size,
            return_intermediates=return_intermediates,
            x_T=x_T,
            ddim_use_original_steps=ddim_use_original_steps,
            eta=eta,
            verbose=verbose,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
            temperature=temperature,
            noise_dropout=noise_dropout,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs
        )
    
    def log_images(self, batch: Dict, N: int = 8, n_row: int = 2, 
                   sample: bool = True, ddim_steps: int = 200, 
                   ddim_eta: float = 0.0, return_keys: list = None,
                   quantize_denoised: bool = True, inpaint: bool = False,
                   plot_denoise_rows: bool = False, plot_progressive_rows: bool = True,
                   plot_diffusion_rows: bool = True, **kwargs) -> Dict:
        """
        Log images for monitoring
        """
        use_ddim = ddim_steps is not None
        
        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                         return_first_stage_outputs=True,
                                         force_c_encode=True,
                                         return_original_cond=True,
                                         bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["conditioning_rec"] = self.cond_stage_model.decode(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = torch.clamp((diffusion_grid + 1.0) / 2.0, min=0.0, max=1.0)

            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                       ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not self.decode_first_stage.encoder.quantize.legacy:
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                           ddim_steps=ddim_steps,eta=ddim_eta,
                                                           quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                    quantize_denoised=True)
                x_samples = self.decode_first_stage(samples)
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                h, w = z.shape[-2:]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h//4:3*h//4, w//4:3*w//4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):
                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples)
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples)
                log["samples_outpainting"] = x_samples

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
    
    def get_input(self, batch, k):
        """Get input data"""
        x = batch[k]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # Add channel dimension
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def shared_step(self, batch, batch_idx, z: Optional[torch.Tensor] = None):
        """
        Shared training/validation step
        Based on MCDDPM numerical conditional diffusion training
        """
        # Get input data (image -> latent space)
        if z is None:
            x = self.get_input(batch, 'image')  # [B, 1, D, H, W]
            if hasattr(self, 'vae') and self.vae is not None:
                with torch.no_grad():
                    
                    x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]
                    
                   
                    x = torch.clamp(x, -1.0, 1.0)  # Prevent input from exceeding range
                    
                    posterior = self.vae.encode(x)
                    # Usually sample during training; use posterior.mode() for deterministic
                    z = posterior.sample()
                    
                  
                    if torch.isnan(z).any() or torch.isinf(z).any():
                        print(f"\nWarning: VAE encoder output contains NaN/Inf!")
                        print(f"   Batch: {batch_idx}")
                        print(f"   NaN count: {torch.isnan(z).sum().item()}")
                        print(f"   Inf count: {torch.isinf(z).sum().item()}")
                        print(f"   Using zero tensor as replacement to avoid training crash")
                        z = torch.zeros_like(z)
                
                # Scale to training scale per Latent Diffusion
                if hasattr(self, 'scale_factor') and self.scale_factor is not None:
                    z = z * self.scale_factor
                    
                    # Check again after scaling
                    if torch.isnan(z).any() or torch.isinf(z).any():
                        print(f"\nWarning: VAE output contains NaN/Inf after scaling!")
                        print(f"   scale_factor: {self.scale_factor}")
                        z = torch.zeros_like(z)
            else:
                # Compatibility mode: if VAE not injected, assume input is already in latent space
                z = x
        
        # Get condition data
        conditions = self.get_condition_inputs(batch)
        
        # Add noise
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        noise = torch.randn_like(z)
        x_noisy = self.q_sample(x_start=z, t=t, noise=noise)
        
        # Model prediction
        model_output = self.apply_model(x_noisy, t, conditions)
        # Ensure same spatial scale as target tensor
        if model_output.shape[2:] != z.shape[2:]:
            model_output = F.interpolate(model_output, size=z.shape[2:], mode='nearest')
        
        # Compute loss
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = z
        else:
            raise NotImplementedError(f"parameterization {self.parameterization} not yet supported")
        
        # MSE loss
        mse_loss = F.mse_loss(model_output, target, reduction='mean')
        
        # Condition consistency loss (based on MCDDPM)
        condition_loss = self.compute_condition_consistency_loss(batch, model_output)
        
        
        # Ensure generated data looks like real rock in Avizo
        structure_loss = torch.tensor(0.0, device=self.device)
        
        if self.use_structure_loss and hasattr(self, 'structure_loss'):
            # Estimate x0 from noise prediction (denoised latent encoding)
            if self.parameterization == "eps":
                # x0 = (x_noisy - sqrt(1-alpha_t) * eps_pred) / sqrt(alpha_t)
                alpha_t = self.alphas_cumprod[t]  # [B]
                alpha_t = alpha_t.view(-1, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                
                # Predicted x0 (denoised latent encoding)
                pred_x0 = (x_noisy - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t
            else:
                # If directly predicting x0
                pred_x0 = model_output
            
            # Compute structure loss in latent space (avoid decoding every step, save computation)
            # Structural features in latent space also reflect image space structure
            structure_loss, structure_loss_dict = self.structure_loss(
                pred_x0,  # Predicted latent encoding
                z,        # Real latent encoding
                return_dict=True
            )
            structure_loss = structure_loss * self.structure_loss_weight
            
            # Log detailed structure loss
            if hasattr(self, 'global_step') and self.global_step % 100 == 0:
                print(f"\n  Structure similarity loss details:")
                for key, value in structure_loss_dict.items():
                    print(f"     {key}: {value:.6f}")
        
        # Total loss
        total_loss = mse_loss + self.condition_weight * condition_loss + structure_loss
        
        # Log metrics
        log_dict = {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'condition_loss': condition_loss,
            'structure_loss': structure_loss,
            'condition_weight': self.condition_weight,
            'structure_weight': self.structure_loss_weight
        }
        
        return total_loss, log_dict
    
    def get_porosity_range_weight(self, porosity_value):
        
        for (min_p, max_p), weight in self.porosity_range_weights.items():
            if min_p <= porosity_value < max_p:
                return weight
        return 1.0  # Default weight (for out-of-range values)
    
    def get_dynamic_condition_weights(self, epoch):
       
        if epoch < 50:
            return 2.5, 1.0, 1.0  # Stage 1: porosity priority
        elif epoch < 100:
            return 1.5, 1.5, 1.0  # Stage 2: pore size mean priority
        elif epoch < 150:
            return 1.0, 1.0, 1.5  # Stage 3: pore size std priority
        elif epoch < 200:
            return 1.0, 1.0, 1.0  # Stage 4: balanced training
        else:
            return 2.0, 0.8, 1.2  # Stage 5: fine-tuning
    
    def on_train_epoch_start(self):
       
        super().on_train_epoch_start()
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.current_train_epoch = self.trainer.current_epoch
            if self.use_dynamic_condition_weights:
                porosity_w, pore_mean_w, pore_std_w = self.get_dynamic_condition_weights(self.current_train_epoch)
                print(f"\nEpoch {self.current_train_epoch} started")
                print(f"   Dynamic weights: porosity={porosity_w:.1f}, pore_size_mean={pore_mean_w:.1f}, pore_size_std={pore_std_w:.1f}")
    
    def compute_condition_consistency_loss(self, batch, model_output):
       
        # Check if conditions exist
        if not any([self.use_porosity_condition, 
                    self.use_pore_size_mean_condition, 
                    self.use_pore_size_std_condition]):
            return torch.tensor(0.0, device=self.device)
        
        # Extract condition vectors
        conditions = []
        
        if self.use_porosity_condition and 'porosity' in batch:
            porosity = batch['porosity']  # [B, 1] or [B]
            if porosity.dim() == 1:
                porosity = porosity.unsqueeze(1)
            conditions.append(porosity)
        
        if self.use_pore_size_mean_condition and 'pore_size_mean' in batch:
            pore_mean = batch['pore_size_mean']
            if pore_mean.dim() == 1:
                pore_mean = pore_mean.unsqueeze(1)
            conditions.append(pore_mean)
        
        if self.use_pore_size_std_condition and 'pore_size_std' in batch:
            pore_std = batch['pore_size_std']
            if pore_std.dim() == 1:
                pore_std = pore_std.unsqueeze(1)
            conditions.append(pore_std)
        
        if len(conditions) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Concatenate all conditions [B, num_conditions]
        condition_vector = torch.cat(conditions, dim=1)
        B = condition_vector.shape[0]
        
        # If batch too small, cannot compute pairwise distances
        if B < 2:
            return torch.tensor(0.0, device=self.device)
        
        # ====== Compute condition pairwise distance matrix ======
        # [B, B] - condition distance between each pair of samples
        condition_dist = torch.cdist(
            condition_vector, 
            condition_vector, 
            p=2  # Euclidean distance
        )
        
        # Numerical stability check
        if torch.isnan(condition_dist).any() or torch.isinf(condition_dist).any():
            print("Condition distance matrix contains NaN/Inf, skipping condition consistency loss")
            return torch.tensor(0.0, device=self.device)
        
        # ====== Compute model_output pairwise distance matrix ======
        # Compress model_output to feature vectors (global average pooling)
        output_features = model_output.mean(dim=[2, 3, 4])  # [B, C]
        
        # Compute output feature pairwise distances
        output_dist = torch.cdist(
            output_features,
            output_features,
            p=2
        )
        
        # Numerical stability check
        if torch.isnan(output_dist).any() or torch.isinf(output_dist).any():
            print("Output distance matrix contains NaN/Inf, skipping condition consistency loss")
            return torch.tensor(0.0, device=self.device)
        
        # ====== Normalize distance matrices ======
        # Avoid excessive numerical range differences
        eps = 1e-8
        condition_dist_max = torch.clamp(condition_dist.max(), min=eps)  # Prevent max from being 0
        output_dist_max = torch.clamp(output_dist.max(), min=eps)
        
        condition_dist_norm = condition_dist / condition_dist_max
        output_dist_norm = output_dist / output_dist_max
        
        # ====== Compute loss ======
        # Goal: condition distance ~ output distance
        # i.e.: samples with similar conditions should have similar outputs
        consistency_loss = F.mse_loss(output_dist_norm, condition_dist_norm)
        
        # Final NaN check
        if torch.isnan(consistency_loss) or torch.isinf(consistency_loss):
            print("Condition consistency loss is NaN/Inf, skipping")
            return torch.tensor(0.0, device=self.device)
        
        # ====== Apply dynamic weights (if enabled) ======
        if self.use_dynamic_condition_weights:
            porosity_w, pore_mean_w, pore_std_w = self.get_dynamic_condition_weights(
                self.current_train_epoch
            )
            # Use average weight
            avg_weight = (porosity_w + pore_mean_w + pore_std_w) / 3.0
            consistency_loss = consistency_loss * avg_weight
        
        # ====== Apply small base weight ======
        # Use 0.1 base weight to avoid over-constraining the main task (denoising)
        weighted_loss = consistency_loss * 0.1
        
        # ====== Debug log (every 100 steps) ======
        if hasattr(self, 'global_step') and self.global_step % 100 == 0:
            print(f"\n  Condition consistency loss (relative relationship based):")
            print(f"     Batch size: {B}")
            print(f"     Condition dimensions: {condition_vector.shape[1]}")
            print(f"     Condition distance range: [{condition_dist.min():.4f}, {condition_dist.max():.4f}]")
            print(f"     Output distance range: [{output_dist.min():.4f}, {output_dist.max():.4f}]")
            print(f"     Normalized loss: {consistency_loss.item():.6f}")
            print(f"     Weighted loss: {weighted_loss.item():.6f}")
            if self.use_dynamic_condition_weights:
                print(f"     Dynamic weight: {avg_weight:.2f}")
        
        return weighted_loss
    
   


