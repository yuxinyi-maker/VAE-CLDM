
import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util_new import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)

class Rock3DDDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        """
        3D rock conditional DDIM sampler - complete version
        
        Args:
            model: Trained LatentDiffusion model
            schedule: Beta schedule type
        """
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda')
        
        print(f"Initialized 3D DDIM sampler: device={self.device}")

    def register_buffer(self, name, attr):
        """Device-compatible buffer registration"""
        if isinstance(attr, torch.Tensor):
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        """Create 3D DDIM sampling schedule"""
        print(f"Creating DDIM sampling schedule: {ddim_num_steps} steps, eta={ddim_eta}")
        
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose
        )
        
        # Get model parameters
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas_cumprod length mismatch'

        # Register diffusion parameters
        buffers_to_register = {
            'betas': self.model.betas,
            'alphas_cumprod': alphas_cumprod,
            'alphas_cumprod_prev': self.model.alphas_cumprod_prev,
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1. - alphas_cumprod),
            'log_one_minus_alphas_cumprod': torch.log(1. - alphas_cumprod),
            'sqrt_recip_alphas_cumprod': torch.sqrt(1. / alphas_cumprod),
            'sqrt_recipm1_alphas_cumprod': torch.sqrt(1. / alphas_cumprod - 1),
        }
        
        for name, buffer in buffers_to_register.items():
            self.register_buffer(name, buffer)

        # DDIM specific parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose
        )
        
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

        print("DDIM sampling schedule created")

    @torch.no_grad()
    def sample(
        self,
        steps=50,
        batch_size=1,
        shape=(4, 16, 16, 16),
        text_conditioning=None,
        porosity_conditioning=None,
        unconditional_guidance_scale=5.0,
        unconditional_text="",
        eta=0.0,
        temperature=1.0,
        x_T=None,
        verbose=True,
        **kwargs
    ):
        """
        3D rock conditional generation entry method
        """
        # Validate conditions
        if porosity_conditioning is not None:
            assert porosity_conditioning.shape[0] == batch_size, "Porosity condition batch size mismatch"
        
        # Create sampling schedule
        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)

        # 3D shape validation
        C, D, H, W = shape
        assert C == 4, f"Latent space channels should be 4, got {C}"
        size = (batch_size, C, D, H, W)
        
        print(f'3D DDIM sampling: shape {size}, steps {steps}, CFG scale {unconditional_guidance_scale}')

        # Prepare conditions
        if text_conditioning is None:
            text_conditioning = ["a 3d rock sample"] * batch_size
        
        # Get text encoding
        if isinstance(text_conditioning, list):
            text_conditioning = self.model.get_learned_conditioning(text_conditioning)
        
        # Prepare unconditional conditions
        if unconditional_guidance_scale != 1.0:
            if unconditional_text == "":
                unconditional_text = [""] * batch_size
            unconditional_conditioning = self.model.get_learned_conditioning(unconditional_text)
        else:
            unconditional_conditioning = None

        # Execute sampling
        samples, intermediates = self.ddim_sampling(
            cond=text_conditioning,
            shape=size,
            x_T=x_T,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            porosity_conditioning=porosity_conditioning,
            temperature=temperature,
        )
        
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self, cond, shape, x_T=None,
        callback=None, quantize_denoised=False,
        img_callback=None, log_every_t=100,
        temperature=1., noise_dropout=0., 
        unconditional_guidance_scale=1., unconditional_conditioning=None,
        porosity_conditioning=None
    ):
        """Main sampling loop - 3D rock conditional version"""
        device = self.device
        b = shape[0]
        
        # Initialize 3D noise
        if x_T is None:
            img = torch.randn(shape, device=device)
            print(f"Generated 3D random noise: {img.shape}")
        else:
            img = x_T
            
        timesteps = self.ddim_timesteps
        total_steps = timesteps.shape[0]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        
        print(f"Running DDIM sampling, {total_steps} steps total")
        iterator = tqdm(reversed(range(0, total_steps)), desc='DDIM Sampling', total=total_steps)
        
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Single step sampling
            outs = self.p_sample_ddim(
                x=img, c=cond, t=ts, index=index,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                porosity_conditioning=porosity_conditioning
            )
            img, pred_x0 = outs

            # Callbacks and processing
            if callback: 
                callback(i)
            if img_callback: 
                img_callback(pred_x0, i)
                
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(
        self, x, c, t, index, repeat_noise=False, use_original_steps=False,
        quantize_denoised=False, temperature=1., noise_dropout=0.,
        score_corrector=None, corrector_kwargs=None,
        unconditional_guidance_scale=1., unconditional_conditioning=None,
        porosity_conditioning=None
    ):
        """
        Single step DDIM sampling
        """
        b, *_, device = *x.shape, x.device
    
        # Build extra_conditions dict
        extra_conditions = {}
        if porosity_conditioning is not None:
            if porosity_conditioning.device != device:
                porosity_conditioning = porosity_conditioning.to(device)
            extra_conditions['porosity'] = porosity_conditioning
    
        # Single condition mode
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(
                x, t, 
                cond=c,
                extra_conditions=extra_conditions if extra_conditions else None
            )
        else:
            # Classifier-Free Guidance
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            
            # Build CFG conditions
            if porosity_conditioning is not None:
                porosity_in = torch.cat([porosity_conditioning] * 2)
                cfg_extra_conditions = {'porosity': porosity_in}
            else:
                cfg_extra_conditions = {}
            
            # Text condition: unconditional + conditional
            cond_in = torch.cat([unconditional_conditioning, c])
            
            # Apply model
            e_t_uncond, e_t_cond = self.model.apply_model(
                x_in, t_in,
                cond=cond_in,
                extra_conditions=cfg_extra_conditions if cfg_extra_conditions else None
            ).chunk(2)
            
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)
    
        # Get DDIM parameters
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
    
        # 3D shape parameters [B, 1, 1, 1, 1]
        a_t = torch.full((b, 1, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
    
        # Predict x0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        
        # Quantize (if supported)
        if quantize_denoised and hasattr(self.model.first_stage_model, "quantize"):
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
    
        # Direction term and noise
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    
        # Compute x_{t-1}
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        """Encode to specified timestep"""
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
            
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
            extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

class Rock3DGenerator:
    """
    3D rock generator - complete end-to-end generation interface
    """
    
    def __init__(self, model, sampler_class=Rock3DDDIMSampler):
        self.model = model
        self.sampler = sampler_class(model)
        self.device = next(model.parameters()).device
        
        print(f"Initialized 3D rock generator: device={self.device}")

    @torch.no_grad()
    def generate_rock_samples(
        self,
        text_prompts=None,
        porosity_values=None,
        rock_types=None,
        num_samples=1,
        ddim_steps=50,
        guidance_scale=5.0,
        seed=None,
        output_dir=None
    ):
        """
        Generate 3D rock samples
        
        Args:
            text_prompts: List of text descriptions
            porosity_values: List of porosity values (0-1 range)
            rock_types: List of rock types ['sandstone', 'limestone']
            num_samples: Number of samples to generate
            ddim_steps: DDIM steps
            guidance_scale: CFG guidance scale
            seed: Random seed
            output_dir: Output directory
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Prepare conditions
        if text_prompts is None:
            text_prompts = ["a 3d rock microstructure"] * num_samples
            
        if porosity_values is None:
            porosity_values = [0.2] * num_samples  # Default porosity
            
        if rock_types is None:
            rock_types = ["sandstone"] * num_samples
            
        # Convert to tensors
        porosity_tensor = torch.tensor(porosity_values, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        print(f"Generation conditions:")
        print(f"  Text prompts: {text_prompts}")
        print(f"  Porosity: {porosity_values}")
        print(f"  Rock types: {rock_types}")
        
        # Execute sampling
        latent_samples, _ = self.sampler.sample(
            steps=ddim_steps,
            batch_size=num_samples,
            shape=(4, 16, 16, 16),
            text_conditioning=text_prompts,
            porosity_conditioning=porosity_tensor,
            unconditional_guidance_scale=guidance_scale,
            unconditional_text="",
            eta=0.0
        )
        
        # Decode to image space
        print("Decoding latent samples...")
        with torch.no_grad():
            samples = self.model.decode_first_stage(latent_samples)
            
        # Post-process to binary rock data
        binary_samples = self.postprocess_to_binary(samples)
        
        return binary_samples, latent_samples

    def postprocess_to_binary(self, samples, threshold=0.5):
        """
        Post-process generated samples to binary rock data
        
        Args:
            samples: Generated 3D samples [B, 1, 64, 64, 64]
            threshold: Binarization threshold
        """
        # Ensure in [0,1] range
        samples = torch.clamp(samples, 0, 1)
        
        # Binarize
        binary_samples = (samples > threshold).float()
        
        print(f"Binarization complete: shape {binary_samples.shape}")
        
        return binary_samples

    def save_as_npy(self, samples, output_dir, prefix="generated_rock"):
        """
        Save as .npy files
        
        Args:
            samples: 3D sample tensors
            output_dir: Output directory
            prefix: File prefix
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        samples_np = samples.cpu().numpy()
        
        for i, sample in enumerate(samples_np):
            filename = f"{prefix}_{i:04d}.npy"
            filepath = os.path.join(output_dir, filename)
            
            # Save as 0/1 array
            np.save(filepath, sample.astype(np.uint8))



if __name__ == "__main__":
    print("3D rock DDIM sampler loaded")

