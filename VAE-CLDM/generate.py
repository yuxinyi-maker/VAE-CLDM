
import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import argparse
import glob

# Add path - adapted for sandstone project
sys.path.append('/hy-tmp/VAEDDPM_sandstone')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import os
    POROSITY_BOOST = float(os.environ.get('POROSITY_BOOST', '1.0'))  # e.g., 1.15 ~ 1.30 for low targets
    RESEED_TRIALS = int(os.environ.get('RESEED_TRIALS', '2'))        # extra seeds per sample for low targets
except Exception:
    POROSITY_BOOST = 1.0
    RESEED_TRIALS = 2


def resolve_checkpoint(ckpt_path: str = None, ckpt_dir: str = None, epoch: int = None):
    """Resolve the final checkpoint path to use
    Priority: explicit ckpt_path > (ckpt_dir + epoch match) > default
    """
    if ckpt_path and os.path.exists(ckpt_path):
        return ckpt_path
    if ckpt_dir and epoch is not None:
        pattern = os.path.join(ckpt_dir, f"numerical_diffusion-epoch={epoch:03d}-*.ckpt")
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def load_model_from_checkpoint(checkpoint_path, vae_ckpt_path, device='cuda:0'):
    """Load model from checkpoint"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not os.path.exists(vae_ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt_path}")
    
    # Load checkpoint info
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')
    
    print(f"   Epoch: {epoch}, Val Loss: {val_loss}")
    
    try:
        from main_vaedpm_new import NumericalConditionalDiffusionModel
        
        print("   Loading with PyTorch Lightning...")
        
        model = NumericalConditionalDiffusionModel.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            vae_ckpt_path=vae_ckpt_path,
        )
        
        print("   Model loaded successfully")
        
    except Exception as e:
        print(f"   Load failed: {str(e)[:200]}...")
        raise
    
    model.eval()
    model.to(device)
    print(f"Model loaded, moved to {device}")
    
    return model


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999), alphas_cumprod[:-1]


def normalize_conditions(conditions, condition_ranges=None):
    """
    Critical fix: use the exact same normalization as during training
    
    Normalization used during training (in numerical_conditional_diffusion.py get_condition_inputs):
    - Porosity: (porosity - 0.20) / 0.05, range ~[-3, 3]
    - Pore size mean: (pore_mean - 5.8) / 3.0, range ~[-3, 3]
    - Pore size std: (pore_std - 5.8) / 4.0, range ~[-3, 3]
    
    This is Z-score normalization (based on mean and std), NOT min-max normalization!
    
    Args:
        conditions: raw condition dictionary
        condition_ranges: condition range dictionary (deprecated, kept for API compatibility)
    
    Returns:
        Normalized condition dictionary (consistent with training)
    """
    normalized = {}
    
    for key, value in conditions.items():
        if key == 'porosity':
            # Fix: use the exact same normalization as during training
            # Training: porosity_norm = (porosity - 0.20) / 0.05
            if isinstance(value, torch.Tensor):
                normalized_value = (value - 0.20) / 0.05
                normalized_value = torch.clamp(normalized_value, -3.0, 3.0)
            else:
                normalized_value = (value - 0.20) / 0.05
                normalized_value = max(-3.0, min(3.0, normalized_value))
            normalized[key] = normalized_value
        elif key == 'pore_size_mean':
            # Fix: use the exact same normalization as during training
            # Training: pore_mean_norm = (pore_mean - 5.8) / 3.0
            if isinstance(value, torch.Tensor):
                normalized_value = (value - 5.8) / 3.0
                normalized_value = torch.clamp(normalized_value, -3.0, 3.0)
            else:
                normalized_value = (value - 5.8) / 3.0
                normalized_value = max(-3.0, min(3.0, normalized_value))
            normalized[key] = normalized_value
        elif key == 'pore_size_std':
            # Fix: use the exact same normalization as during training
            # Training: pore_std_norm = (pore_std - 5.8) / 4.0
            if isinstance(value, torch.Tensor):
                normalized_value = (value - 5.8) / 4.0
                normalized_value = torch.clamp(normalized_value, -3.0, 3.0)
            else:
                normalized_value = (value - 5.8) / 4.0
                normalized_value = max(-3.0, min(3.0, normalized_value))
            normalized[key] = normalized_value
        else:
            # Other conditions used directly
            normalized[key] = value
    
    return normalized


def adaptive_ddim_sampling(model, conditions, batch_size=1, 
                           target_porosity=None, device='cuda:0', verbose=True,
                           condition_ranges=None,
                           steps_override: int = None,
                           guidance_override: float = None,
                           seed: int = None,
                           porosity_boost: float = None):
    """
    Adaptive DDIM sampling - adjust sampling parameters based on target porosity (adapted for sandstone data distribution)
    
    Updates:
    - Added time-varying CFG (stronger in later steps), more aggressive only for low porosity targets
    - Supports optional porosity_boost (only for low range), mild amplification of normalized porosity z-value
    - Supports specifying seed for external multi-seed resampling
    """
    
    if target_porosity is None:
        target_porosity = conditions['porosity'][0, 0].item()
    
    is_low_target = target_porosity < 0.16
    
    # Adaptive base strategy (determine steps and base CFG)
    if steps_override is not None and guidance_override is not None:
        steps = int(steps_override)
        base_guidance = float(guidance_override)
        eta = 0.0
        sampling_strategy = f"External override: {steps} steps + CFG {base_guidance:.1f}"
    else:
        if target_porosity < 0.12:
            steps = 240
            base_guidance = 20.0
            eta = 0.0
            sampling_strategy = "Very low porosity strategy: 240 steps + CFG 20.0 (time-varying)"
        elif target_porosity < 0.14:
            steps = 220
            base_guidance = 18.0
            eta = 0.0
            sampling_strategy = "Medium-high porosity strategy: 220 steps + CFG 18.0 (time-varying)"
        elif target_porosity < 0.16:
            steps = 200
            base_guidance = 14.0
            eta = 0.0
            sampling_strategy = "High porosity strategy: 200 steps + CFG 14.0 (time-varying)"
        elif target_porosity < 0.18:
            steps = 160
            base_guidance = 10.0
            eta = 0.0
            sampling_strategy = "Very high porosity strategy (low end): 160 steps + CFG 10.0"
        elif target_porosity < 0.20:
            steps = 150
            base_guidance = 8.0
            eta = 0.0
            sampling_strategy = "Very high porosity strategy: 150 steps + CFG 8.0"
        elif target_porosity < 0.25:
            steps = 150
            base_guidance = 8.0
            eta = 0.0
            sampling_strategy = "Ultra high porosity strategy: 150 steps + CFG 8.0"
        else:
            steps = 180
            base_guidance = 10.0
            eta = 0.0
            sampling_strategy = "Maximum porosity strategy: 180 steps + CFG 10.0"
    
    # Time-varying CFG: stronger in later steps (significant boost only for low range)
    if guidance_override is not None:
        gs_start = base_guidance
        gs_end = base_guidance + (6.0 if is_low_target else 0.0)
    else:
        gs_start = max(1.0, base_guidance - (4.0 if is_low_target else 2.0))
        gs_end = base_guidance + (6.0 if is_low_target else 0.0)
    
    # porosity_boost (only enabled for low range)
    boost = POROSITY_BOOST if porosity_boost is None else float(porosity_boost)
    if is_low_target and boost > 1.0:
        # Mild amplification of the z-score (only for porosity, not pore size conditions)
        try:
            conditions['porosity'] = conditions['porosity'] * boost
        except Exception:
            pass
    
    if verbose:
        print(f"\nAdaptive sampling: {sampling_strategy}")
        print(f"   Target porosity: {target_porosity:.4f}")
        print(f"   DDIM steps: {steps}")
        print(f"   CFG time-varying: start={gs_start:.1f} -> end={gs_end:.1f}")
        if is_low_target and boost > 1.0:
            print(f"   Low range porosity_boost: x{boost:.2f}")
    
    # Noise schedule
    timesteps = 1000
    betas, alphas_cumprod = cosine_beta_schedule(timesteps)
    betas = betas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    
    # Initial noise (latent space: 8x8x8) + optional seed
    gen = None
    if seed is not None:
        try:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
        except Exception:
            gen = None
    x = torch.randn((batch_size, 4, 8, 8, 8), device=device, generator=gen)
    
    # DDIM timesteps
    ddim_timesteps = torch.linspace(timesteps - 1, 0, steps, device=device).long()
    
    # Unconditional input (for CFG) - use zero conditions
    null_conditions = {
        'porosity': torch.zeros_like(conditions['porosity']),
        'pore_size_mean': torch.zeros_like(conditions['pore_size_mean']),
        'pore_size_std': torch.zeros_like(conditions['pore_size_std'])
    }
    
    # Denoising process
    for i, t in enumerate(ddim_timesteps):
        # Linear time-varying CFG
        if steps > 1:
            guidance_scale = gs_start + (gs_end - gs_start) * (i / (steps - 1))
        else:
            guidance_scale = gs_end
        
        if verbose and (i % 10 == 0 or i == len(ddim_timesteps) - 1):
            print(f"   Step {i+1}/{len(ddim_timesteps)}  CFG={guidance_scale:.2f}", end='\r')
        
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            if guidance_scale <= 1.0:
                noise_pred = model.diffusion.model(
                    x, t_batch,
                    porosity_condition=conditions['porosity'],
                    pore_size_mean_condition=conditions['pore_size_mean'],
                    pore_size_std_condition=conditions['pore_size_std']
                )
            else:
                noise_pred_cond = model.diffusion.model(
                    x, t_batch,
                    porosity_condition=conditions['porosity'],
                    pore_size_mean_condition=conditions['pore_size_mean'],
                    pore_size_std_condition=conditions['pore_size_std']
                )
                noise_pred_uncond = model.diffusion.model(
                    x, t_batch,
                    porosity_condition=null_conditions['porosity'],
                    pore_size_mean_condition=null_conditions['pore_size_mean'],
                    pore_size_std_condition=null_conditions['pore_size_std']
                )
                if torch.isnan(noise_pred_cond).any() or torch.isinf(noise_pred_cond).any():
                    noise_pred = noise_pred_uncond
                elif torch.isnan(noise_pred_uncond).any() or torch.isinf(noise_pred_uncond).any():
                    noise_pred = noise_pred_cond
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                        noise_pred = noise_pred_cond
        
        # DDIM update
        alpha_t = alphas_cumprod[t]
        if i < len(ddim_timesteps) - 1:
            t_next = ddim_timesteps[i + 1]
            alpha_t_prev = alphas_cumprod[t_next]
        else:
            alpha_t_prev = torch.tensor(1.0, device=device)
        
        sqrt_alpha_t = torch.sqrt(alpha_t).view(1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(1, 1, 1, 1, 1)
        pred_x0 = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)
        
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev).view(1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev).view(1, 1, 1, 1, 1)
        dir_xt = sqrt_one_minus_alpha_t_prev * noise_pred
        x = sqrt_alpha_t_prev * pred_x0 + dir_xt
        
        if eta > 0 and i < len(ddim_timesteps) - 1:
            sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            sigma = sigma.view(1, 1, 1, 1, 1)
            noise = torch.randn_like(x)
            x = x + sigma * noise
    
    if verbose:
        print(f"   Step {len(ddim_timesteps)}/{len(ddim_timesteps)} - Done!")
    
    return x


def calculate_rock_properties(binary_image):
    """Calculate rock physical properties"""
    from demoto import demotoo1
    
    # Compute porosity
    porosity = (binary_image == 0).sum() / binary_image.size
    
    # Compute pore size parameters
    try:
        label = np.squeeze(demotoo1(binary_image))
        pore_size_mean = float(label[0])
        pore_size_std = float(label[1])
    except:
        pore_size_mean = 0.0
        pore_size_std = 0.0
    
    return {
        'porosity': porosity,
        'pore_size_mean': pore_size_mean,
        'pore_size_std': pore_size_std
    }


def evaluate_with_adaptive_sampling(model, target_conditions, num_samples=5, device='cuda:0',
                                    condition_ranges=None,
                                    reseed_trials: int = None,
                                    porosity_boost: float = None):
    """
    Evaluate generation quality using adaptive sampling (adapted for sandstone data)
    
    Enhancements:
    - When generated porosity is significantly higher than target, increase guidance and retry (up to N times)
    - For low porosity targets, supports multi-seed resampling (reseed) with optional porosity_boost
    """
    if condition_ranges is None:
        condition_ranges = {
            'pore_size_mean': [1.5, 13.0],
            'pore_size_std': [0.9, 18.0]
        }
    
    if reseed_trials is None:
        reseed_trials = RESEED_TRIALS
    if porosity_boost is None:
        porosity_boost = POROSITY_BOOST
    
    print(f"\nAdaptive sampling evaluation (sandstone data):")
    print(f"   Target porosity: {target_conditions['porosity']:.4f}")
    print(f"   Target pore size mean: {target_conditions['pore_size_mean']:.2f}")
    print(f"   Target pore size std: {target_conditions['pore_size_std']:.2f}")
    
    results = []
    is_low_target = target_conditions['porosity'] < 0.16
    
    for i in range(num_samples):
        print(f"\n   Sample {i+1}/{num_samples}:")
        
        raw_conditions = {
            'porosity': target_conditions['porosity'],
            'pore_size_mean': target_conditions['pore_size_mean'],
            'pore_size_std': target_conditions['pore_size_std']
        }
        
        normalized_conditions = normalize_conditions(raw_conditions, condition_ranges)
        
        base_conditions = {
            'porosity': torch.tensor([[normalized_conditions['porosity']]], dtype=torch.float32, device=device),
            'pore_size_mean': torch.tensor([[normalized_conditions['pore_size_mean']]], dtype=torch.float32, device=device),
            'pore_size_std': torch.tensor([[normalized_conditions['pore_size_std']]], dtype=torch.float32, device=device)
        }
        
        # Initial parameters
        guidance = None
        steps = None
        retries = 0
        max_retries = 3
        retry_increment = 3.0
        
        # reseed attempts (only enabled for low range)
        max_reseed = reseed_trials if is_low_target else 0
        reseed_id = -1
        
        generated_latent = None
        properties = None
        errors = None
        
        while True:
            reseed_id += 1
            if reseed_id > max_reseed:
                break
            
            # Use different random seed (if reseed triggered)
            seed = None
            if reseed_id > 0:
                seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                print(f"      Changing random seed: reseed#{reseed_id} seed={seed}")
            
            # Clone conditions to avoid in-place modification
            conditions = {
                'porosity': base_conditions['porosity'].clone(),
                'pore_size_mean': base_conditions['pore_size_mean'].clone(),
                'pore_size_std': base_conditions['pore_size_std'].clone(),
            }
            
            # Sampling (time-varying CFG + optional porosity_boost)
            try:
                generated_latent = adaptive_ddim_sampling(
                    model, conditions,
                    batch_size=1,
                    target_porosity=target_conditions['porosity'],
                    device=device,
                    verbose=False,
                    condition_ranges=condition_ranges,
                    steps_override=steps,
                    guidance_override=guidance,
                    seed=seed,
                    porosity_boost=porosity_boost,
                )
                if torch.isnan(generated_latent).any() or torch.isinf(generated_latent).any():
                    print(f"      Error: generated latent contains NaN/Inf, skipping sample")
                    generated_latent = None
                    continue
            except Exception as e:
                print(f"      Error: sampling exception: {str(e)[:100]}")
                generated_latent = None
                continue
            
            # Decode and compute porosity
            with torch.no_grad():
                try:
                    generated_image = model.vae.decode(generated_latent)
                    generated_image = (generated_image + 1.0) / 2.0
                    generated_image = torch.clamp(generated_image, 0, 1)
                    
                    if torch.isnan(generated_image).any() or torch.isinf(generated_image).any():
                        print(f"      Error: VAE decode output contains NaN/Inf, skipping sample")
                        generated_latent = None
                        continue
                except Exception as e:
                    print(f"      Error: VAE decode exception: {str(e)[:100]}")
                    generated_latent = None
                    continue
            
            image_np = generated_image[0, 0].cpu().numpy()
            if np.isnan(image_np).any() or np.isinf(image_np).any():
                print(f"      Error: image data contains NaN/Inf, skipping sample")
                generated_latent = None
                continue
            
            image_binary = (image_np >= 0.5).astype(np.uint8)
            try:
                properties = calculate_rock_properties(image_binary)
            except Exception as e:
                print(f"      Error: physical property calculation exception: {str(e)[:100]}")
                generated_latent = None
                continue
            
            errors = {
                'porosity_error': abs(properties['porosity'] - target_conditions['porosity']),
                'porosity_error_pct': abs(properties['porosity'] - target_conditions['porosity']) / target_conditions['porosity'] * 100 if target_conditions['porosity'] > 0 else float('inf'),
                'pore_size_mean_error': abs(properties['pore_size_mean'] - target_conditions['pore_size_mean']),
                'pore_size_std_error': abs(properties['pore_size_std'] - target_conditions['pore_size_std'])
            }
            
            # Low range: if generated porosity is significantly higher, increase CFG and retry; if still not met, reseed
            need_retry = False
            if is_low_target and properties['porosity'] > target_conditions['porosity'] * 1.15 and retries < max_retries:
                retries += 1
                guidance = (guidance or 0.0) + retry_increment if guidance is not None else (16.0 if target_conditions['porosity'] < 0.14 else 14.0)
                print(f"      Retry {retries}/{max_retries}: increasing CFG to {guidance:.1f}")
                # No reseed, try stronger guidance on current seed
                continue
            
            # If low range still significantly higher and CFG exhausted, trigger reseed (next while loop iteration changes seed)
            if is_low_target and properties['porosity'] > target_conditions['porosity'] * 1.15 and retries >= max_retries and reseed_id < max_reseed:
                print("      Switching to multi-seed resampling (CFG reached limit)")
                # Reset retry count, allow retrying with increased CFG on new seed
                retries = 0
                guidance = None
                continue
            
            # Target met or non-low range, exit loop
            break
        
        if generated_latent is None or properties is None:
            continue
        
        failed = errors['porosity_error_pct'] > 150
        result = {
            'target': target_conditions.copy(),
            'actual': properties,
            'errors': errors,
            'image': image_binary,
            'failed': failed
        }
        results.append(result)
        
        print(f"      Actual porosity: {properties['porosity']:.4f} (error: {errors['porosity_error']:.4f}, {errors['porosity_error_pct']:.2f}%)")
        print(f"      Actual pore size mean: {properties['pore_size_mean']:.2f} (error: {errors['pore_size_mean_error']:.2f})")
    
    porosity_errors = [r['errors']['porosity_error'] for r in results]
    porosity_error_pcts = [r['errors']['porosity_error_pct'] for r in results]
    
    summary = {
        'mean_porosity_error': float(np.mean(porosity_errors)) if porosity_errors else float('nan'),
        'std_porosity_error': float(np.std(porosity_errors)) if porosity_errors else float('nan'),
        'mean_porosity_error_pct': float(np.mean(porosity_error_pcts)) if porosity_error_pcts else float('nan'),
        'max_porosity_error': float(np.max(porosity_errors)) if porosity_errors else float('nan'),
        'min_porosity_error': float(np.min(porosity_errors)) if porosity_errors else float('nan')
    }
    
    print(f"\n   Statistics summary:")
    if porosity_errors:
        print(f"      Mean porosity error: {summary['mean_porosity_error']:.4f} ({summary['mean_porosity_error_pct']:.2f}%)")
        print(f"      Porosity error range: {summary['min_porosity_error']:.4f} ~ {summary['max_porosity_error']:.4f}")
    else:
        print("      No valid samples")
    
    return results, summary


def main():
    """Main function - sandstone data generation (adapted for sandstone training set)"""
    parser = argparse.ArgumentParser(description="Sandstone conditional generation")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ckpt', default='', help='explicit checkpoint path')
    parser.add_argument('--ckpt_dir', default='/hy-tmp/VAEDDPM_sandstone/outputs/conditionaldiffusion', help='directory of checkpoints')
    parser.add_argument('--epoch', type=int, default=100, help='prefer checkpoint epoch (e.g., 100)')
    parser.add_argument('--vae_ckpt', default='/hy-tmp/VAEDDPM_sandstone/outputs/unconditionalVAE_fixed/checkpoints/last-v1.ckpt')
    parser.add_argument('--out_dir', default='/hy-tmp/VAEDDPM_sandstone/evaluation_results')
    parser.add_argument('--samples_per_condition', type=int, default=10)
    args = parser.parse_args()

    # Resolve final ckpt
    resolved_ckpt = resolve_checkpoint(args.ckpt, args.ckpt_dir, args.epoch)
    if resolved_ckpt is None:
        # Fall back to last.ckpt in directory
        fallback = os.path.join(args.ckpt_dir, 'last.ckpt')
        resolved_ckpt = fallback
    
    # Sandstone configuration
    config = {
        'device': args.device,
        'checkpoint_path': resolved_ckpt,
        'vae_ckpt_path': args.vae_ckpt,
        'output_dir': args.out_dir,
        'samples_output_dir': os.path.join(args.out_dir, 'generated_samples'),
        'num_samples_per_condition': args.samples_per_condition,
        # Sandstone condition normalization range (kept for log display; actual normalization uses Z-score)
        'condition_ranges': {
            'pore_size_mean': [1.5, 13.0],
            'pore_size_std': [0.9, 18.0]
        },
        # Test condition list (modify as needed)
        'test_conditions': [
            {'name': 'medium_high_porosity_medium_pore', 'porosity': 0.13, 'pore_size_mean': 5.0, 'pore_size_std': 5.0},
            {'name': 'high_porosity_medium_pore',   'porosity': 0.15, 'pore_size_mean': 6.0, 'pore_size_std': 6.0},
            {'name': 'very_high_porosity_large_pore', 'porosity': 0.18, 'pore_size_mean': 7.0, 'pore_size_std': 7.0},
            {'name': 'ultra_high_porosity_large_pore', 'porosity': 0.20, 'pore_size_mean': 8.0, 'pore_size_std': 8.0}
        ]
    }
    
    print("\n" + "="*80)
    print("Sandstone Data Generation - Adaptive Sampling Strategy Test")
    print("="*80)
    print(f"Device: {config['device']}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print(f"VAE checkpoint: {config['vae_ckpt_path']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Samples output directory: {config['samples_output_dir']}")
    print(f"Samples per condition: {config['num_samples_per_condition']}")
    print(f"Condition normalization ranges:")
    print(f"   Pore size mean: {config['condition_ranges']['pore_size_mean']}")
    print(f"   Pore size std: {config['condition_ranges']['pore_size_std']}")
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['samples_output_dir'], exist_ok=True)
    
    # Load model
    model = load_model_from_checkpoint(
        config['checkpoint_path'],
        config['vae_ckpt_path'],
        config['device']
    )
    
    # Evaluation
    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80)
    
    results_all = {}
    summary_all = {}
    
    for condition in config['test_conditions']:
        print(f"\n{'='*80}")
        print(f"Test condition: {condition['name']}")
        print(f"{'='*80}")
        
        results, summary = evaluate_with_adaptive_sampling(
            model,
            condition,
            num_samples=config['num_samples_per_condition'],
            device=config['device'],
            condition_ranges=config['condition_ranges']
        )
        
        results_all[condition['name']] = results
        summary_all[condition['name']] = summary
        
        # Save samples
        condition_name_safe = condition['name'].replace('/', '_').replace(' ', '_')
        condition_dir = os.path.join(config['samples_output_dir'], condition_name_safe)
        os.makedirs(condition_dir, exist_ok=True)
        for idx, result in enumerate(results):
            filename = f"poro{condition['porosity']:.3f}_mean{condition['pore_size_mean']:.1f}_std{condition['pore_size_std']:.1f}_sample{idx+1:03d}.npy"
            filepath = os.path.join(condition_dir, filename)
            np.save(filepath, result['image'])
            if (idx + 1) % 5 == 0 or idx == len(results) - 1:
                print(f"   Saved {idx+1}/{len(results)} samples to: {condition_dir}/")
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': summary_all
    }
    report_path = os.path.join(config['output_dir'], 'adaptive_evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "="*80)
    print("Final Evaluation Summary")
    print("="*80)
    for name, summary in summary_all.items():
        print(f"\n{name}:")
        print(f"  Mean porosity error: {summary['mean_porosity_error']:.4f} ({summary['mean_porosity_error_pct']:.2f}%)")
        print(f"  Porosity error range: {summary['min_porosity_error']:.4f} ~ {summary['max_porosity_error']:.4f}")
    
    print(f"\nReport saved: {report_path}")
    print(f"Binarized samples saved: {config['samples_output_dir']}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

