
import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat, rearrange

from ldm.util_new import instantiate_from_config




def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
   
    print(f"🔧 Create{schedule} betaschedule: {n_timestep}steps, range[{linear_start:.2e}, {linear_end:.2e}]")
    
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"English textscheduleEnglish text '{schedule}'")
    
    # Convert to numpy array for validation
    betas_np = betas.numpy()
    
  
    has_invalid_low = np.any(betas_np <= 0)
    has_invalid_high = np.any(betas_np >= 1)
    
    if has_invalid_low or has_invalid_high:
        print("⚠️ Warning: beta values exceed(0,1)range,clipping")
        betas_np = np.clip(betas_np, 1e-6, 1-1e-6)
    
    print(f"✅ BetascheduleCreateEnglish text: range[{betas_np.min():.3e}, {betas_np.max():.3e}]")
    return betas_np


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'No DDIM discretization method named "{ddim_discr_method}" ')

    # Add one to obtain the final alpha value
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'🔧 English textDDIMSamplingEnglish textsteps: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    
    # English textscheduleEnglish textalpha
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # Compute sigma according to the formula in the DDIM paper
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    
    # Numerical stability handling
    sigmas = np.nan_to_num(sigmas, nan=0.0, posinf=1.0, neginf=0.0)
    sigmas = np.clip(sigmas, 0, 1)
    
    if verbose:
        print(f'🔧 DDIMSamplingEnglish textalpha: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'🔧 eta={eta}English textsigmaschedule: {sigmas}')
    
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        beta = min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
        # Prevent numerical issues
        beta = max(beta, 1e-6)
        betas.append(beta)
    
    betas = np.array(betas)
    print(f"✅ Alpha bar betaschedule: {len(betas)}steps, range[{betas.min():.3e}, {betas.max():.3e}]")
    return betas





def extract_into_tensor(arr, timesteps, broadcast_shape):
   
    device = timesteps.device
    if arr.device != device:
        arr = arr.to(device)
    
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)





def checkpoint(func, inputs, params, flag):

    return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        try:
            with torch.no_grad():
                output_tensors = ctx.run_function(*ctx.input_tensors)
            return output_tensors
        except Exception as e:
            print(f"❌ Gradient checkpoint forward pass failed: {e}")
            raise

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
          
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            try:
                output_tensors = ctx.run_function(*shallow_copies)
            except Exception as e:
                print(f"❌ Gradient checkpoint backward pass failed: {e}")
                raise
                
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
   
    # English textstepsEnglish textrangeEnglish text
    if torch.any(timesteps < 0) or torch.any(timesteps > 1000):
        print("⚠️ Warning: English textstepsEnglish textrange [0, 1000]")
        timesteps = torch.clamp(timesteps, 0, 1000)
    
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    
    return embedding


def zero_module(module):
    
    try:
        for p in module.parameters():
            p.detach().zero_()
        return module
    except Exception as e:
        print(f"❌ Module zeroing failed: {e}")
        return module


def scale_module(module, scale):
    
    try:
        for p in module.parameters():
            p.detach().mul_(scale)
        return module
    except Exception as e:
        print(f"❌ Module scaling failed: {e}")
        return module


def mean_flat(tensor):
    
    # Support 2D, 3D, 4D, and 5D tensors
    if len(tensor.shape) == 5:  # 3Ddata: [B, C, D, H, W]
        return tensor.mean(dim=[1, 2, 3, 4])
    elif len(tensor.shape) == 4:  # 2Ddata: [B, C, H, W]
        return tensor.mean(dim=[1, 2, 3])
    elif len(tensor.shape) == 3:  # 1Ddata: [B, C, L]
        return tensor.mean(dim=[1, 2])
    else:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
   
    return GroupNorm32(32, channels)



class SiLU(nn.Module):
    """SiLU activation function - unchanged"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """32English textGroupNorm - 3DdataEnglish text"""
    def forward(self, x):
        # English text3DdataEnglish textGroupNorm
        if len(x.shape) == 5:  # 3Ddata
            # Rearrange to 2D for normalization
            original_shape = x.shape
            x = x.reshape(original_shape[0], original_shape[1], -1)  # [B, C, D*H*W]
            x = super().forward(x)
            x = x.reshape(original_shape)
        else:
            x = super().forward(x.float())
        return x.type(x.dtype)


def conv_nd(dims, *args, **kwargs):
   
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        # 3DEnglish textdata - English text
        if 'padding_mode' not in kwargs:
            kwargs['padding_mode'] = 'zeros'
        if 'bias' not in kwargs:
            kwargs['bias'] = True
            
        return nn.Conv3d(*args, **kwargs)
    elif dims == 4:
        raise ValueError("4DEnglish textdataEnglish text")
    raise ValueError(f"Unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    
    try:
        return nn.Linear(*args, **kwargs)
    except Exception as e:
        print(f"❌ English textCreateEnglish text: {e}")
        raise


def avg_pool_nd(dims, *args, **kwargs):
    
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)  # 3DEnglish textdata
    raise ValueError(f"Unsupported dimensions: {dims}")


class HybridConditioner(nn.Module):
   

    def __init__(self, c_concat_config, c_crossattn_config, c_porosity_config=None):
        super().__init__()
        print("🔧 Initialize hybrid conditioner...")
        
        try:
            self.concat_conditioner = instantiate_from_config(c_concat_config)
            self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)
            
            # Add porosity conditioner
            if c_porosity_config is not None:
                self.porosity_conditioner = instantiate_from_config(c_porosity_config)
                print("✅ Porosity conditioner enabled")
            else:
                self.porosity_conditioner = None
                print("⚠️ Porosity conditioner not enabled")
                
            print("✅ Hybrid conditioner initialization complete")
            
        except Exception as e:
            print(f"❌ Hybrid conditioner initialization failed: {e}")
            raise

    def forward(self, c_concat, c_crossattn, c_porosity=None):
       
        try:
            c_concat = self.concat_conditioner(c_concat)
            c_crossattn = self.crossattn_conditioner(c_crossattn)

            # Handle porosity condition
            if self.porosity_conditioner is not None and c_porosity is not None:
                c_porosity = self.porosity_conditioner(c_porosity)
                return {
                    'c_concat': [c_concat], 
                    'c_crossattn': [c_crossattn], 
                    'c_porosity': [c_porosity]
                }
            else:
                return {
                    'c_concat': [c_concat], 
                    'c_crossattn': [c_crossattn]
                }
                
        except Exception as e:
            print(f"❌ Hybrid conditioner forward pass failed: {e}")
            # Return safe default value
            return {
                'c_concat': [torch.zeros_like(c_concat)],
                'c_crossattn': [torch.zeros_like(c_crossattn)]
            }


def noise_like(shape, device, repeat=False):
   
    try:
        repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
        noise = lambda: torch.randn(shape, device=device)
        return repeat_noise() if repeat else noise()
    except Exception as e:
        print(f"❌ Noise generation failed: {e}")
        # Return safe default noise
        return torch.zeros(shape, device=device)



def create_3d_noise(shape, device, mean=0.0, std=1.0):
    
    try:
        noise = torch.randn(shape, device=device) * std + mean
        
        # Validate noise statistics
        if torch.isnan(noise).any() or torch.isinf(noise).any():
            print("⚠️ Warning: Noise contains NaN or Inf,cleaning up")
            noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return noise
        
    except Exception as e:
        print(f"❌ 3DEnglish textCreateEnglish text: {e}")
        return torch.zeros(shape, device=device)


def rock_data_preprocessing(rock_data, porosity, rock_type=None):
    
    try:
        # English textdataEnglish text
        device = rock_data.device
        
        # English textdataEnglish text[0,1]rangeEnglish text
        rock_data = torch.clamp(rock_data, 0, 1)
        
        # Handle porosity condition
        if isinstance(porosity, (int, float)):
            porosity_condition = torch.tensor([porosity], device=device).unsqueeze(0)
        else:
            porosity_condition = porosity.to(device)
            
        # Handle rock type condition
        conditions = {'porosity': porosity_condition}
        
        if rock_type is not None:
            if isinstance(rock_type, str):
                # Convert rock type to numerical value
                rock_type_map = {'sandstone': 0.0, 'limestone': 1.0}
                rock_type_value = rock_type_map.get(rock_type.lower(), 0.0)
                rock_type_condition = torch.tensor([rock_type_value], device=device).unsqueeze(0)
            else:
                rock_type_condition = rock_type.to(device)
            conditions['rock_type'] = rock_type_condition
        
        print(f"✅ English textdataEnglish text: shape{rock_data.shape}, porosity{porosity_condition.item()}")
        return rock_data, conditions
        
    except Exception as e:
        print(f"❌ English textdataEnglish text: {e}")
        # Return safe default value
        safe_data = torch.ones_like(rock_data) * 0.5
        safe_condition = torch.tensor([0.5], device=rock_data.device).unsqueeze(0)
        return safe_data, {'porosity': safe_condition}


def create_3d_positional_encoding(shape, device, scale=1.0):
    
    d, h, w = shape
    z_coords = torch.linspace(-1, 1, d, device=device)
    y_coords = torch.linspace(-1, 1, h, device=device) 
    x_coords = torch.linspace(-1, 1, w, device=device)
    
    z_grid, y_grid, x_grid = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    pos_encoding = torch.stack([x_grid, y_grid, z_grid], dim=0) * scale
    return pos_encoding


def normalize_3d_data(data, mean=None, std=None):
    
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
        
    # Prevent division by zero
    if std == 0:
        std = 1.0
        
    normalized = (data - mean) / std
    return normalized, mean, std


def denormalize_3d_data(normalized_data, mean, std):
   
    return normalized_data * std + mean


def check_3d_tensor_shape(tensor, expected_dims=5):
   
    if len(tensor.shape) != expected_dims:
        print(f"⚠️ 3DEnglish textshapeEnglish text: expected{expected_dims}D, actual{len(tensor.shape)}D")
        return False
        
    if any(dim == 0 for dim in tensor.shape):
        print(f"⚠️ 3D tensor contains zero dimension: {tensor.shape}")
        return False
        
    return True


create_noise_3d = create_3d_noise
preprocess_rock_data = rock_data_preprocessing


