
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities import rank_zero_only

# text
from ldm.util_new import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema_new import LitEma
from ldm.modules.distributions.distributions_new import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder_new8 import AutoencoderKL
from ldm.modules.diffusionmodules.util_new import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import Rock3DDDIMSampler

# text
__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

def disabled_train(self, mode=True):
    """text,text"""
    return self

def uniform_on_device(r1, r2, shape, device):
    """text"""
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    """textDDPMtext,text - 3Dtext"""
    
    def __init__(self,
                 unet_config,          # UNettext
                 timesteps=1000,       # text
                 beta_schedule="linear", # betatext
                 loss_type="l2",       # text
                 ckpt_path=None,       # text
                 ignore_keys=[],       # text
                 load_only_unet=False, # textUNet
                 monitor="val/loss",   # text
                 use_ema=True,         # textEMA
                 first_stage_key="image", # text
                 image_size=64,        # 3Dtext64
                 channels=4,           
                 log_every_t=100,      # text
                 clip_denoised=True,   # text
                 linear_start=1e-4,    # textbetatext
                 linear_end=2e-2,      # textbetatext
                 cosine_s=8e-3,        # text
                 given_betas=None,     # textbeta
                 original_elbo_weight=0., # ELBOtext
                 v_posterior=0.,       # text
                 l_simple_weight=1.,   # text
                 conditioning_key=None, # text
                 parameterization="eps", # text:epstextx0
                 scheduler_config=None, # text
                 use_positional_encodings=False, # text
                 learn_logvar=False,   # textlogvar
                 logvar_init=0.,       # logvartext
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'text"eps"text"x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: text {self.parameterization} text")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  
        self.channels = channels     
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"text {len(list(self.model_ema.buffers()))} textEMAtext")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
            
        # text
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        # text - text
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

    def safe_indexing(self, tensor, indices):
        """text - text"""
        if tensor.device != indices.device:
            indices = indices.to(tensor.device)
        return tensor[indices]


    def _ensure_all_tensors_on_device(self, device, *tensors):
        """text"""
        result = []
        for tensor in tensors:
            if tensor is not None and hasattr(tensor, 'to'):
                if tensor.device != device:
                    tensor = tensor.to(device)
            result.append(tensor)
        return result

    def _sync_timestep_tensor(self, t, device):
        """text"""
        if t.device != device:
            t = t.to(device)
        return t

    def _ensure_all_on_device(self, device):
        """text"""
        # text
        self.to(device)
        
        # text
        buffer_names = [
            'betas', 'alphas_cumprod', 'alphas_cumprod_prev',
            'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
            'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod',
            'sqrt_recipm1_alphas_cumprod', 'posterior_variance',
            'posterior_log_variance_clipped', 'posterior_mean_coef1',
            'posterior_mean_coef2', 'lvlb_weights'
        ]
        
        for buffer_name in buffer_names:
            if hasattr(self, buffer_name):
                buffer = getattr(self, buffer_name)
                if isinstance(buffer, torch.Tensor) and buffer.device != device:
                    print(f"🔧 text {buffer_name}: {buffer.device} -> {device}")
                    setattr(self, buffer_name, buffer.to(device))
        
        # textlogvartext
        if hasattr(self, 'logvar') and self.learn_logvar:
            if self.logvar.device != device:
                self.logvar = self.logvar.to(device)
                print(f"🔧 textlogvar: {self.logvar.device} -> {device}")
        
        print(f"✅ text: {device}")


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        """text - text"""
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphastexttimestepstext'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # text
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # text
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # text:textepsilontext
        posterior_variance = np.maximum(posterior_variance, 1e-20)
        
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(posterior_variance)))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # text
        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]  # text
        self.register_buffer('lvlb_weights', lvlb_weights)



    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        """text,textVAEtext"""
        try:
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
                
            # text,text
            model_sd = self.state_dict()
            filtered_sd = {}
            
            for k, v in sd.items():
                # text
                possible_keys = [
                    k,  # text
                    k.replace('first_stage_model.', ''),  # text
                    k.replace('model.', ''),  # text
                    k.replace('encoder.', ''),  # text  
                    k.replace('decoder.', ''),  # text
                ]
                
                for possible_key in possible_keys:
                    if possible_key in model_sd and model_sd[possible_key].shape == v.shape:
                        filtered_sd[possible_key] = v
                        break
            
            keys = list(filtered_sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print(f"text: {k}")
                        del filtered_sd[k]
                        
            missing, unexpected = self.load_state_dict(filtered_sd, strict=False) if not only_model else self.model.load_state_dict(filtered_sd, strict=False)
            print(f"text {path} text,text {len(filtered_sd)} text")
            print(f"text: {len(missing)},text: {len(unexpected)}")
            
        except Exception as e:
            print(f"text: {e},text")

            
    def q_mean_variance(self, x_start, t):
        """
        text q(x_t | x_0)
        :param x_start: text [N x C x ...] text
        :param t: text(text1),0text
        :return: (mean, variance, log_variance),textx_starttext
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        """text x0"""
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """text q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        """text"""
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        """text p(x_{t-1} | x_t) text"""
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # text t == 0 text
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        """text"""
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='text t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        """text"""
        if hasattr(self, 'image_size_3d'):
            image_size = self.image_size_3d
        else:
            # text3Dtext
            image_size = (16, 16, 16)
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, *image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        """text:q(x_t | x_0) - text"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        
  
        if torch.isnan(x_start).any() or torch.isinf(x_start).any():
            print("🚨 text: x_starttextNaNtextInf")
            x_start = torch.nan_to_num(x_start, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def get_loss(self, pred, target, mean=True):
        """text - text"""
       
        if pred.device != target.device:
            print(f"🔧 text: pred={pred.device}, target={target.device}")
            target = target.to(pred.device)
        
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("text '{loss_type}'")
    
        return loss



 


    def p_losses(self, x_start, cond, t, noise=None, extra_conditions=None):
        """text - text"""
  
        device = x_start.device
        x_start = x_start.float().to(device)
        if cond is not None:
            cond = cond.float().to(device)
        
        noise = default(noise, lambda: torch.randn_like(x_start)).to(device)
        t = t.to(device)
        
        # text
        sqrt_alpha = 0.9
        sqrt_one_minus_alpha = 0.4359
        
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        
        # text - text
        try:
            
            if extra_conditions is not None and isinstance(extra_conditions, dict):
                # print(f"🔍 p_losses: textextra_conditionstextapply_model")
                model_output = self.apply_model(x_noisy, t, cond, extra_conditions=extra_conditions)
            else:
                # print(f"🔍 p_losses: textextra_conditions")
                model_output = self.apply_model(x_noisy, t, cond)
        except Exception as e:
            print(f"❌ DDPM.p_lossestext: {e}")
            safe_loss = torch.tensor(0.1, device=device, requires_grad=True)
            return safe_loss, {'train/loss': safe_loss, 'train/loss_simple': safe_loss}
        
        # text
        model_output = model_output.to(device)
        
        # text
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        
        # textepstext
        target = noise
        target = target.to(device)
        
        # textMSEtext
        loss_simple = torch.nn.functional.mse_loss(model_output, target)
        loss_dict.update({f'{prefix}/loss_simple': loss_simple})
        
  
        loss = self.l_simple_weight * loss_simple
        loss_dict.update({f'{prefix}/loss': loss})
        
       
        return loss, loss_dict

    
    def _ensure_buffers_on_device(self, device):
     
        buffers_to_sync = [
            'betas', 'alphas_cumprod', 'alphas_cumprod_prev',
            'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
            'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod',
            'sqrt_recipm1_alphas_cumprod', 'posterior_variance',
            'posterior_log_variance_clipped', 'posterior_mean_coef1',
            'posterior_mean_coef2', 'lvlb_weights'
        ]
        
        for buffer_name in buffers_to_sync:
            if hasattr(self, buffer_name):
                buffer = getattr(self, buffer_name)
                if isinstance(buffer, torch.Tensor) and buffer.device != device:
                    print(f"🔧 text {buffer_name}: {buffer.device} -> {device}")
                    setattr(self, buffer_name, buffer.to(device))
        
     
        if hasattr(self, 'logvar') and self.learn_logvar:
            if self.logvar.device != device:
                print(f"🔧 textlogvar: {self.logvar.device} -> {device}")
                self.logvar = self.logvar.to(device)
        
        print(f"✅ text: {device}")
        


    def forward(self, x, c, extra_conditions=None):
        """text - text"""
        
        x = x.to(self.device)
        if c is not None:
            c = c.to(self.device)
        
        # text
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
 
      
        if self.model.conditioning_key is not None:
            assert c is not None, "textNone"
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        
      
        if extra_conditions is not None and not isinstance(extra_conditions, dict):
           
            if hasattr(extra_conditions, '__dict__'):
                extra_conditions = extra_conditions.__dict__
            else:
                extra_conditions = {'porosity': extra_conditions} if torch.is_tensor(extra_conditions) else {}
        
        return self.p_losses(x, c, t, extra_conditions=extra_conditions)

 

    def get_input(self, batch, k):
       
        x = batch[k]
        
        assert x.shape[1] == 1 and x.shape[2:] == (64, 64, 64), f"textshape: {x.shape}"


   
        # text
        x = x.to(memory_format=torch.contiguous_format).float()

        # text
        if hasattr(self, 'device'):
            x = x.to(self.device)
        
        
        return x




    def shared_step(self, batch, **kwargs):
        """text/text - text"""
        try:
           
            # text
            result = self.get_input(batch, self.first_stage_key)
            
            # text
            if len(result) == 2:
                z, c = result
                extra_conditions = None
            elif len(result) == 3:
                z, c, extra_conditions = result
            else:
                z, c = result[0], result[1]
                extra_conditions = result[2] if len(result) > 2 else None
            
           
            device = self.device
        
            # text
            if z.device != device:
                print(f"🔧 text: {z.device} -> {device}")
                z = z.to(device)
            
            # text
            if c is not None:
                if isinstance(c, (list, tuple)):
                 
                    c = self.get_learned_conditioning(c)
                elif hasattr(c, 'to'):
                    # text,text
                    if c.device != device:
                        # print(f"🔧 text: {c.device} -> {device}")
                        c = c.to(device)
                
                print(f"🔍 text: text{c.shape if hasattr(c, 'shape') else 'N/A'}, text{c.device if hasattr(c, 'device') else 'N/A'}")
            
            # text - textMCDDPMtext
            porosity_condition = None
            pore_size_mean_condition = None
            pore_size_std_condition = None
            
            if extra_conditions is not None and isinstance(extra_conditions, dict):
                # text
                if 'porosity' in extra_conditions:
                    porosity_condition = extra_conditions['porosity']
                    if porosity_condition.device != device:
                        porosity_condition = porosity_condition.to(device)
                
                # text
                if 'pore_size_mean' in extra_conditions:
                    pore_size_mean_condition = extra_conditions['pore_size_mean']
                    if pore_size_mean_condition.device != device:
                        pore_size_mean_condition = pore_size_mean_condition.to(device)
                
                # text
                if 'pore_size_std' in extra_conditions:
                    pore_size_std_condition = extra_conditions['pore_size_std']
                    if pore_size_std_condition.device != device:
                        pore_size_std_condition = pore_size_std_condition.to(device)
          
            self._ensure_model_on_device(device)
            
    
            print("🔍 text: textself(z, c)")
            valid_extra_conditions = None
            if extra_conditions is not None and isinstance(extra_conditions, dict):
              
                valid_extra_conditions = {}
                if 'porosity' in extra_conditions and extra_conditions['porosity'] is not None:
                    valid_extra_conditions['porosity'] = extra_conditions['porosity']
                if 'pore_size_mean' in extra_conditions and extra_conditions['pore_size_mean'] is not None:
                    valid_extra_conditions['pore_size_mean'] = extra_conditions['pore_size_mean']
                if 'pore_size_std' in extra_conditions and extra_conditions['pore_size_std'] is not None:
                    valid_extra_conditions['pore_size_std'] = extra_conditions['pore_size_std']
                if len(valid_extra_conditions) == 0:
                    valid_extra_conditions = None

            if valid_extra_conditions is not None:
                loss, loss_dict = self(z, c, extra_conditions=valid_extra_conditions)
            else:
                loss, loss_dict = self(z, c)
            
           
            return loss, loss_dict
            
        except Exception as e:
            print(f"🚨 shared_steptext: {e}")
            import traceback
            traceback.print_exc()
            
          
            safe_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
            safe_loss_dict = {'train/loss_simple': safe_loss}
            return safe_loss, safe_loss_dict
    
    def _ensure_model_on_device(self, device):
        """text"""
        print(f"🔧 text: {device}")
        
        # text
        if hasattr(self, 'cond_stage_model') and self.cond_stage_model is not None:
            if hasattr(self.cond_stage_model, 'to'):
                current_device = next(self.cond_stage_model.parameters()).device
                if current_device != device:
                    print(f"🔧 text: {current_device} -> {device}")
                    self.cond_stage_model = self.cond_stage_model.to(device)
          
            if hasattr(self.cond_stage_model, 'device'):
                self.cond_stage_model.device = device
                print(f"🔧 textCLIPtext: {device}")
        
       
        if hasattr(self, 'first_stage_model') and self.first_stage_model is not None:
            current_device = next(self.first_stage_model.parameters()).device
            if current_device != device:
                print(f"🔧 textVAE: {current_device} -> {device}")
                self.first_stage_model = self.first_stage_model.to(device)
      
        if hasattr(self, 'model') and self.model is not None:
            current_device = next(self.model.parameters()).device
            if current_device != device:
                print(f"🔧 textUNet: {current_device} -> {device}")
                self.model = self.model.to(device)
        
       
        self._ensure_all_on_device(device)
        
        print(f"✅ text")

    def training_step(self, batch, batch_idx):
      
        try:
            # text
            with torch.cuda.amp.autocast(enabled=True):
                loss, loss_dict = self.shared_step(batch)  # text
            
            # text
            if loss.dim() > 0:
                loss = loss.mean()
                
            # text
            if torch.isnan(loss) or torch.isinf(loss):
                print("🚨 text,text")
                loss = torch.tensor(0.1, device=self.device, requires_grad=True)
            
            self.log("train_loss", loss, prog_bar=True, logger=True, 
                     on_step=True, on_epoch=True, sync_dist=True)
            
            # text
            if self.use_scheduler:
                lr = self.optimizers().param_groups[0]['lr']
                self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            
            return loss
            
        except Exception as e:
            print(f"🚨 text: {e}")
            # text
            safe_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
            self.log("train_loss", safe_loss, prog_bar=True, logger=True)
            return safe_loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """text - text"""
        try:
            _, loss_dict_no_ema = self.shared_step(batch)
            with self.ema_scope():
                _, loss_dict_ema = self.shared_step(batch)
                loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
            self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        except Exception as e:
            print(f"🚨 text: {e}")


    
    @contextmanager
    def ema_scope(self, context=None):
        """EMAtext - text"""
        if self.use_ema:
            try:
                # text
                self.model.eval()
                
                # textEMAtext
                if hasattr(self, 'model_ema'):
                    self.model_ema.to(self.device)
                    self.model_ema(self.model)
                
                yield
            finally:
                # text
                self.model.train()
        else:
            yield



    def on_train_batch_end(self, *args, **kwargs):
        """textEMA"""
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        """text"""
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        """text - text,text"""
        log = dict()
        try:
            x = self.get_input(batch, self.first_stage_key)
            N = min(x.shape[0], N)
            x = x.to(self.device)[:N]
            log["inputs"] = x
            print(f"✅ text: inputstext={x.shape}")
        except Exception as e:
            print(f"⚠️ text: {e}")
            log["inputs"] = torch.zeros((1, 4, 16, 16, 16), device=self.device)
        
        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        """text - text3Dtext"""
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)  # text
        
        print(f"🔧 text: AdamW, lr={lr:.2e}, text={sum(p.numel() for p in params):,}")
        
        return opt


class LatentDiffusion(DDPM):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 use_porosity_condition=True,
                 porosity_dim=1,
                 base_learning_rate=3.0e-05,
                 condition_fusion_dim=128,  # text
                 unconditional_mode=False,  # text
                 *args, **kwargs):
        
        # text
        self.base_learning_rate = base_learning_rate
        self.use_porosity_condition = use_porosity_condition
        self.porosity_dim = porosity_dim
        self.condition_fusion_dim = condition_fusion_dim
        self.unconditional_mode = unconditional_mode
        
        # textcontext_dim
        if 'unet_config' in kwargs:
            self.context_dim = kwargs['unet_config'].get('params', {}).get('context_dim', 512)
        else:
            self.context_dim = 512
        
        # text(text)
        self.condition_encoder = None
        
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        
        # text(text)
        self.shorten_cond_schedule = getattr(self, 'shorten_cond_schedule', False)
        
      
        parent_kwargs = kwargs.copy()
       
        parent_kwargs.pop('condition_fusion', None)
        parent_kwargs.pop('condition_fusion_dim', None)
        parent_kwargs.pop('use_porosity_condition', None)
        parent_kwargs.pop('porosity_dim', None)
        parent_kwargs.pop('unconditional_mode', None)
        
        # text
        super().__init__(
            conditioning_key=conditioning_key,
            *args, 
            **parent_kwargs
        )
        
        # text
        self._init_condition_projection()
        
 
        self._validate_projection_dtypes()
   
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
            
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
            
        # text(VAE)
        self.instantiate_first_stage(first_stage_config)
        # text
        self.instantiate_cond_stage(cond_stage_config)
        
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        
        # text
        if self.use_porosity_condition:
            self._init_porosity_conditioning()
            
        print(f"✅ LatentDiffusiontext: text={self.channels}text, text={self.scale_factor}")

    
    def __call__(self, *args, **kwargs):
        
        # text
        if len(args) >= 2:
            x, c = args[0], args[1]
            extra_args = args[2:] if len(args) > 2 else []
        else:
            raise ValueError("textxtextctext")
        
        # textextra_conditions - text
        extra_conditions = kwargs.pop('extra_conditions', None)
        
        if extra_conditions is True:
            print("⚠️ textextra_conditions=True,text")
            extra_conditions = {}
        elif extra_conditions is False or extra_conditions is None:
            extra_conditions = None
        
     
        return self.forward(x, c, extra_conditions=extra_conditions)


    def _init_condition_projection(self):  
      
        print("🔧 text...")
        
        # textcontext_dim
        if not hasattr(self, 'context_dim'):
            # textUNettext
            if hasattr(self, 'model') and hasattr(self.model, 'diffusion_model'):
                self.context_dim = getattr(self.model.diffusion_model, 'context_dim', 768)
            else:
                self.context_dim = 768
            print(f"🔧 textcontext_dim: {self.context_dim}")
        
      
        if hasattr(self, 'cond_stage_model') and self.cond_stage_model is not None:
            try:
                # text
                if hasattr(self.cond_stage_model, 'output_dim'):
                    original_dim = self.cond_stage_model.output_dim
                else:
                    # CLIPtext768text
                    original_dim = 768
                
                print(f"🔧 text: {original_dim}, text: {self.context_dim}")
                
                if original_dim != self.context_dim:
                    self.text_proj = nn.Linear(original_dim, self.context_dim)
                    print(f"🔧 text: {original_dim} -> {self.context_dim}")
                else:
                    self.text_proj = nn.Identity()
                    print("🔧 text,textIdentitytext")
                    
            except AttributeError:
                print("⚠️ textcond_stage_modeltextoutput_dim,text")
                self.text_proj = nn.Linear(768, self.context_dim)
        else:
            print("🔧 textcond_stage_model,text")
            self.text_proj = nn.Identity()
        
     
        if self.use_porosity_condition:
            self.porosity_proj = nn.Linear(self.porosity_dim, self.context_dim)
            print(f"🔧 text: {self.porosity_dim} -> {self.context_dim}")
        else:
            self.porosity_proj = None
        
        print("✅ text")

    def _validate_projection_dtypes(self):  
        """text"""
        print("🔍 text...")
        
        # text
        if hasattr(self, 'porosity_proj') and self.porosity_proj is not None:
            weight_dtype = self.porosity_proj.weight.dtype
            bias_dtype = self.porosity_proj.bias.dtype if self.porosity_proj.bias is not None else weight_dtype
            
            print(f"  🎯 text: text{weight_dtype}, text{bias_dtype}")
            
            # textfloat32
            if weight_dtype != torch.float32:
                print(f"  🔄 text: {weight_dtype} -> float32")
                self.porosity_proj.weight.data = self.porosity_proj.weight.data.float()
                if self.porosity_proj.bias is not None:
                    self.porosity_proj.bias.data = self.porosity_proj.bias.data.float()
        
        # text
        if hasattr(self, 'text_proj') and self.text_proj is not None:
            if hasattr(self.text_proj, 'weight'):
                weight_dtype = self.text_proj.weight.dtype
                print(f"  🎯 text: text{weight_dtype}")
                
                if weight_dtype != torch.float32:
                    print(f"  🔄 text: {weight_dtype} -> float32")
                    self.text_proj.weight.data = self.text_proj.weight.data.float()
                    if hasattr(self.text_proj, 'bias') and self.text_proj.bias is not None:
                        self.text_proj.bias.data = self.text_proj.bias.data.float()
        
        print("✅ text")



    def _init_porosity_conditioning(self):  # text
        """text"""
        print(f"🔧 text,text: {self.porosity_dim}")
        
        # textUNettextcontext_dim
        if hasattr(self, 'model') and hasattr(self.model, 'diffusion_model'):
            context_dim = getattr(self.model.diffusion_model, 'context_dim', 768)
        else:
            context_dim = 768  # text
            
        print(f"🔧 text: {self.porosity_dim} -> {context_dim}")
        
        if self.use_porosity_condition:
            self.porosity_proj = nn.Linear(self.porosity_dim, context_dim)
            
            # text
            nn.init.normal_(self.porosity_proj.weight, std=0.02)
            if self.porosity_proj.bias is not None:
                nn.init.constant_(self.porosity_proj.bias, 0)
                
            print(f"✅ text: {self.porosity_dim} -> {context_dim}")
        else:
            self.porosity_proj = None
            print("❌ text")


    
    def instantiate_first_stage(self, config):
        """text(VAE)text"""
        try:
            model = instantiate_from_config(config)
            self.first_stage_model = model.eval()
            self.first_stage_model.train = disabled_train
            for param in self.first_stage_model.parameters():
                param.requires_grad = False
            print("✅ VAEtext")
            
            # textVAEtext
            if hasattr(self.first_stage_model, 'embed_dim'):
                print(f"🔍 VAEtext: {self.first_stage_model.embed_dim}")
            if hasattr(self.first_stage_model, 'use_conditions'):
                print(f"🔍 VAEtext: {self.first_stage_model.use_conditions}")
                
        except Exception as e:
            print(f"❌ VAEtext: {e}")
            raise

    def instantiate_cond_stage(self, config):
        """text"""
        # text,text,textNonetext
        if config is None:
            print("ℹ️ textcond_stage_config,textZeroTextConditionertext")
            class ZeroTextConditioner(nn.Module):
                def __init__(self, seq_len=77, dim=768, device=None):
                    super().__init__()
                    self.seq_len = seq_len
                    self.dim = dim
                    self.register_parameter('_dummy_param', nn.Parameter(torch.zeros(1)))
                    self.register_buffer('_dummy', torch.zeros(1))
                def forward(self, x):
                    if isinstance(x, (list, tuple)):
                        b = len(x)
                    elif isinstance(x, torch.Tensor) and x.dim() > 0:
                        b = x.shape[0]
                    else:
                        b = 1
                    return torch.zeros(b, self.seq_len, self.dim, device=self._dummy.device)
                __call__ = forward
            self.cond_stage_model = ZeroTextConditioner().to(self.device)
            for p in self.cond_stage_model.parameters():
                p.requires_grad = False
            print("✅ text ZeroTextConditioner text")
            return
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("text")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"text {self.__class__.__name__} text")
                self.cond_stage_model = None
               
                class ZeroTextConditioner(nn.Module):
                    def __init__(self, seq_len=77, dim=768, device=None):
                        super().__init__()
                        self.seq_len = seq_len
                        self.dim = dim
                        
                        self.register_parameter('_dummy_param', nn.Parameter(torch.zeros(1)))
                        self.register_buffer('_dummy', torch.zeros(1))
                    def forward(self, x):
                        if isinstance(x, (list, tuple)):
                            b = len(x)
                        elif isinstance(x, torch.Tensor) and x.dim() > 0:
                            b = x.shape[0]
                        else:
                            b = 1
                        return torch.zeros(b, self.seq_len, self.dim, device=self._dummy.device)
                    __call__ = forward
                try:
                    self.cond_stage_model = ZeroTextConditioner().to(self.device)
                    for p in self.cond_stage_model.parameters():
                        p.requires_grad = False
                    print("✅ text ZeroTextConditioner text")
                except Exception as e:
                    print(f"⚠️ ZeroTextConditioner text: {e}")
            else:
                try:
                    model = instantiate_from_config(config)
                    self.cond_stage_model = model.eval()
                    self.cond_stage_model.train = disabled_train
                    for param in self.cond_stage_model.parameters():
                        param.requires_grad = False
                    print("✅ text")
                except Exception as e:
                    print(f"❌ text: {e}")
                    
                    class ZeroTextConditioner(nn.Module):
                        def __init__(self, seq_len=77, dim=768, device=None):
                            super().__init__()
                            self.seq_len = seq_len
                            self.dim = dim
                            
                            self.register_parameter('_dummy_param', nn.Parameter(torch.zeros(1)))
                            self.register_buffer('_dummy', torch.zeros(1))
                        def forward(self, x):
                            if isinstance(x, (list, tuple)):
                                b = len(x)
                            elif isinstance(x, torch.Tensor) and x.dim() > 0:
                                b = x.shape[0]
                            else:
                                b = 1
                            return torch.zeros(b, self.seq_len, self.dim, device=self._dummy.device)
                        __call__ = forward
                    self.cond_stage_model = ZeroTextConditioner().to(self.device)
                    for p in self.cond_stage_model.parameters():
                        p.requires_grad = False
                    print("✅ text ZeroTextConditioner text")
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
            print("✅ text(text)")


    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        """text"""
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"text '{type(encoder_posterior)}' text")
        
        print(f"🔍 text: {z.shape}")
        
        # text3Dtext [B, 4, D, H, W]
        if z.dim() == 5:
            # text
            current_depth = z.shape[2]
            target_depth = 16  # textVAEtext:64 / 4 = 16
            
            if current_depth != target_depth:
                print(f"🔧 text: {current_depth} -> {target_depth}")
                
                # text3Dtext
                if current_depth == 1:
                    # text,text
                    z = z.repeat(1, 1, target_depth, 1, 1)
                    print(f"✅ text: {z.shape}")
                else:
                    # text
                    z = F.interpolate(z, size=(target_depth, z.shape[3], z.shape[4]), 
                                    mode='trilinear', align_corners=False)
                    print(f"✅ text: {z.shape}")
        
     
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("🚨 text: textNaNtextInf,text")
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
            
        print(f"✅ text: {z.shape}")
        return self.scale_factor * z




    def get_learned_conditioning(self, c):
   
        if self.unconditional_mode:
            print("🔧 text:text")
            batch_size = 1
            if isinstance(c, list):
                batch_size = len(c)
            elif isinstance(c, torch.Tensor) and c.dim() > 0:
                batch_size = c.shape[0]
            # cond_stage_model text None;text
            device = self.device
            return torch.zeros(batch_size, 77, 768, device=device)
        
        print(f"🔍 text: text{type(c)}, text: {len(c) if isinstance(c, list) else 'N/A'}")
        
        if getattr(self, 'cond_stage_model', None) is None:
            batch_size = 1
            if isinstance(c, list):
                batch_size = len(c)
            elif isinstance(c, torch.Tensor) and c.dim() > 0:
                batch_size = c.shape[0]
            default_cond = torch.zeros(batch_size, 77, 768, device=self.device)
            print(f"🔧 cond_stage_modeltext,text: {default_cond.shape}")
            return default_cond
        
        # text
        device = self.device
        if hasattr(self.cond_stage_model, 'to'):
            self.cond_stage_model = self.cond_stage_model.to(device)
        
        try:
    
            if isinstance(c, (str, list)):
                # text - text
                print(f"🔧 text: {len(c) if isinstance(c, list) else 1} text")
                c_encoded = self.cond_stage_model(c)
            elif isinstance(c, torch.Tensor):
                # text - text
                print(f"🔍 text: text{c.shape}, text{c.dtype}, text{c.device}")
                if c.dim() == 3 and c.shape[1] == 77 and c.shape[2] == 768:
                    print("🔧 text")
                    c_encoded = c.to(device)
                else:
                    # text,text
                    batch_size = c.shape[0] if c.dim() > 0 else 1
                    c_encoded = torch.zeros(batch_size, 77, 768, device=device)
            else:
                # text,text
                batch_size = len(c) if isinstance(c, list) else 1
                c_encoded = torch.zeros(batch_size, 77, 768, device=device)
            
            print(f"✅ text: text{c_encoded.shape}, text{c_encoded.device}")
            return c_encoded
            
        except Exception as e:
            print(f"❌ text: {e}")
            import traceback
            traceback.print_exc()
            batch_size = len(c) if isinstance(c, list) else 1
            default_cond = torch.zeros(batch_size, 77, 768, device=device)
            print(f"🔧 text: text{default_cond.shape}")
            return default_cond

    def meshgrid(self, h, w):
        """text"""
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)
        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        text
        :param h: text
        :param w: text
        :return: text,text0,text0.5
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        """text"""
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])
            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):
        """
        text
        :param x: text (bs, c, h, w)
        :return: ntext (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # text
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)
            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)
            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # text
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)
            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)
            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)
            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)
            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

   
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        """
        text - text
        """
        # text
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
            
        x = x.to(self.device)
        
        # text
        c = None
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
                
            xc = None
            if 'txt' in batch:
                xc = batch['txt']
            elif 'conditions' in batch and 'text' in batch['conditions']:
                xc = batch['conditions']['text']
            else:
                # text
                batch_size = x.shape[0]
                xc = [""] * batch_size
                
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc)
            else:
                c = xc
                
            if bs is not None:
                c = c[:bs]
        
        # text
        extra_conditions = None
        if self.use_porosity_condition:
            if 'porosity' in batch:
                porosity_condition = batch['porosity'].to(self.device)
            elif 'conditions' in batch and 'porosity' in batch['conditions']:
                porosity_condition = batch['conditions']['porosity'].to(self.device)
            else:
                porosity_condition = None
            
            if porosity_condition is not None:
                # text
                if porosity_condition.dim() == 1:
                    porosity_condition = porosity_condition.unsqueeze(1)
                
                if porosity_condition.dtype != torch.float32:
                    porosity_condition = porosity_condition.float()
                
                extra_conditions = {'porosity': porosity_condition}
        
      
        vae_conditions = None
        if extra_conditions is not None and 'porosity' in extra_conditions:
        
            try:
                from main_new import create_128d_conditions
                # text
                if c is not None and hasattr(c, 'shape') and c.shape[1] > 0:
                 
                    batch_size = x.shape[0]
                    rock_types = []
                    for i in range(batch_size):
                        # text:text
                        rock_types.append('limestone' if i % 2 == 0 else 'sandstone')
                    
                    # text
                    # text/text/text,textfloattext
                    try:
                        porosity_value = float(porosity_condition.detach().reshape(-1)[0].item())
                    except Exception:
                        try:
                            porosity_value = float(porosity_condition)
                        except Exception:
                            porosity_value = 0.0

                    conditions_dict = {
                        'rock_type': rock_types[0],  # text:text
                        'porosity': porosity_value  # text
                    }
                    
                   
                    vae_conditions = create_128d_conditions(conditions_dict, self.device)
            except Exception as e:
                print(f"⚠️ textVAEtext: {e}")
                vae_conditions = None
        
       
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        
       
        return z, c, extra_conditions

    
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False, conditions=None):
 
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z


        if torch.isnan(z).any() or torch.isinf(z).any():
            print("🚨 text: textNaNtextInf")
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            if hasattr(self.first_stage_model, "decode") and callable(self.first_stage_model.decode):
                # textVAEtextdecodetext(textconditions)
                x = self.first_stage_model.decode(z)
            else:
              
                x = self.first_stage_model(z, regularize=False)
                
            # text
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("🚨 text: textNaNtextInf")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                
            return x
            
        except Exception as e:
            print(f"❌ text: {e}")
            # text
            return torch.zeros(z.shape[0], 1, 64, 64, 64, device=z.device)


    @torch.no_grad()
    def encode_first_stage(self, x, conditions=None):
        """text - text,text"""
        try:
           
            if x.dim() == 5:
                # text [B, C, D, H, W]
                if x.shape[1] != 1:
                    print(f"🚨 text: text! text1text, text{x.shape[1]}text")
                    print(f"🔄 text: {x.shape} -> ", end="")
                    
                    # text (B, 64, 1, 64, 64) text,text
                    if x.shape[2] == 1 and x.shape[3] == 64 and x.shape[4] == 64:
                        # (B, 64, 1, 64, 64) -> (B, 1, 64, 64, 64)
                        x = x.transpose(1, 2)  # text
                        print(f"{x.shape}")
                    else:
                        # text,text
                        x = x[:, :1]  # text
                        print(f"{x.shape}")
                
                # text
                if x.shape[2] != 64:
                    print(f"🔄 text: {x.shape[2]} -> 64")
                    x = F.interpolate(x, size=(64, 64, 64), mode='trilinear')
                    
            elif x.dim() == 4:
                # 4Dtext,text
                print(f"🔄 4Dtext5D: {x.shape} -> ", end="")
                x = x.unsqueeze(2)  # text [B, C, 1, H, W]
                print(f"{x.shape}")
                
                # text64
                if x.shape[2] != 64:
                    print(f"🔄 text: {x.shape[2]} -> 64")
                    x = x.repeat(1, 1, 64, 1, 1)
            

            
            if hasattr(self.first_stage_model, "encode") and callable(self.first_stage_model.encode):
                # textconditions,text
                h = self.first_stage_model.encode(x)
                if isinstance(h, DiagonalGaussianDistribution):
                    return h
                else:
                    return DiagonalGaussianDistribution(h)
            else:
                return self.first_stage_model(x)
                
        except Exception as e:
            print(f"❌ text: {e}")
            import traceback
            traceback.print_exc()
            # text
            device = x.device
            batch_size = x.shape[0]
            safe_moments = torch.zeros(batch_size, 2*4, 16, 16, 16, device=device)
            return DiagonalGaussianDistribution(safe_moments)



    def shared_step(self, batch, **kwargs):
 
        try:
            print("🔍 text: textshared_step")
            
            # text
            result = self.get_input(batch, self.first_stage_key)
            
            # text
            if len(result) == 2:
                z, c = result
                extra_conditions = None
            elif len(result) == 3:
                z, c, extra_conditions = result
            else:
                z, c = result[0], result[1]
                extra_conditions = result[2] if len(result) > 2 else None
            
            print(f"🔍 text: ztext={z.shape}, text{z.device}")
            print(f"🔍 text: ctext={type(c)}, extra_conditionstext={type(extra_conditions)}")
            
            # text:text
            device = self.device
            print(f"🔍 text: {device}")
            
            # text
            if z.device != device:
                print(f"🔧 text: {z.device} -> {device}")
                z = z.to(device)
            
            # text
            if c is not None:
                if isinstance(c, (list, tuple)):
                    # text,text
                    print("🔧 text...")
                    c = self.get_learned_conditioning(c)
                elif hasattr(c, 'to'):
                    # text,text
                    if c.device != device:
                        print(f"🔧 text: {c.device} -> {device}")
                        c = c.to(device)
                
                print(f"🔍 text: text{c.shape if hasattr(c, 'shape') else 'N/A'}, text{c.device if hasattr(c, 'device') else 'N/A'}")
            
       
            porosity_condition = None
            if extra_conditions is not None and isinstance(extra_conditions, dict) and 'porosity' in extra_conditions:
                porosity_condition = extra_conditions['porosity']
                if porosity_condition.device != device:
                    print(f"🔧 text: {porosity_condition.device} -> {device}")
                    porosity_condition = porosity_condition.to(device)
                
                print(f"🔍 text: text{porosity_condition.shape}, text{porosity_condition.device}")
            
           
            self._ensure_model_on_device(device)
         
            print("🔍 text: textself(z, c)")
            if porosity_condition is not None:
               
                valid_extra_conditions = {'porosity': porosity_condition}
                loss, loss_dict = self(z, c, extra_conditions=valid_extra_conditions)
            else:
                loss, loss_dict = self(z, c)
            
            print(f"✅ shared_steptext: loss={loss.item()}")
            return loss, loss_dict
            
        except Exception as e:
            print(f"🚨 shared_steptext: {e}")
            import traceback
            traceback.print_exc()
            
            # text
            safe_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
            safe_loss_dict = {'train/loss_simple': safe_loss}
            return safe_loss, safe_loss_dict



    def forward(self, x, c, extra_conditions=None):
       
        print(f"🔍 forward: text, xtext={x.shape}, ctext={c.shape}, extra_conditionstext={type(extra_conditions)}")
        
        # text
        x = x.to(self.device)
        if c is not None:
            c = c.to(self.device)
        
        # text
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        # print(f"🔍 forward: text={t.shape}")
        
      
        if self.model.conditioning_key is not None:
            assert c is not None, "textNone"
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        
        
       
        if extra_conditions is not None and not isinstance(extra_conditions, dict):
            print(f"⚠️ textextra_conditionstext: {type(extra_conditions)} -> dict")
            if hasattr(extra_conditions, '__dict__'):
                extra_conditions = extra_conditions.__dict__
            else:
                extra_conditions = {'porosity': extra_conditions} if torch.is_tensor(extra_conditions) else {}
        
        return self.p_losses(x, c, t, extra_conditions=extra_conditions)



    def _rescale_annotations(self, bboxes, crop_coordinates):
        """text"""
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    
    def apply_model(self, x_noisy, t, cond, extra_conditions=None, return_ids=False):
      
        porosity_condition = None
        pore_size_mean_condition = None
        pore_size_std_condition = None
        
        if extra_conditions is not None:
            # text
            if 'porosity' in extra_conditions:
                porosity_condition = extra_conditions['porosity'].to(x_noisy.device)
                if porosity_condition.dtype != x_noisy.dtype:
                    porosity_condition = porosity_condition.to(x_noisy.dtype)
            
            # text
            if 'pore_size_mean' in extra_conditions:
                pore_size_mean_condition = extra_conditions['pore_size_mean'].to(x_noisy.device)
                if pore_size_mean_condition.dtype != x_noisy.dtype:
                    pore_size_mean_condition = pore_size_mean_condition.to(x_noisy.dtype)
            
            # text
            if 'pore_size_std' in extra_conditions:
                pore_size_std_condition = extra_conditions['pore_size_std'].to(x_noisy.device)
                if pore_size_std_condition.dtype != x_noisy.dtype:
                    pore_size_std_condition = pore_size_std_condition.to(x_noisy.dtype)
        
        # textUNet 
        try:
            if hasattr(self.model, 'diffusion_model'):
                # textUNet,text
                x_recon = self.model.diffusion_model(
                    x_noisy, 
                    t, 
                    porosity_condition=porosity_condition,
                    pore_size_mean_condition=pore_size_mean_condition,
                    pore_size_std_condition=pore_size_std_condition
                )
            else:
                # text
                x_recon = self.model(
                    x_noisy, 
                    t, 
                    porosity_condition=porosity_condition,
                    pore_size_mean_condition=pore_size_mean_condition,
                    pore_size_std_condition=pore_size_std_condition
                )
                
         
            
            if isinstance(x_recon, tuple) and not return_ids:
                return x_recon[0]
            else:
                return x_recon
                
        except Exception as e:
            print(f"❌ apply_modeltext: {e}")
            import traceback
            traceback.print_exc()
            
            # text
            return torch.zeros_like(x_noisy)


    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """textx0texteps"""
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """textKLtext"""
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device).long()
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)





    def p_losses(self, x_start, cond, t, noise=None, extra_conditions=None):
       
        device = x_start.device
        x_start = x_start.float().to(device)
        if cond is not None:
            cond = cond.float().to(device)
        
        noise = default(noise, lambda: torch.randn_like(x_start)).to(device)
        t = t.to(device)
        
        sqrt_alpha = 0.9
        sqrt_one_minus_alpha = 0.4359
        
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        
       
        try:
           
            if extra_conditions is not None and isinstance(extra_conditions, dict):
                
                model_output = self.apply_model(x_noisy, t, cond, extra_conditions=extra_conditions)
            else:
                print(f"🔍 p_losses: textextra_conditions")
                model_output = self.apply_model(x_noisy, t, cond)
        except Exception as e:
            print(f"❌ DDPM.p_lossestext: {e}")
            safe_loss = torch.tensor(0.1, device=device, requires_grad=True)
            return safe_loss, {'train/loss': safe_loss, 'train/loss_simple': safe_loss}
        
        # text
        model_output = model_output.to(device)
        
      
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        
        # textepstext
        target = noise
        target = target.to(device)
        
        # textMSEtext
        loss_simple = torch.nn.functional.mse_loss(model_output, target)
        loss_dict.update({f'{prefix}/loss_simple': loss_simple})
        
        # text
        loss = self.l_simple_weight * loss_simple
        loss_dict.update({f'{prefix}/loss': loss})
        
       
        return loss, loss_dict





    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None, extra_conditions=None):
        """text"""
        t_in = t
        model_out = self.apply_model(x, t_in, c, extra_conditions=extra_conditions, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                 extra_conditions=None):
        """text p(x_{t-1} | x_t) text"""
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs,
                                       extra_conditions=extra_conditions)
        if return_codebook_ids:
            raise DeprecationWarning("text return_codebook_ids")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # text t == 0 text
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        """text"""
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size
            shape = [batch_size] + list(shape)
        else:
            b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='text',
                        total=timesteps) if verbose else reversed(range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(self.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                     x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                     mask=None, x0=None, img_callback=None, start_T=None,
                     log_every_t=None, extra_conditions=None):
        """text"""
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='text t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # text

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised,
                                extra_conditions=extra_conditions)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, extra_conditions=None, **kwargs):
        """text"""
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0, extra_conditions=extra_conditions)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        """text"""
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, **kwargs):
        """text"""
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
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log["conditioning"] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # text
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
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # text
            with self.ema_scope("text"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # text
                with self.ema_scope("text"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # text
                h, w = z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # text1/8text
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("text"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta,
                                                mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # text
                mask = 1. - mask
                with self.ema_scope("text"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta,
                                                mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("text"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="text")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        """text - text"""
        lr = self.base_learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: text!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('text')
            params.append(self.logvar)
            
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        
        print(f"🔧 text: AdamW, lr={lr:.2e}, text={sum(p.numel() for p in params):,}")
        
        if self.use_scheduler:
            print("🔧 text...")
            try:
                # text
                from ldm.lr_scheduler_new import create_safe_scheduler
                scheduler_instance = create_safe_scheduler(
                    {'scheduler_config': self.scheduler_config},
                    total_steps=1000000  # text
                )
                
                scheduler_config = {
                    'scheduler': torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=scheduler_instance),
                    'interval': 'step',
                    'frequency': 1
                }
                
                print("✅ text")
                return [opt], [scheduler_config]
                
            except Exception as e:
                print(f"❌ text: {e},text")
                return opt
                
        return opt


    @torch.no_grad()
    def to_rgb(self, x):
        """textRGBtext"""
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

class DiffusionWrapper(pl.LightningModule):
    """text - text"""
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, porosity_condition=None):
        """text - text"""
        # print(f"🔍 DiffusionWrapper: xtext={x.shape}, ttext={t.shape}")
        # print(f"🔍 DiffusionWrapper: c_concattext={type(c_concat)}, c_crossattntext={type(c_crossattn)}")
        # print(f"🔍 DiffusionWrapper: porosity_condition={porosity_condition is not None}")
        
        # text:textNone
        if c_concat is None:
            c_concat = []
        if c_crossattn is None:
            c_crossattn = []
        
        # text:text
        context = None
        if c_crossattn and len(c_crossattn) > 0 and c_crossattn[0] is not None:
            context = c_crossattn[0]
            # print(f"🔍 text: {context.shape}")
        
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t, porosity_condition=porosity_condition)
        elif self.conditioning_key == 'concat':
            # text:textconcattext
            valid_concat = [tensor for tensor in c_concat if tensor is not None]
            if valid_concat:
                concat_tensor = valid_concat[0]
                # print(f"🔍 Concattext: {concat_tensor.shape}")
                
                # textconcattextxtext
                if concat_tensor.shape[2:] != x.shape[2:]:
                    print(f"🔧 textconcattext: {concat_tensor.shape[2:]} -> {x.shape[2:]}")
                    concat_tensor = F.interpolate(
                        concat_tensor, 
                        size=x.shape[2:], 
                        mode='trilinear' if x.dim() == 5 else 'bilinear'
                    )
                
                xc = torch.cat([x, concat_tensor], dim=1)
                print(f"🔍 Concattext: {xc.shape}")
            else:
                xc = x
            out = self.diffusion_model(xc, t, porosity_condition=porosity_condition)
        elif self.conditioning_key == 'crossattn':
           
            if context is not None:
                # textcontexttext
                if context.dim() == 3 and context.shape[1] != 77:
                    print(f"🔧 textcontexttext: {context.shape[1]} -> 77")
                    if context.shape[1] > 77:
                        context = context[:, :77, :]
                    else:
                        batch_size, seq_len, feature_dim = context.shape
                        padding = torch.zeros(batch_size, 77 - seq_len, feature_dim, 
                                            device=context.device, dtype=context.dtype)
                        context = torch.cat([context, padding], dim=1)
                
                print(f"🔍 text: text{context.shape}")
                out = self.diffusion_model(x, t, context=context, porosity_condition=porosity_condition)
            else:
                # text,text
                batch_size = x.shape[0]
                default_context = torch.zeros(batch_size, 77, 768, device=x.device, dtype=x.dtype)
                print(f"🔍 text: text{default_context.shape}")
                out = self.diffusion_model(x, t, context=default_context, porosity_condition=porosity_condition)
        elif self.conditioning_key == 'hybrid':
            # text
            valid_concat = [tensor for tensor in c_concat if tensor is not None]
            valid_crossattn = [tensor for tensor in c_crossattn if tensor is not None]
            
            if valid_concat:
                xc = torch.cat([x] + valid_concat, dim=1)
            else:
                xc = x
                
            if valid_crossattn:
                cc = torch.cat(valid_crossattn, 1)
            else:
                batch_size = x.shape[0]
                cc = torch.zeros(batch_size, 1, 768, device=x.device, dtype=x.dtype)
                
            out = self.diffusion_model(xc, t, context=cc, porosity_condition=porosity_condition)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0] if c_crossattn and c_crossattn[0] is not None else torch.zeros(x.shape[0], 1, device=x.device)
            out = self.diffusion_model(x, t, y=cc, porosity_condition=porosity_condition)
        else:
            raise NotImplementedError()
    
        print(f"✅ DiffusionWrappertext: {out.shape}")
        return out

class LatentInpaintDiffusion(LatentDiffusion):
  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model.conditioning_key == 'concat', 'textconcattext'

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, *args, **kwargs):
        
        # text
        x = super().get_input(batch, self.first_stage_key, *args, **kwargs)  # text (B, 1, 64, 64, 64)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
    
        # text
        mask = batch['mask'].to(self.device)
        if bs is not None:
            mask = mask[:bs]
        # textmasktext (B, 1, 64, 64, 64)
        # text,textrearrange
        assert mask.shape[1:] == (1, 64, 64, 64), f"Mask shapetext: {mask.shape}"
    
        # text
        masked_x = batch['masked_image'].to(self.device)
        if bs is not None:
            masked_x = masked_x[:bs]
        # textmasked_xtext (B, 1, 64, 64, 64)
        assert masked_x.shape[1:] == (1, 64, 64, 64), f"Masked image shapetext: {masked_x.shape}"
    
        # text
        encoder_posterior_masked = self.encode_first_stage(masked_x)
        masked_z = self.get_first_stage_encoding(encoder_posterior_masked).detach()
    
        # text
        # textmasked_ztextmasktextshapetext (B, C, D, H, W)
        cond = torch.cat([masked_z, mask], dim=1)
        return z, cond

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        """text"""
        log = super().log_images(*args, **kwargs)
        # text
        z, cond = self.get_input(args[0], self.first_stage_key, return_first_stage_outputs=True)
        mask = cond[:, -1:, ...]
        masked_z = cond[:, :-1, ...]
        log["mask"] = mask
        log["masked_image"] = self.decode_first_stage(masked_z)
        return log

class LatentUpscaleDiffusion(LatentDiffusion):
    """text"""
    def __init__(self, *args, low_scale_config, low_scale_key="LR", **kwargs):
        super().__init__(*args, **kwargs)
        # text
        self.instantiate_low_stage(low_scale_config)
        self.low_scale_key = low_scale_key

    def instantiate_low_stage(self, config):
        """text"""
        model = instantiate_from_config(config)
        self.low_stage_model = model.eval()
        self.low_stage_model.train = disabled_train
        for param in self.low_stage_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, *args, **kwargs):
        """text,text"""
        # text
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        if bs is not None:
            x = x[:bs]
            c = c[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        # text
        x_low = batch[self.low_scale_key].to(self.device)
        if bs is not None:
            x_low = x_low[:bs]
        x_low = rearrange(x_low, 'b h w c -> b c h w')
        x_low = x_low.to(self.device)
        # text
        z_low = self.get_first_stage_encoding(self.encode_first_stage(x_low))

        # text
        cond = torch.cat([c, z_low], dim=1)
        return z, cond

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        """text"""
        log = super().log_images(*args, **kwargs)
        # text
        z, cond = self.get_input(args[0], self.first_stage_key, return_first_stage_outputs=True)
        z_low = cond[:, -self.channels:, ...]
        log["low_res"] = self.decode_first_stage(z_low)
        return log




