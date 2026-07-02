
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

# Provide fallback implementation to avoid full dependency
try:
    from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
    from taming.modules.losses.lpips import LPIPS
except ImportError:
    print("Warning: taming module not found, using fallback implementation")
    
    # Simplified weight initialization
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    # Simplified LPIPS implementation
    class LPIPS(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = nn.L1Loss()
        
        def forward(self, x, y):
            return self.loss_fn(x, y)
    
    # Simplified discriminator
    class NLayerDiscriminator(nn.Module):
        def __init__(self, input_nc=1, n_layers=3, ndf=64, use_actnorm=False):
            super().__init__()
            layers = []
            layers.append(nn.Conv2d(input_nc, ndf, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.2, True))
            
            for n in range(1, n_layers):
                layers.append(nn.Conv2d(ndf, ndf * 2, 4, 2, 1))
                layers.append(nn.LeakyReLU(0.2, True))
                ndf *= 2
            
            layers.append(nn.Conv2d(ndf, 1, 4, 1, 1))
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)


# Loss function implementation
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) + 
        torch.mean(F.softplus(logits_fake))
    )
    return d_loss

def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1. - logits_real), dim=[1,2,3,4])  # modified to 4 dimensions for 3D
    loss_fake = torch.mean(F.relu(1. + logits_fake), dim=[1,2,3,4])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def measure_perplexity(predicted_indices, n_embed):
    # Evaluate cluster perplexity
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use

def l1(x, y):
    return torch.abs(x-y)

def l2(x, y):
    return torch.pow((x-y), 2)

def exists(val):
    return val is not None


class NLayer3DDiscriminator(nn.Module):
    """
    3D version of multi-layer discriminator, specialized for 3D rock data
    """
    
    def __init__(self, input_nc=1, n_layers=3, ndf=64, use_actnorm=False):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Conv3d(input_nc, ndf, kernel_size=4, stride=2, padding=1))
        if use_actnorm:
            layers.append(nn.BatchNorm3d(ndf))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, 
                                   kernel_size=4, stride=2, padding=1))
            if use_actnorm:
                layers.append(nn.BatchNorm3d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Last layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, 
                               kernel_size=4, stride=1, padding=1))
        if use_actnorm:
            layers.append(nn.BatchNorm3d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer
        layers.append(nn.Conv3d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class VQLPIPSWithDiscriminator3D(nn.Module):
   
    
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=1, disc_factor=1.0, disc_weight=0.5,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", n_classes=None, perceptual_loss="lpips",
                 pixel_loss="l1", is_3d=True, unconditional_mode=False):
        """
        Initialize 3D VQ perceptual loss
        
        Parameters:
        - disc_start: step at which discriminator starts training
        - disc_in_channels: input channel count (1=single-channel rock data)
        - disc_weight: discriminator weight (0.5 for balancing)
        - is_3d: whether data is 3D
        - unconditional_mode: whether unconditional mode
        """
        super().__init__()
        
        assert disc_loss in ["hinge", "vanilla"]
        assert perceptual_loss in ["lpips", "clips", "dists"]
        assert pixel_loss in ["l1", "l2"]
        
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.is_3d = is_3d
        self.unconditional_mode = unconditional_mode
        
        # Perceptual loss config
        if perceptual_loss == "lpips":
            print(f"{self.__class__.__name__}: Running with LPIPS.")
            if is_3d:
                # 3D data uses simplified perceptual loss
                self.perceptual_loss = self._dummy_perceptual_loss
            else:
                self.perceptual_loss = LPIPS().eval()
        else:
            raise ValueError(f"Unknown perceptual loss: >> {perceptual_loss} <<")
        
        self.perceptual_weight = perceptual_weight

        # Pixel loss config
        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        # Select 2D or 3D discriminator
        if is_3d:
            self.discriminator = NLayer3DDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm
            ).apply(weights_init)
        else:
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm,
                ndf=disc_ndf
            ).apply(weights_init)
        
        self.discriminator_iter_start = disc_start
        
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        
        print(f"VQLPIPSWithDiscriminator3D running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional and not unconditional_mode  # unconditional mode disables conditional discriminator
        self.n_classes = n_classes
        
        # Last layer
        self.last_layer = None
        
        # Unconditional mode logging
        if unconditional_mode:
            print("🔧 Unconditional mode: disable conditional discriminator, simplify VQ loss computation")
    
    def _dummy_perceptual_loss(self, inputs, reconstructions):
        """
        Simplified perceptual loss for 3D data(replaces LPIPS)
        """
        # Use L1 loss as perceptual loss substitute for 3D data
        return torch.abs(inputs - reconstructions).mean()
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Compute adaptive weight between generator and reconstruction loss
        """
        try:
            if last_layer is not None and len(last_layer) > 0:
                # Check if last_layer contains tensors that require gradients
                valid_layers = [layer for layer in last_layer if layer.requires_grad]
                if valid_layers:
                    nll_grads = torch.autograd.grad(nll_loss, valid_layers[0], retain_graph=True, allow_unused=True)[0]
                    g_grads = torch.autograd.grad(g_loss, valid_layers[0], retain_graph=True, allow_unused=True)[0]
                else:
                    # If no layers require gradients, return fixed weight
                    return torch.tensor(self.discriminator_weight, device=nll_loss.device)
            elif hasattr(self, 'last_layer') and self.last_layer is not None and len(self.last_layer) > 0:
                valid_layers = [layer for layer in self.last_layer if layer.requires_grad]
                if valid_layers:
                    nll_grads = torch.autograd.grad(nll_loss, valid_layers[0], retain_graph=True, allow_unused=True)[0]
                    g_grads = torch.autograd.grad(g_loss, valid_layers[0], retain_graph=True, allow_unused=True)[0]
                else:
                    return torch.tensor(self.discriminator_weight, device=nll_loss.device)
            else:
                # If no valid layers, return fixed weight
                return torch.tensor(self.discriminator_weight, device=nll_loss.device)
            
            # Check if gradients are valid
            if nll_grads is None or g_grads is None:
                return torch.tensor(self.discriminator_weight, device=nll_loss.device)
            
            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            d_weight = d_weight * self.discriminator_weight
            return d_weight
            
        except Exception as e:
            print(f"Adaptive weight computation failed: {e},using fixed weight")
            return torch.tensor(self.discriminator_weight, device=nll_loss.device)

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None):
        """
        Forward pass for loss computation
        
        Parameters:
        - codebook_loss: codebook loss
        - inputs: original input data [B, C, D, H, W]
        - reconstructions: reconstruction data [B, C, D, H, W]
        - optimizer_idx: optimizer index (0=generator, 1=discriminator)
        """
        
        if not exists(codebook_loss):
            codebook_loss = torch.tensor([0.]).to(inputs.device)
        
        # Reconstruction loss
        rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0]).to(inputs.device)

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        # Last layer
        self.last_layer = last_layer

        # Unconditional mode: simplify condition handling
        if self.unconditional_mode:
            cond = None
            self.disc_conditional = False
        
        # GAN section
        if optimizer_idx == 0:
            # generator
            if self.unconditional_mode or cond is None:
                # :reconstruction data
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                # Conditional mode: concatenate condition information
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                # Adaptive weight computation failed,using fixed weight
                d_weight = torch.tensor(self.discriminator_weight, device=nll_loss.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            #  = Reconstruction loss + discriminator + codebook loss
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            # Log metrics
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/p_loss".format(split): p_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            
            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
            
            return loss, log

        elif optimizer_idx == 1:
            # discriminator
            if self.unconditional_mode or cond is None:
                # Unconditional mode: directly use input and reconstruction
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                # Conditional mode: concatenate condition information
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }
            
            return d_loss, log


# Backward compatible alias
VQLPIPSWithDiscriminator = VQLPIPSWithDiscriminator3D