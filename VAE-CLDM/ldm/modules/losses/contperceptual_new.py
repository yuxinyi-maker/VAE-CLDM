
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from taming.modules.losses.lpips import LPIPS
    from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss, weights_init, adopt_weight
except ImportError:
    # Provide fallback implementation to avoid taming dependency
    print("Warning: taming module not found, using fallback implementation")
    
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
    
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def adopt_weight(weight, global_step, threshold=0, value=0.0):
        if global_step < threshold:
            weight = value
        return weight
    
    # Simplified LPIPS implementation
    class LPIPS(nn.Module):
        def __init__(self):
            super().__init__()
            # Use simple L1 loss as substitute
            self.loss_fn = nn.L1Loss()
        
        def forward(self, x, y):
            return self.loss_fn(x, y)


class NLayer3DDiscriminator(nn.Module):
  
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


class LPIPSWithDiscriminator3D(nn.Module):
    
    
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0e-6, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=1, disc_factor=1.0, disc_weight=0.5,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", is_3d=True, unconditional_mode=False):
        """
        Initialize 3D perceptual loss
        
        Parameters:
        - disc_start: step at which discriminator starts training
        - kl_weight: KL divergence weight (1.0e-6 for 3D rock data)
        - disc_in_channels: input channel count (1=single-channel rock data)
        - is_3d: whether data is 3D
        - unconditional_mode: whether unconditional mode
        """
        super().__init__()
        
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.is_3d = is_3d
        self.unconditional_mode = unconditional_mode
        
        # Output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        
        # Select 2D or 3D discriminator
        if is_3d:
            self.perceptual_loss = self._dummy_perceptual_loss  # 3D temporarily uses simple loss
            self.discriminator = NLayer3DDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm
            ).apply(weights_init)
        else:
            # Keep original 2D implementation(backward compatible)
            from taming.modules.losses.vqperceptual import NLayerDiscriminator
            self.perceptual_loss = LPIPS().eval()
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm
            ).apply(weights_init)
        
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional and not unconditional_mode  # unconditional mode disables conditional discriminator
        
        # Last layer
        self.last_layer = None
        
        # Unconditional mode logging
        if unconditional_mode:
            print("🔧 Unconditional mode: disable conditional discriminator, simplify loss computation")
    
    def _dummy_perceptual_loss(self, inputs, reconstructions):
        
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

    def forward(self, inputs, reconstructions, posterior=None, optimizer_idx=0,
        global_step=0, last_layer=None, cond=None, split="train",
        weights=None, **kwargs):

    
        # Unconditional mode: simplify condition handling
        if self.unconditional_mode:
            # :Parameters
            cond = None
            self.disc_conditional = False
        
        # Add numerical stability check
        if torch.isnan(inputs).any() or torch.isnan(reconstructions).any():
            print("Warning: input or reconstruction data contains NaN values")
            # Return a safe loss value
            if optimizer_idx == 0:
                return torch.tensor(0.0, device=inputs.device), {}
            else:
                return torch.tensor(0.0, device=inputs.device), {}
        
        # Reconstruction loss(L1 loss)- add stability handling
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        
        # Reconstruction lossNaN
        if torch.isnan(rec_loss).any():
            print(": Reconstruction lossNaN,using zero loss")
            rec_loss = torch.zeros_like(rec_loss)
        
        # Perceptual loss(use simplified version for 3D)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            # Perceptual loss
            if torch.isnan(p_loss).any():
                print(": Perceptual lossNaN,")
                p_loss = torch.tensor(0.0, device=inputs.device)
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        
        # Weighted negative log-likelihood loss - add stability handling
        logvar_clamped = torch.clamp(self.logvar, min=-10.0, max=10.0)  # constrain logvar range
        nll_loss = rec_loss / torch.exp(logvar_clamped) + logvar_clamped
        
        # Check nll_loss
        if torch.isnan(nll_loss).any():
            print(": NLLNaN,Reconstruction loss")
            nll_loss = rec_loss
        
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        
        # KL - add stability handling
        if posterior is not None and hasattr(posterior, 'kl'):
            try:
                kl_loss = posterior.kl()
                # Check if KL loss is NaN or Inf
                if torch.isnan(kl_loss).any() or torch.isinf(kl_loss).any():
                    print("Warning: KL loss contains NaN or Inf, using zero loss")
                    kl_loss = torch.tensor(0.0, device=inputs.device)
                else:
                    # Safely compute KL loss mean
                    if kl_loss.dim() > 0:
                        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                    else:
                        kl_loss = kl_loss
            except Exception as e:
                print(f"KL loss computation error: {e},using zero loss")
                kl_loss = torch.tensor(0.0, device=inputs.device)
        else:
            kl_loss = torch.tensor(0.0, device=inputs.device)
        
        # Last layer
        self.last_layer = last_layer

        # GAN section
        if optimizer_idx == 0:
            # Generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    # Adaptive weight computation failed,using fixed weight
                    d_weight = torch.tensor(self.discriminator_weight, device=nll_loss.device)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            # Total loss = Reconstruction loss + KL loss + generator
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            # Log metrics
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),  # key: add rec_loss
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            
            if self.perceptual_weight > 0:
                log["{}/perceptual_loss".format(split)] = p_loss.detach().mean()
            
            return loss, log


        elif optimizer_idx == 1:
            # Discriminator update
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

        else:
            # optimizer index
            raise ValueError(f"optimizer index: {optimizer_idx}")


# backward compatible
LPIPSWithDiscriminator = LPIPSWithDiscriminator3D