"""
Structure similarity loss module

Purpose: Ensure generated 3D rock data looks similar to real rock when visualized in Avizo

Included losses:
1. SSIM loss - local structural similarity
2. Multi-scale gradient loss - edge/interface clarity
3. Pore connectivity loss - pore network topology features
4. Frequency domain loss - global frequency features

Author: Auto-generated for rock generation project
Date: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
import numpy as np


class StructureSimilarityLoss(nn.Module):
    """
    Structure similarity loss(SSIM for 3D)
    
    Ensure local structural patterns of generated samples are similar to real samples
    """
    def __init__(self, window_size=11, size_average=True):
        super(StructureSimilarityLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        # Do not create window at init; create in forward based on actual channel count
        self.window = None
    
    def gaussian(self, window_size, sigma=1.5):
        """Create 1D Gaussian kernel"""
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window_3d(self, window_size, channel):
        """Create 3D Gaussian window"""
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _2D_window.unsqueeze(0) * _1D_window.unsqueeze(2)
        _3D_window = _3D_window.unsqueeze(0).unsqueeze(0)
        
        window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
        return window
    
    def ssim_3d(self, img1, img2):
        """
        Compute 3D SSIM
        
        Args:
            img1, img2: [B, C, D, H, W]
        
        Returns:
            ssim_value: scalar
        """
        (_, channel, _, _, _) = img1.size()
        
        # 🔥 Fix: dynamically create window based on actual channel count
        if self.window is None or self.window.size(0) != channel:
            self.window = self.create_window_3d(self.window_size, channel)
            self.window = self.window.type_as(img1)
        elif self.window.data.type() != img1.data.type():
            self.window = self.window.type_as(img1)
        
        # Compute mean
        mu1 = F.conv3d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv3d(img2, self.window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variance and covariance
        sigma1_sq = F.conv3d(img1*img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2*img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1*img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 🚨 Emergency fix: numerical stability check
        # Ensure denominator is not too small to prevent division by zero and NaN
        eps = 1e-8
        
        # SSIM formula(add numerical stability)
        numerator = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        # prevent division by zero
        denominator = torch.clamp(denominator, min=eps)
        
        ssim_map = numerator / denominator
        
        # 🚨 Clip SSIM value to reasonable range to prevent outliers
        ssim_map = torch.clamp(ssim_map, -1.0, 1.0)
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        """
        Compute SSIM loss
        
        Args:
            img1: generated sample [B, C, D, H, W]
            img2: real sample [B, C, D, H, W]
        
        Returns:
            loss: 1 - SSIM (lower is better)
        """
        ssim_value = self.ssim_3d(img1, img2)
        return 1 - ssim_value


class MultiScaleGradientLoss(nn.Module):
    """
    Multi-scale gradient loss
    
    generated sample/real sample
    Compute gradients at multiple scales to capture pore edges of different sizes
    """
    def __init__(self, scales=[1, 2, 4]):
        super(MultiScaleGradientLoss, self).__init__()
        self.scales = scales
    
    def gradient_3d(self, x):
        """
        Compute 3D gradient(Sobel operator)
        
        Args:
            x: [B, C, D, H, W]
        
        Returns:
            gradient_magnitude: [B, C, D, H, W]
        """
        # Sobel operator(simplified version)
        # Compute gradient in three directions
        grad_d = x[:, :, 2:, :, :] - x[:, :, :-2, :, :]
        grad_h = x[:, :, :, 2:, :] - x[:, :, :, :-2, :]
        grad_w = x[:, :, :, :, 2:] - x[:, :, :, :, :-2]
        
        # Padding to match size
        grad_d = F.pad(grad_d, (0, 0, 0, 0, 1, 1))
        grad_h = F.pad(grad_h, (0, 0, 1, 1, 0, 0))
        grad_w = F.pad(grad_w, (1, 1, 0, 0, 0, 0))
        
        # Gradient magnitude
        gradient_magnitude = torch.sqrt(grad_d**2 + grad_h**2 + grad_w**2 + 1e-8)
        
        return gradient_magnitude
    
    def forward(self, img1, img2):
        """
        Multi-scale gradient loss
        
        Args:
            img1: generated sample [B, C, D, H, W]
            img2: real sample [B, C, D, H, W]
        
        Returns:
            loss: gradient difference
        """
        loss = 0.0
        
        for scale in self.scales:
            if scale > 1:
                # downsample
                img1_scaled = F.avg_pool3d(img1, kernel_size=scale, stride=scale)
                img2_scaled = F.avg_pool3d(img2, kernel_size=scale, stride=scale)
            else:
                img1_scaled = img1
                img2_scaled = img2
            
            # Compute gradient
            grad1 = self.gradient_3d(img1_scaled)
            grad2 = self.gradient_3d(img2_scaled)
            
            # L1 loss
            loss += F.l1_loss(grad1, grad2)
        
        return loss / len(self.scales)


class PoreConnectivityLoss(nn.Module):
    """
    Pore connectivity loss
    
    generated sampleporereal sample
    
    Statistical features:
    - maximum connected pore volume fraction
    - Number of pore clusters
    - Average pore cluster size
    """
    def __init__(self):
        super(PoreConnectivityLoss, self).__init__()
    
    def compute_connectivity_stats(self, binary_volume):
        """
        Compute pore connectivity statistics(using differentiable approximation)
        
        Args:
            binary_volume: [B, C, D, H, W] binary volume(0=pore,1=solid)
        
        Returns:
            stats: Statistical features [B, num_features]
        """
        B, C, D, H, W = binary_volume.shape
        stats_list = []
        
        for b in range(B):
            # Extract single sample
            vol = binary_volume[b, 0].detach().cpu().numpy()
            
            # Binarize(<0.5pore)
            pore_mask = (vol < 0.5).astype(np.uint8)
            
            # Connected component analysis
            labeled, num_features = ndimage.label(pore_mask)
            
            if num_features == 0:
                # pore
                stats = torch.tensor([0.0, 0.0, 0.0], device=binary_volume.device)
            else:
                # Compute size of each connected component
                sizes = ndimage.sum(pore_mask, labeled, range(num_features + 1))
                
                # Feature 1: maximum connected pore volume fraction
                max_pore_ratio = sizes.max() / (D * H * W)
                
                # Feature 2: Number of pore clusters
                num_clusters_norm = num_features / 100.0  # normalize to reasonable range
                
                # Feature 3: Average pore cluster size()
                avg_cluster_size_norm = sizes.mean() / (D * H * W)
                
                stats = torch.tensor([
                    max_pore_ratio,
                    num_clusters_norm,
                    avg_cluster_size_norm
                ], device=binary_volume.device)
            
            stats_list.append(stats)
        
        return torch.stack(stats_list)  # [B, 3]
    
    def forward(self, img1, img2):
        """
        Pore connectivity loss
        
        Args:
            img1: generated sample [B, C, D, H, W]
            img2: real sample [B, C, D, H, W]
        
        Returns:
            loss: Statistical features
        """
        # Statistical features
        stats1 = self.compute_connectivity_stats(img1)
        stats2 = self.compute_connectivity_stats(img2)
        
        # MSE
        loss = F.mse_loss(stats1, stats2)
        
        return loss


class FrequencyDomainLoss(nn.Module):
    """
    Frequency domain loss
    
    generated samplereal sample
    FFT
    """
    def __init__(self):
        super(FrequencyDomainLoss, self).__init__()
    
    def forward(self, img1, img2):
        """
        Frequency domain loss
        
        Args:
            img1: generated sample [B, C, D, H, W]
            img2: real sample [B, C, D, H, W]
        
        Returns:
            loss: 
        """
        # 3D FFT
        fft1 = torch.fft.fftn(img1, dim=(-3, -2, -1))
        fft2 = torch.fft.fftn(img2, dim=(-3, -2, -1))
        
        # 
        mag1 = torch.abs(fft1)
        mag2 = torch.abs(fft2)
        
        # L1 loss
        loss = F.l1_loss(mag1, mag2)
        
        return loss


class CombinedStructureLoss(nn.Module):
    """
    Combined structure loss
    
    
    """
    def __init__(
        self,
        ssim_weight=0.5,
        gradient_weight=0.3,
        connectivity_weight=0.2,
        frequency_weight=0.1,
        use_ssim=True,
        use_gradient=True,
        use_connectivity=True,
        use_frequency=True
    ):
        super(CombinedStructureLoss, self).__init__()
        
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.connectivity_weight = connectivity_weight
        self.frequency_weight = frequency_weight
        
        # 
        if use_ssim:
            self.ssim_loss = StructureSimilarityLoss(window_size=7)
        else:
            self.ssim_loss = None
        
        if use_gradient:
            self.gradient_loss = MultiScaleGradientLoss(scales=[1, 2, 4])
        else:
            self.gradient_loss = None
        
        if use_connectivity:
            self.connectivity_loss = PoreConnectivityLoss()
        else:
            self.connectivity_loss = None
        
        if use_frequency:
            self.frequency_loss = FrequencyDomainLoss()
        else:
            self.frequency_loss = None
    
    def forward(self, generated, real, return_dict=False):
        """
        Combined structure loss
        
        Args:
            generated: generated sample [B, C, D, H, W]
            real: real sample [B, C, D, H, W]
            return_dict: Loss dictionary
        
        Returns:
            total_loss: Total loss
            loss_dict (optional): 
        """
        total_loss = 0.0
        loss_dict = {}
        
        # SSIM
        if self.ssim_loss is not None:
            ssim_loss_val = self.ssim_loss(generated, real)
            # 🚨 NaN
            if torch.isnan(ssim_loss_val) or torch.isinf(ssim_loss_val):
                print(f"⚠️  SSIM: {ssim_loss_val.item()}, ")
                ssim_loss_val = torch.tensor(0.0, device=generated.device)
            total_loss += self.ssim_weight * ssim_loss_val
            loss_dict['ssim_loss'] = ssim_loss_val.item()
        
        # 
        if self.gradient_loss is not None:
            gradient_loss_val = self.gradient_loss(generated, real)
            # 🚨 NaN
            if torch.isnan(gradient_loss_val) or torch.isinf(gradient_loss_val):
                print(f"⚠️  : {gradient_loss_val.item()}, ")
                gradient_loss_val = torch.tensor(0.0, device=generated.device)
            total_loss += self.gradient_weight * gradient_loss_val
            loss_dict['gradient_loss'] = gradient_loss_val.item()
        
        # Connectivity loss
        if self.connectivity_loss is not None:
            connectivity_loss_val = self.connectivity_loss(generated, real)
            # 🚨 NaN
            if torch.isnan(connectivity_loss_val) or torch.isinf(connectivity_loss_val):
                print(f"⚠️  Connectivity loss: {connectivity_loss_val.item()}, ")
                connectivity_loss_val = torch.tensor(0.0, device=generated.device)
            total_loss += self.connectivity_weight * connectivity_loss_val
            loss_dict['connectivity_loss'] = connectivity_loss_val.item()
        
        # Frequency domain loss
        if self.frequency_loss is not None:
            frequency_loss_val = self.frequency_loss(generated, real)
            # 🚨 NaN
            if torch.isnan(frequency_loss_val) or torch.isinf(frequency_loss_val):
                print(f"⚠️  Frequency domain loss: {frequency_loss_val.item()}, ")
                frequency_loss_val = torch.tensor(0.0, device=generated.device)
            total_loss += self.frequency_weight * frequency_loss_val
            loss_dict['frequency_loss'] = frequency_loss_val.item()
        
        loss_dict['total_structure_loss'] = total_loss.item()
        
        if return_dict:
            return total_loss, loss_dict
        else:
            return total_loss


# ====================================================================
# 
# ====================================================================

if __name__ == "__main__":
    # 
    print("Structure similarity loss module...")
    
    # 
    B, C, D, H, W = 2, 1, 64, 64, 64
    generated = torch.randn(B, C, D, H, W)
    real = torch.randn(B, C, D, H, W)
    
    # SSIM
    print("\n1. SSIM...")
    ssim_loss = StructureSimilarityLoss()
    loss = ssim_loss(generated, real)
    print(f"   SSIM Loss: {loss.item():.4f}")
    
    # 
    print("\n2. Multi-scale gradient loss...")
    gradient_loss = MultiScaleGradientLoss()
    loss = gradient_loss(generated, real)
    print(f"   Gradient Loss: {loss.item():.4f}")
    
    # Connectivity loss
    print("\n3. Pore connectivity loss...")
    connectivity_loss = PoreConnectivityLoss()
    binary_gen = (generated > 0).float()
    binary_real = (real > 0).float()
    loss = connectivity_loss(binary_gen, binary_real)
    print(f"   Connectivity Loss: {loss.item():.4f}")
    
    # Frequency domain loss
    print("\n4. Frequency domain loss...")
    frequency_loss = FrequencyDomainLoss()
    loss = frequency_loss(generated, real)
    print(f"   Frequency Loss: {loss.item():.4f}")
    
    # 
    print("\n5. Combined structure loss...")
    combined_loss = CombinedStructureLoss()
    loss, loss_dict = combined_loss(generated, real, return_dict=True)
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   Loss Dict: {loss_dict}")
    
    print("\n✅ !")

