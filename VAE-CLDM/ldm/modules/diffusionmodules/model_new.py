"""
Core functions

1. UNet architecture: implements the core network structure of the diffusion process
2,encoder/decoder: handles conversion to latent space and reconstruction
3,timestep processing: manages timestep embeddings in the diffusion process
4,attention mechanism: provides spatial and conditional attention functionality
"""

"""
Core modules of the diffusion model, including UNet architecture, encoder, decoder, and other key components
This is the core of Stable Diffusion and performs the diffusion process in latent space


Main purposes:
1. Implement the UNet architecture for the diffusion process
2. Provide encoder/decoder for latent space conversion
3. Support condition injection and multimodal input
4. Manage timestep embeddings and attention mechanisms
"""

import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from ldm.util_new import instantiate_from_config
from ldm.modules.attention_new import LinearAttention, SpatialTransformer


class Rock3DConfigUtils:
    """3Drock data configuration utilities"""
    
    @staticmethod
    def get_safe_3d_config(resolution=64, base_channels=32):
        """3D rock data configuration utilities"""
        
        # Compute maximum safe number of downsampling operations
        max_downsample = max(1, int(np.log2(resolution)) - 2)  # ensure minimum size>=4
        
        # Adjust channel multipliers
        if resolution == 64:
            ch_mult = (1, 2, 4, 4)  # 3downsampling operations: 64->32->16->8
            attn_resolutions = [32, 16, 8]  # use attention at larger resolutions
        elif resolution == 32:
            ch_mult = (1, 2, 4)  # 2downsampling operations: 32->16->8
            attn_resolutions = [16, 8]
        else:
            ch_mult = (1, 2)  # 1downsampling operations
            attn_resolutions = [resolution // 2]
        
        return {
            "ch_mult": ch_mult,
            "attn_resolutions": attn_resolutions,
            "num_res_blocks": 2,
            "dropout": 0.1
        }
    
    @staticmethod
    def create_robust_3d_unet_config(resolution=64, in_channels=1, out_channels=1):
        """Create robust 3D UNet config"""
        safe_config = Rock3DConfigUtils.get_safe_3d_config(resolution)
        
        return {
            "in_channels": in_channels,
            "out_ch": out_channels,
            "ch": 32,
            "ch_mult": safe_config["ch_mult"],
            "num_res_blocks": safe_config["num_res_blocks"],
            "attn_resolutions": safe_config["attn_resolutions"],
            "dropout": safe_config["dropout"],
            "resamp_with_conv": True,
            "resolution": resolution,
            "use_timestep": True,
            "attn_type": "vanilla",
            "spatial_dims": 3
        }


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Generate sinusoidal positional embeddings for timesteps
    Matches DDPM and Fairseq implementations

    Args:
        timesteps: timestep tensor
        embedding_dim: embedding dimension

    Returns:
        torch.Tensor: timestep embedding
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero padding
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    """Swish activation function"""
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32, spatial_dims=3):
    """
    Normalization layer supporting 2D and 3D data

    Args:
        in_channels: number of input channels
        num_groups: number of groups
        spatial_dims: spatial dimensions(2or3)
    """
    if spatial_dims == 3:
        # 3Dgroup normalization
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        # 2Dgroup normalization
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    """Upsampling module supporting 2D and 3D"""

    def __init__(self, in_channels, with_conv, spatial_dims=3):
        super().__init__()
        self.with_conv = with_conv
        self.spatial_dims = spatial_dims
        if self.with_conv:
            # Select 2D or 3D convolution
            conv_class = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
            self.conv = conv_class(in_channels, in_channels,
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if self.spatial_dims == 3:
            # 3Dinterpolation
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="trilinear", align_corners=False)
        else:
            # 2Dinterpolation
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling module supporting 2D and 3D"""

    def __init__(self, in_channels, with_conv, spatial_dims=3):
        super().__init__()
        self.with_conv = with_conv
        self.spatial_dims = spatial_dims
        if self.with_conv:
            conv_class = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
            # Use stride-2 convolution for downsampling to avoid padding issues
            self.conv = conv_class(in_channels, in_channels, 
                                   kernel_size=3, stride=2, padding=1)  # change padding to 1

    def forward(self, x):
        if self.with_conv:
            # Use stride-2 convolution directly, no manual padding needed
            x = self.conv(x)
        else:
            if self.spatial_dims == 3:
                x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
            else:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """Residual block supporting 2D and 3D"""

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, spatial_dims=3):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # Select 2D or 3D convolution
        conv_class = nn.Conv3d if spatial_dims == 3 else nn.Conv2d

        self.norm1 = Normalize(in_channels, spatial_dims=spatial_dims)
        self.conv1 = conv_class(in_channels, out_channels,
                                kernel_size=3, stride=1, padding=1)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels, spatial_dims=spatial_dims)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv_class(out_channels, out_channels,
                                kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_class(in_channels, out_channels,
                                                kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = conv_class(in_channels, out_channels,
                                               kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            # Timestep embedding projection
            if self.spatial_dims == 3:
                h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
            else:
                h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LinAttnBlock(LinearAttention):
    """Linear attention block"""

    def __init__(self, in_channels, spatial_dims=3):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels, spatial_dims=spatial_dims)


class AttnBlock(nn.Module):
    """Residual block supporting 2D and 3D"""

    def __init__(self, in_channels, spatial_dims=3):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims

        self.norm = Normalize(in_channels, spatial_dims=spatial_dims)

        # Select 2D or 3D convolution
        conv_class = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.q = conv_class(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = conv_class(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = conv_class(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = conv_class(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        if self.spatial_dims == 3:
            # 3Ddata: (B, C, D, H, W)
            b, c, d, h, w = q.shape
            q = q.reshape(b, c, d * h * w)
            q = q.permute(0, 2, 1)  # b, d*h*w, c
            k = k.reshape(b, c, d * h * w)  # b, c, d*h*w
        else:
            # 2Ddata: (B, C, H, W)
            b, c, h, w = q.shape
            q = q.reshape(b, c, h * w)
            q = q.permute(0, 2, 1)  # b, h*w, c
            k = k.reshape(b, c, h * w)  # b, c, h*w

        # Compute attention weights
        w_ = torch.bmm(q, k)  # b, (spatial), (spatial)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # Apply attention to values
        if self.spatial_dims == 3:
            v = v.reshape(b, c, d * h * w)
        else:
            v = v.reshape(b, c, h * w)

        w_ = w_.permute(0, 2, 1)  # b, (spatial), (spatial)
        h_ = torch.bmm(v, w_)  # b, c, (spatial)

        if self.spatial_dims == 3:
            h_ = h_.reshape(b, c, d, h, w)
        else:
            h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_


def make_attn(in_channels, attn_type="vanilla", spatial_dims=3):
    """Create attention module"""
    assert attn_type in ["vanilla", "linear", "none", "spatial"], f'attn_type {attn_type} unknown'

    if attn_type == "vanilla":
        return AttnBlock(in_channels, spatial_dims=spatial_dims)
    elif attn_type == "linear":
        return LinAttnBlock(in_channels, spatial_dims=spatial_dims)
    elif attn_type == "spatial":
        # Spatial Transformer attention
        return SpatialTransformer(in_channels, n_heads=8, d_head=64,
                                  depth=1, dropout=0.1, spatial_dims=spatial_dims)
    else:  # "none"
        return nn.Identity()


class UNet3D(nn.Module):
    """
    3D UNet diffusion model supporting multi-condition input

    Modification notes:
    - Full support for 3D data
    - Extended multi-condition handling(text + porosity)
    - Optimize memory usage
    """

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False,
                 attn_type="vanilla", spatial_dims=3, condition_dim=None):
        super().__init__()

        if use_linear_attn:
            attn_type = "linear"

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.condition_dim = condition_dim

        # Fix: if condition dimension is provided, adjust input channel count
        if condition_dim is not None:
            self.in_channels += condition_dim

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                nn.Linear(self.ch, self.temb_ch),
                nn.Linear(self.temb_ch, self.temb_ch),
            ])

        # Input convolution
        conv_class = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.conv_in = conv_class(self.in_channels, self.ch,
                                  kernel_size=3, stride=1, padding=1)

        # Downsampling
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch * in_ch_mult[0]

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=self.temb_ch, dropout=dropout,
                    spatial_dims=spatial_dims
                ))
                block_in = block_out

                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, spatial_dims=spatial_dims))

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv, spatial_dims=spatial_dims)
                curr_res = curr_res // 2

            self.down.append(down)

        # Middle layer
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=self.temb_ch, dropout=dropout,
            spatial_dims=spatial_dims
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, spatial_dims=spatial_dims)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=self.temb_ch, dropout=dropout,
            spatial_dims=spatial_dims
        )

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]

                block.append(ResnetBlock(
                    in_channels=block_in + skip_in, out_channels=block_out,
                    temb_channels=self.temb_ch, dropout=dropout,
                    spatial_dims=spatial_dims
                ))
                block_in = block_out

                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, spatial_dims=spatial_dims))

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, spatial_dims=spatial_dims)
                curr_res = curr_res * 2

            self.up.insert(0, up)

        # Output layer
        self.norm_out = Normalize(block_in, spatial_dims=spatial_dims)
        self.conv_out = conv_class(block_in, out_ch,
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None, context=None, text_context=None, porosity_context=None):
        """
        Forward pass,support multi-condition input

        Args:
            x: input data
            t: timestep
            context: general condition context
            text_context: text condition
            porosity_context: porosity condition
        """
        # Fix:handle multi-condition input
        if context is not None:
            # Ensure condition is compatible with input data shape
            if context.dim() == 2 and x.dim() == 5:  # condition is 2D, data is 5D
                # Broadcast condition to spatial dimensions
                b, c = context.shape
                _, _, d, h, w = x.shape
                context = context.view(b, c, 1, 1, 1).expand(b, c, d, h, w)
            # concatenate condition along channel axis
            x = torch.cat((x, context), dim=1)

        if self.use_timestep:
            assert t is not None
            # timestep embedding
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle layer
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    """3Dencoder"""

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False,
                 attn_type="vanilla", spatial_dims=3, **ignore_kwargs):
        super().__init__()

        # Fix: use full ch_mult without truncation
        actual_ch_mult = ch_mult

        if use_linear_attn:
            attn_type = "linear"

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(actual_ch_mult)  # Use full number of downsampling operations
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims

        # Input convolution
        conv_class = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.conv_in = conv_class(in_channels, self.ch,
                                  kernel_size=3, stride=1, padding=1)

        # Downsampling
        curr_res = resolution
        in_ch_mult = (1,) + tuple(actual_ch_mult)
        self.down = nn.ModuleList()
        block_in = ch * in_ch_mult[0]

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * actual_ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=self.temb_ch, dropout=dropout,
                    spatial_dims=spatial_dims
                ))
                block_in = block_out

                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, spatial_dims=spatial_dims))

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                # Use full number of downsampling operations
                if curr_res // 2 >= 4:  # Use full number of downsampling operations4
                    down.downsample = Downsample(block_in, resamp_with_conv, spatial_dims=spatial_dims)
                    curr_res = curr_res // 2
                else:
                    # If size is too small, skip downsampling
                    down.downsample = nn.Identity()
                    print(f"Warning: skipping downsampling at level {i_level}, current resolution {curr_res} is too small")

            self.down.append(down)

        # Middle layer
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=self.temb_ch, dropout=dropout,
            spatial_dims=spatial_dims
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, spatial_dims=spatial_dims)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=self.temb_ch, dropout=dropout,
            spatial_dims=spatial_dims
        )

        # Output
        self.norm_out = Normalize(block_in, spatial_dims=spatial_dims)
        self.conv_out = conv_class(block_in, 2 * z_channels if double_z else z_channels,
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None  # Encoder does not use timestep

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle layer
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """3Ddecoder"""

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False,
                 use_linear_attn=False, attn_type="vanilla", spatial_dims=3, **ignore_kwargs):
        super().__init__()

        if use_linear_attn:
            attn_type = "linear"

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.spatial_dims = spatial_dims

        # Compute parameters at the lowest resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        if spatial_dims == 3:
            self.z_shape = (1, z_channels, curr_res, curr_res, curr_res)
        else:
            self.z_shape = (1, z_channels, curr_res, curr_res)

        print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        # Input convolution
        conv_class = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.conv_in = conv_class(z_channels, block_in,
                                  kernel_size=3, stride=1, padding=1)

        # Middle layer
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=self.temb_ch, dropout=dropout,
            spatial_dims=spatial_dims
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, spatial_dims=spatial_dims)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=self.temb_ch, dropout=dropout,
            spatial_dims=spatial_dims
        )

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=self.temb_ch, dropout=dropout,
                    spatial_dims=spatial_dims
                ))
                block_in = block_out

                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, spatial_dims=spatial_dims))

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, spatial_dims=spatial_dims)
                curr_res = curr_res * 2

            self.up.insert(0, up)

        # Output
        self.norm_out = Normalize(block_in, spatial_dims=spatial_dims)
        self.conv_out = conv_class(block_in, out_ch,
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape

        temb = None
        h = self.conv_in(z)

        # Middle layer
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if self.tanh_out:
            h = torch.tanh(h)

        return h


# Model utility class dedicated to 3D rock data
class Rock3DModelUtils:
    """3D rock data model utility class"""

    @staticmethod
    def create_3d_unet_config(in_channels=1, out_channels=1, base_channels=32,
                              resolution=64, use_attention=True):
        """
        Create 3D UNet config

        Returns:
            dict: UNetconfig dictionary
        """
        attn_resolutions = [resolution // 8, resolution // 16] if use_attention else []

        return {
            "in_channels": in_channels,
            "out_ch": out_channels,
            "ch": base_channels,
            "ch_mult": (1, 2, 4, 4),
            "num_res_blocks": 2,
            "attn_resolutions": attn_resolutions,
            "dropout": 0.1,
            "resamp_with_conv": True,
            "resolution": resolution,
            "use_timestep": True,
            "attn_type": "vanilla",
            "spatial_dims": 3
        }

    @staticmethod
    def create_3d_encoder_config(in_channels=1, z_channels=4, base_channels=32,
                                 resolution=64, use_attention=True):
        """
        Create 3D encoder config

        Returns:
            dict: encoderconfig dictionary
        """
        attn_resolutions = [resolution // 8, resolution // 16] if use_attention else []

        return {
            "in_channels": in_channels,
            "out_ch": None,
            "ch": base_channels,
            "ch_mult": (1, 2, 4, 4),
            "num_res_blocks": 2,
            "attn_resolutions": attn_resolutions,
            "dropout": 0.0,
            "resamp_with_conv": True,
            "resolution": resolution,
            "z_channels": z_channels,
            "double_z": True,
            "attn_type": "vanilla",
            "spatial_dims": 3
        }

    @staticmethod
    def create_3d_decoder_config(out_channels=1, z_channels=4, base_channels=32,
                                 resolution=64, use_attention=True):
        """
        Create 3D decoder config

        Returns:
            dict: decoderconfig dictionary
        """
        attn_resolutions = [resolution // 8, resolution // 16] if use_attention else []

        return {
            "out_ch": out_channels,
            "ch": base_channels,
            "ch_mult": (1, 2, 4, 4),
            "num_res_blocks": 2,
            "attn_resolutions": attn_resolutions,
            "dropout": 0.0,
            "resamp_with_conv": True,
            "resolution": resolution,
            "z_channels": z_channels,
            "give_pre_end": False,
            "tanh_out": False,
            "attn_type": "vanilla",
            "spatial_dims": 3
        }

    @staticmethod
    def init_3d_model_from_config(config, model_class=UNet3D):
        """
        Initialize 3D model from config

        Args:
            config: model config
            model_class: model class

        Returns:
            nn.Module: initialized model
        """
        return model_class(**config)