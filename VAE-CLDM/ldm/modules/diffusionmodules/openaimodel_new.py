
from abc import abstractmethod
from functools import partial
import math
from typing import Iterable, Dict, Any, Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Import modified utility modules
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,  # supports 3D convolution
    linear,
    avg_pool_nd,  # supports 3D pooling
    zero_module,
    normalization,  # supports 3D normalization
    timestep_embedding,
)
from ldm.modules.attention_new import SpatialTransformer

# Placeholder replacement functions(maintain compatibility)
def convert_module_to_f16(x):
    """Convert to half precision(maintain compatibility)"""
    pass

def convert_module_to_f32(x):
    """Convert to single precision(maintain compatibility)"""
    pass

class AttentionPool3d(nn.Module):
    """
    3D attention pooling layer adapted for 3D rock data
    Extended from the 2D CLIP version
    """

    def __init__(
            self,
            spatial_dims: tuple,  # (D, H, W) tuple
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        d, h, w = spatial_dims
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, d * h * w + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, d, h, w = x.shape  # 3D data shape
        x = x.reshape(b, c, -1)  # NC(DHW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(DHW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(DHW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]

class TimestepBlock(nn.Module):
    """
    Abstract base class for timestep blocks
    Any module that takes timestep embedding as the second argument
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to input x with timestep embedding emb
        :param x: input feature tensor
        :param emb: timestep embedding
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Sequential module - fixed error handling version
    """

    def forward(self, x, emb, context=None, text_context=None, porosity_context=None):
        """
        Forward pass supporting multi-condition input and error handling
        """
        for layer in self:
            try:
                if isinstance(layer, TimestepBlock):
                    # For residual blocks, pass porosity condition
                    if isinstance(layer, ResBlock) and hasattr(layer, 'condition_dim') and layer.condition_dim:
                        x = layer(x, emb, condition=porosity_context)
                    else:
                        x = layer(x, emb)
                elif isinstance(layer, SpatialTransformer):
                    # Spatial Transformer handles text condition
                    x = layer(x, context=context)
                elif isinstance(layer, AttentionBlock):
                    # Traditional attention block
                    x = layer(x)
                else:
                    x = layer(x)
            except Exception as e:
                print(f"❌ Layer {type(layer).__name__} failed: {e}")
                # Skip failed layer and continue execution
                continue
                
        return x


class Upsample(nn.Module):
    """
    Upsampling layer supporting 3D data
    :param channels: input/output channel count
    :param use_conv: whether to apply convolution
    :param dims: number of dimensions (2 or 3)
    """

    def __init__(self, channels, use_conv, dims=3, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            # 3D interpolation: upsample depth, height, and width by 2x
            x = F.interpolate(
                x, scale_factor=(2, 2, 2), mode="trilinear", align_corners=False
            )
        else:
            # 2D interpolation
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    """
    Learned 2x upsampling(without padding)
    adapted for 3D transposed convolution
    """

    def __init__(self, channels, out_channels=None, ks=5, dims=3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims

        if dims == 3:
            self.up = nn.ConvTranspose3d(
                self.channels, self.out_channels,
                kernel_size=ks, stride=2, padding=2, output_padding=1
            )
        else:
            self.up = nn.ConvTranspose2d(
                self.channels, self.out_channels,
                kernel_size=ks, stride=2, padding=2, output_padding=1
            )

    def forward(self, x):
        return self.up(x)

class Downsample(nn.Module):
    """
    Downsampling layer supporting 3D data
    :param channels: input/output channel count
    :param use_conv: whether to apply convolution
    :param dims: number of dimensions (2 or 3)
    """

    def __init__(self, channels, use_conv, dims=3, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        if dims == 3:
            # 🔥 Fix:3D downsampling should halve depth, height, and width
            stride = (2, 2, 2)  # downsample all dimensions by 2x
        else:
            stride = 2

        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    """
    Residual block supporting 3D data and condition injection
    Extended to support porosity condition handling
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=3,  # default 3D
            use_checkpoint=False,
            up=False,
            down=False,
            # New: condition dimension(for porosity condition)
            condition_dim=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dims = dims
        self.condition_dim = condition_dim

        # inputLayer
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # Embedding layer - extended to support condition information
        total_emb_dim = emb_channels
        if condition_dim is not None:
            total_emb_dim += condition_dim

        # print(f"🔧 ResBlock embedding config: input dimension={total_emb_dim}, output dimension={self.out_channels}")
        
        # Key fix: ensure embedding layer output dimension is correct
        if use_scale_shift_norm:
            emb_out_dim = 2 * self.out_channels
        else:
            emb_out_dim = self.out_channels

            
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                total_emb_dim,
                emb_out_dim,  # use the correct output dimension
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
            
    def forward(self, x, emb, condition=None):
        """
        Apply block to tensor conditioned on timestep embedding and condition information
        """
        return checkpoint(
            self._forward, (x, emb, condition), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, condition=None):
        # Handle up/down sampling
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        # Fix: handle condition and timestep embedding
        if condition is not None and self.condition_dim:
            # Ensure condition dimension is correct
            if condition.shape[1] != self.condition_dim:
                # print(f"🔧 Adjust condition dimension: {condition.shape[1]} -> {self.condition_dim}")
                if condition.shape[1] > self.condition_dim:
                    condition = condition[:, :self.condition_dim]
                else:
                    # repeat expansion
                    repeat_factor = (self.condition_dim + condition.shape[1] - 1) // condition.shape[1]
                    condition = condition.repeat(1, repeat_factor)[:, :self.condition_dim]
            
            # Enhance condition influence:scale condition information before concatenation
            condition_enhanced = condition * 2.0  # scale condition influence
            combined_emb = th.cat([emb, condition_enhanced], dim=1)
        else:
            combined_emb = emb

        # Key fix: ensure embedding input dimension is correct
        if combined_emb.shape[1] != self.emb_layers[1].in_features:
            # print(f"Adjust embedding input dimension: {combined_emb.shape[1]} -> {self.emb_layers[1].in_features}")
            target_dim = self.emb_layers[1].in_features
            if combined_emb.shape[1] < target_dim:
                # repeat expansion
                repeat_factor = (target_dim + combined_emb.shape[1] - 1) // combined_emb.shape[1]
                combined_emb = combined_emb.repeat(1, repeat_factor)[:, :target_dim]
            else:
                # truncate
                combined_emb = combined_emb[:, :target_dim]

        emb_out = self.emb_layers(combined_emb).type(h.dtype)
        # print(f"🔍 ResBlock embedding config: input{combined_emb.shape} -> {emb_out.shape}, hshape{h.shape}")

        # Expand dimensions to match spatial dimensions
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    Attention block - fixed 3D support version
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
            dims=3,  # add dims parameter
    ):
        super().__init__()
        self.channels = channels
        self.dims = dims

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,vchannel count {channels} cannot be divided by head channel count {num_head_channels} evenly"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        
        # Select correct convolution class based on dimensions
        if dims == 3:
            conv_class = nn.Conv3d
        else:
            conv_class = nn.Conv2d
            
        self.qkv = conv_class(channels, channels * 3, 1)

        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_class(channels, channels, 1))

    def forward(self, x):
        # If multi-head attention is not enabled or input is not 5D, safely short-circuit and return
        if getattr(self, 'num_heads', 0) <= 0 or x.dim() != 5:
            return x
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        # print(f"🔍 AttentionBlockinput: shape{x.shape}, dimension{self.dims}")
        
        # Apply 1x1x1 projection on 5D directly, flatten space as sequence for attention, then restore
        h_norm = self.norm(x)
        qkv = self.qkv(h_norm)
        q, k, v = th.chunk(qkv, 3, dim=1)
        seq_len = spatial[0] * spatial[1] * spatial[2] if self.dims == 3 else spatial[0] * spatial[1]
        q = q.view(b, self.num_heads, c // self.num_heads, seq_len)
        k = k.view(b, self.num_heads, c // self.num_heads, seq_len)
        v = v.view(b, self.num_heads, c // self.num_heads, seq_len)
        scale = 1.0 / math.sqrt(math.sqrt(c // self.num_heads))
        attn = th.einsum("bhct,bhcs->bhts", q * scale, k * scale)
        attn = th.softmax(attn.float(), dim=-1).type(attn.dtype)
        out = th.einsum("bhts,bhcs->bhct", attn, v)
        out = out.reshape(b, c, *spatial)
        out = self.proj_out(out)
        return x + out
        

class QKVAttentionLegacy(nn.Module):
    """
    Traditional QKV attention implementation
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

class QKVAttention(nn.Module):
    """
    Improved QKV attention implementation
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

class UNetModel(nn.Module):
    def __init__(self,
                 image_size=64,        
                 in_channels=4,        
                 model_channels=128,   
                 out_channels=4,       
                 num_res_blocks=2,
                 attention_resolutions=[16, 8],  # matches config file [16, 8]
                 dropout=0.1,                    # matches config file 0.1
                 channel_mult=[1, 2, 4], 
                 conv_resample=True,
                 dims=3,              
                 num_classes=None,
                 use_checkpoint=False,
                 num_heads=8,                    # matches config file 4
                 num_heads_upsample=-1,
                 use_scale_shift_norm=True,      # matches config file True
                 resblock_updown=True,           # matches config file True
                 use_new_attention_order=False,
                 use_spatial_transformer=True,
                 transformer_depth=1,            # matches config file 1
                 context_dim=768,                # key fix: matches config file 768
                 use_porosity_condition=True,
                 porosity_dim=1,
                 legacy=False,
                 num_head_channels=32,           # matches config file 64
                 **kwargs):
        
        super().__init__()

        self.dims = dims
        
        # Parameter validation and logging
        print("🔧 Initialize 3D UNet model ")
        print(f"  📐 Input blocks: {in_channels}")
        print(f"  📐 Output channels: {out_channels}")
        print(f"  🎯 Porosity condition: {'enabled' if use_porosity_condition else 'disabled'}")
        print(f"  📊 Porosity dimension: {porosity_dim}")
        print(f"  🔷 3Ddimension: {dims}")
        print(f"  📈 Channel multiplier: {channel_mult}")
        
        # Key fix: ensure input/output channels match
        if in_channels != 4:
            print(f"⚠️ Warning: UNet input channels should be 4 to match VAE, current value is {in_channels}")
        
        if out_channels != 4:
            print(f"⚠️ Warning: UNetcannot be divided by head channel count4VAE,{out_channels}")
        
        # Model config
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        self.use_spatial_transformer = use_spatial_transformer
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim
        self.porosity_dim = porosity_dim
        self.use_porosity_condition = use_porosity_condition
        self.dtype = th.float32
        self.predict_codebook_ids = False
        
        # timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        # Porosity conditionprojection layer - key fix:Porosity dimension
        if self.use_porosity_condition and self.porosity_dim > 0:
            # key fix:Porosity dimension,Enhance condition influence
            self.porosity_proj = nn.Sequential(
                nn.Linear(porosity_dim, 128),  # Porosity dimension
                nn.SiLU(),
                nn.Dropout(0.1),  # add dropout to prevent overfitting
                nn.Linear(128, time_embed_dim)  # Project to timestep embedding dimension
            )
            print(f"  🔧 projection layer: {porosity_dim} -> 128 -> {time_embed_dim}")
        else:
            self.porosity_proj = None

        self.resblock_condition_dim = time_embed_dim  # Project to timestep embedding dimensionPorosity dimension


        
        # Input blocks - 3D
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1  # downsampling factor

        # Compute spatial size after downsampling
        # 64 -> 32 -> 16 -> 8 (based on channel_mult length)
        levels = len(channel_mult)
        print(f"  📊 UNet downsampling levels: {levels}")
        
        # Build downsampling path
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        condition_dim=self.resblock_condition_dim,  # Porosity dimension
                    )
                ]
                ch = mult * model_channels

                # projection layer
                current_resolution = 64 // (2 ** level)
                if current_resolution in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                
                    # Fix:pass dims parameter to SpatialTransformer
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                            dims=dims,  # Porosity dimension
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth,
                            context_dim=context_dim,
                            dims=dims  # key fix:Porosity dimension
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            # Downsampling
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            condition_dim=time_embed_dim,  # key fix
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle block
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                condition_dim=self.resblock_condition_dim,  # Porosity dimension

            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth,
                context_dim=context_dim
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                condition_dim=time_embed_dim,  # key fix
            ),
        )
        self._feature_size += ch

        # Output blocks(upsampling path)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        condition_dim=self.resblock_condition_dim,  # Porosity dimension

                    )
                ]
                ch = model_channels * mult

                # projection layer
                current_resolution = 64 // (2 ** (len(channel_mult) - 1 - level))
                if current_resolution in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth,
                            context_dim=context_dim
                        )
                    )

                # Upsampling
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            condition_dim=time_embed_dim,  # key fix
                        ) if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # projection layer
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        print(f"✅ 3D UNetmodel initialization complete")
        print(f"  📊 Total parameter count: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  🎯 inputshape: (B, {in_channels}, 16, 16, 16)")  # correction:64downsampled to 16
        print(f"  🎯 Output shape: (B, {out_channels}, 16, 16, 16)")

    def forward(self, x, timesteps, context=None, porosity_condition=None, **kwargs):
        """
        Forward pass - FixPorosity dimension
        """
        # timestep embedding
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        
        # Handle porosity condition
        porosity_emb = None
        if self.use_porosity_condition and porosity_condition is not None:
            if self.porosity_proj is not None:
                porosity_emb = self.porosity_proj(porosity_condition)
        
        # Forward pass
        hs = []
        h = x.type(self.dtype)
    
        # Downsampling path
        for i, module in enumerate(self.input_blocks):
            if isinstance(module, TimestepEmbedSequential):
                h = module(h, emb, context=context, porosity_context=porosity_emb)
            else:
                h = module(h, emb)
            hs.append(h)
    
        # Middle block
        h = self.middle_block(h, emb, context=context, porosity_context=porosity_emb)
    
        # upsampling path - fix skip connections
        for i, module in enumerate(self.output_blocks):
            # Get corresponding skip connection
            skip_connection = hs.pop()
            
            # key fix:use instance attribute self.dims
            if self.dims == 3:
                # Porosity dimension
                if h.shape[2:] != skip_connection.shape[2:]:
                    # 3DPorosity dimension
                    h = F.interpolate(
                        h, size=skip_connection.shape[2:], mode='trilinear', align_corners=False
                    )
            else:
                # 2D interpolation
                if h.shape[2:] != skip_connection.shape[2:]:
                    h = F.interpolate(
                        h, size=skip_connection.shape[2:], mode='bilinear', align_corners=False
                    )
            
            # Concatenate
            h = th.cat([h, skip_connection], dim=1)
            
            if isinstance(module, TimestepEmbedSequential):
                h = module(h, emb, context=context, porosity_context=porosity_emb)
            else:
                h = module(h, emb)
    
        h = h.type(x.dtype)
        
        # Final output
        return self.out(h)
        
    def convert_to_fp16(self):
        """Convert model body to float16"""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert model body to float32"""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

# Encoder UNet(half UNet)keeps a similar structure, but is optimized for encoding tasks
class EncoderUNetModel(nn.Module):
    """
    Half UNet model for encoding tasks
    adapted for 3D rock data
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=3,  # default 3D
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            pool="adaptive",
            *args,
            **kwargs
    ):
        super().__init__()
        
        print("🔧 Initialize 3D encoder UNet model")
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Input blocks
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ])

        # Build encoding path(similar to UNet but simpler)
        ch = model_channels
        input_block_chans = [model_channels]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        condition_dim=self.resblock_condition_dim,  # Porosity dimension

                    )
                ]
                ch = mult * model_channels
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                condition_dim=self.resblock_condition_dim,  # Porosity dimension

            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # projection layer
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),  # 3D adaptive pooling
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(dims, ch, out_channels, 3, padding=1),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"Unsupported pooling type: {pool}")

    def forward(self, x, timesteps):
        """Encoder forward pass"""
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Encoding path
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)

        # Middle block
        h = self.middle_block(h, emb)

        # Output processing
        return self.out(h)


        