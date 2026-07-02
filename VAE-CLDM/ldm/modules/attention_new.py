
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    """Check whether value exists(not None)"""
    return val is not None


def uniq(arr):
    """Get unique value list"""
    return {el: True for el in arr}.keys()


def default(val, d):
    """Provide default value"""
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    """Get maximum negative value of data type"""
    return -torch.finfo(t.dtype).max


def init_(tensor):
    """Initialize tensor"""
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# Feed-forward network
class GEGLU(nn.Module):
    """GLUFeed-forward network"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Parameters
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels, num_groups=32):
    """
    Normalization layer supporting 2D and 3D data

    Args:
        in_channels: input channel count
        num_groups: number of groups
    """
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    """Linear attention mechanism with higher computational efficiency"""
    def __init__(self, dim, heads=4, dim_head=32, spatial_dims=2):
        super().__init__()
        self.heads = heads
        self.spatial_dims = spatial_dims
        hidden_dim = dim_head * heads

        # Select 2D or 3D convolution
        conv_class = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.to_qkv = conv_class(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = conv_class(hidden_dim, dim, 1)

    def forward(self, x):
        if self.spatial_dims == 3:
            # 3D data: (B, C, D, H, W)
            b, c, d, h, w = x.shape
            qkv = self.to_qkv(x)
            q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)',
                                heads=self.heads, qkv=3)
        else:
            # 2D data: (B, C, H, W)
            b, c, h, w = x.shape
            qkv = self.to_qkv(x)
            q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)',
                                heads=self.heads, qkv=3)

        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)

        if self.spatial_dims == 3:
            out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w',
                            heads=self.heads, d=d, h=h, w=w)
        else:
            out = rearrange(out, 'b heads c (h w) -> b (heads c) h w',
                            heads=self.heads, h=h, w=w)

        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    """Spatial self-attention mechanism"""
    def __init__(self, in_channels, spatial_dims=2):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims

        self.norm = Normalize(in_channels)

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
            # 3D data: (B, C, D, H, W)
            b, c, d, h, w = q.shape
            q = rearrange(q, 'b c d h w -> b (d h w) c')
            k = rearrange(k, 'b c d h w -> b c (d h w)')
        else:
            # 2D data: (B, C, H, W)
            b, c, h, w = q.shape
            q = rearrange(q, 'b c h w -> b (h w) c')
            k = rearrange(k, 'b c h w -> b c (h w)')

        w_ = torch.einsum('bij,bjk->bik', q, k)
        w_ = w_ * (int(c )**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        if self.spatial_dims == 3:
            v = rearrange(v, 'b c d h w -> b c (d h w)')
        else:
            v = rearrange(v, 'b c h w -> b c (h w)')

        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)

        if self.spatial_dims == 3:
            h_ = rearrange(h_, 'b c (d h w) -> b c d h w', d=d, h=h, w=w)
        else:
            h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h, w=w)

        h_ = self.proj_out(h_)
        return x + h_


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for condition injection

    Modification notes:
    - Support multi-condition input(text + porosity)
    - Optimize memory usage
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # Attention computation
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MultiConditionCrossAttention(nn.Module):
    """
    Multi-condition cross-attention supporting text and porosity conditions

    """
    def __init__(self, query_dim, text_dim=None, porosity_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        text_dim = default(text_dim, query_dim)
        porosity_dim = default(porosity_dim, 1)  # porosity is usually scalar

        self.scale = dim_head ** -0.5
        self.heads = heads

        # Query projection
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)

        # Text condition projection
        self.to_k_text = nn.Linear(text_dim, inner_dim, bias=False)
        self.to_v_text = nn.Linear(text_dim, inner_dim, bias=False)

        # Porosity condition projection
        self.to_k_porosity = nn.Linear(porosity_dim, inner_dim, bias=False)
        self.to_v_porosity = nn.Linear(porosity_dim, inner_dim, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim * 2, query_dim),  # double dimension for condition fusion
            nn.Dropout(dropout)
        )

        # Parameters
        self.text_weight = nn.Parameter(torch.tensor(1.0))
        self.porosity_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, text_context=None, porosity_context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        # Text condition processing
        if text_context is not None:
            k_text = self.to_k_text(text_context)
            v_text = self.to_v_text(text_context)
            k_text, v_text = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                                 (k_text, v_text))
        else:
            k_text, v_text = None, None

        # Porosity condition processing
        if porosity_context is not None:
            # Ensure porosity condition has the correct shape
            if porosity_context.dim() == 1:
                porosity_context = porosity_context.unsqueeze(-1)
            k_porosity = self.to_k_porosity(porosity_context)
            v_porosity = self.to_v_porosity(porosity_context)
            k_porosity, v_porosity = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                                         (k_porosity, v_porosity))
        else:
            k_porosity, v_porosity = None, None

        # Compute attention
        outputs = []

        if k_text is not None and v_text is not None:
            sim_text = einsum('b i d, b j d -> b i j',
                              rearrange(q, 'b n (h d) -> (b h) n d', h=h),
                              k_text) * self.scale
            attn_text = sim_text.softmax(dim=-1)
            out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)
            out_text = rearrange(out_text, '(b h) n d -> b n (h d)', h=h)
            outputs.append(out_text * self.text_weight)

        if k_porosity is not None and v_porosity is not None:
            sim_porosity = einsum('b i d, b j d -> b i j',
                                  rearrange(q, 'b n (h d) -> (b h) n d', h=h),
                                  k_porosity) * self.scale
            attn_porosity = sim_porosity.softmax(dim=-1)
            out_porosity = einsum('b i j, b j d -> b i d', attn_porosity, v_porosity)
            out_porosity = rearrange(out_porosity, '(b h) n d -> b n (h d)', h=h)
            outputs.append(out_porosity * self.porosity_weight)

        # Fuse outputs from different conditions
        if outputs:
            out = torch.cat(outputs, dim=-1)
        else:
            # If no condition exists, use self-attention
            out = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None,
                 gated_ff=True, checkpoint=True, disable_self_attn=False):
        super().__init__()
        
        # :disable_self_attnParameters
        self.disable_self_attn = disable_self_attn
        
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads,
                                   dim_head=d_head, dropout=dropout,
                                   context_dim=context_dim if not disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                   heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # Key fix: use stored disable_self_attn attribute
        if not self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context) + x
        else:
            # If self-attention is disabled, use only cross-attention
            x = self.attn1(self.norm1(x), context=context) + x
            
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
        
  
class SpatialTransformer(nn.Module):
   
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, dims=3):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.n_heads = n_heads
        self.d_head = d_head
        self.dims = dims
        
        print(f"🔧 Initialize SpatialTransformer: input channels{in_channels}, heads{n_heads}, dimensions{dims}")
        
        # Use 1D convolution to process sequence data
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1)
        
        
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim, n_heads, d_head, dropout=dropout,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,  # Parameters
                checkpoint=use_checkpoint
            )
            for d in range(depth)
        ])
        
        self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1))

    def forward(self, x, context=None):
        
        b, c, *spatial = x.shape
        x_in = x
        
       
        if self.dims == 3:
            # dimensions
            x_reshaped = x.reshape(b, c, -1)  # [b, c, d*h*w]
        else:
            # 2D data
            x_reshaped = x.reshape(b, c, -1)  # [b, c, h*w]
        
        # Project input - use 1D convolution
        x_reshaped = self.proj_in(x_reshaped)  # [b, inner_dim, sequence_length]
        x_reshaped = x_reshaped.transpose(1, 2)  # [b, sequence_length, inner_dim]
        
        # Apply Transformer blocks
        for block in self.transformer_blocks:
            x_reshaped = block(x_reshaped, context=context)
        
        # Project output and restore shape
        x_reshaped = x_reshaped.transpose(1, 2)  # [b, inner_dim, sequence_length]
        x_reshaped = self.proj_out(x_reshaped)  # [b, c, sequence_length]
        
        # Restore original spatial shape
        if self.dims == 3:
            x_out = x_reshaped.reshape(b, c, *spatial)  # [b, c, d, h, w]
        else:
            x_out = x_reshaped.reshape(b, c, *spatial)  # [b, c, h, w]
        
        # print(f"✅ SpatialTransformeroutput: shape{x_out.shape}")
        return x_out + x_in



class Rock3DAttentionUtils:
   

    @staticmethod
    def create_3d_attention_config(base_channels=128, heads=8, depth=4):
        """
        Create 3D attention config

        Returns:
            dict: attention config dictionary
        """
        return {
            "in_channels": base_channels * 4,  # corresponds to UNet middle channel count
            "n_heads": heads,
            "d_head": base_channels // heads,
            "depth": depth,
            "dropout": 0.1,
            "spatial_dims": 3,
            "use_multi_condition": True
        }

    @staticmethod
    def process_conditions(text_embeddings, porosity_values, device='cuda'):
        """
        Process condition information for attention mechanism

        Args:
            text_embeddings: text embeddings
            porosity_values: porosity values
            device: device

        Returns:
            tuple: processed conditions
        """
        # Process text condition
        if text_embeddings is not None:
            if isinstance(text_embeddings, list):
 
                text_embeddings = torch.randn(len(text_embeddings), 77, 768, device=device)
            text_context = text_embeddings
        else:
            text_context = None

        # Process porosity condition
        if porosity_values is not None:
            if isinstance(porosity_values, (int, float)):
                porosity_values = torch.tensor([porosity_values], device=device)
            elif porosity_values.dim() == 1:
                porosity_values = porosity_values.unsqueeze(-1)
            porosity_context = porosity_values
        else:
            porosity_context = None

        return text_context, porosity_context

    @staticmethod
    def apply_3d_attention(features, text_context=None, porosity_context=None,
                           attention_layer=None, config=None):
        """
        Apply 3D attention

        Args:
            features: input features
            text_context: text condition
            porosity_context: porosity condition
            attention_layer: attention layer(created if None)
            config: config dictionary

        Returns:
            torch.Tensor: features after attention processing
        """
        if attention_layer is None:
            if config is None:
                config = Rock3DAttentionUtils.create_3d_attention_config()
            attention_layer = SpatialTransformer(**config)

        # Process conditions
        text_processed, porosity_processed = Rock3DAttentionUtils.process_conditions(
            text_context, porosity_context, features.device
        )

        # Apply attention
        if attention_layer.use_multi_condition:
            return attention_layer(features, text_context=text_processed,
                                   porosity_context=porosity_processed)
        else:
            # Merge conditions(if multi-condition handling is not needed)
            if text_processed is not None and porosity_processed is not None:
                # Simply concatenate conditions
                combined_context = torch.cat([text_processed, porosity_processed.expand_as(text_processed)], dim=-1)
                return attention_layer(features, context=combined_context)
            elif text_processed is not None:
                return attention_layer(features, context=text_processed)
            elif porosity_processed is not None:
                return attention_layer(features, context=porosity_processed)
            else:
                return attention_layer(features)