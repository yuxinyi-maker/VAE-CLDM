"""
This script implements a complete Transformer encoder architecture, including:

1,multiple positional encoding methods(absolute positional encoding,fixed positional encoding)
2,multiple normalization methods(LayerNorm,ScaleNorm,RMSNorm)
3,multiple attention mechanisms(standard attention,sparse attention,text)
4,flexible network architecture configuration(text,Residual connectiontext)

"""
"""shout-out to https://github.com/lucidrains/x-transformers/tree/main/x_transformers"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from collections import namedtuple
from einops import rearrange, repeat, reduce

# Add imports to adapt previous changes
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldm.util_new import default, exists  # textUtility functions

# constants
DEFAULT_DIM_HEAD = 64

# textStoretext
Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',  # softmaxtextattention weights
    'post_softmax_attn'  # softmaxtextattention weights
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',  # hidden states
    'attn_intermediates'  # attention intermediate results
])


class AbsolutePositionalEmbedding(nn.Module):
    """absolute positional encoding:text"""
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        """Initialize weights"""
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        """Forward pass"""
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]


class FixedPositionalEmbedding(nn.Module):
    """fixed positional encoding:text"""
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        """Forward pass"""
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]




# textkeytextparameterstext
def pick_and_pop(keys, d):
    """textkey"""
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    """text"""
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    """textcharacterstext"""
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    """textkeytext"""
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    """textkeytext"""
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


# Base module classes
class Scale(nn.Module):
    """text:text"""
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward pass"""
        x, *rest = self.fn(x, **kwargs)
        return (x * self.value, *rest)


class Rezero(nn.Module):
    """ReZerotext:textparameters"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        """Forward pass"""
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)


class ScaleNorm(nn.Module):
    """text:text"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward pass"""
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Module):
    """RMStext:text"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Forward pass"""
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Residual(nn.Module):
    """Residual connection:text"""
    def forward(self, x, residual):
        return x + residual


class GRUGating(nn.Module):
    """GRUtext:textGRUtext"""
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):
        """Forward pass"""
        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )
        return gated_output.reshape_as(x)


# Feed-forward networktext
class GEGLU(nn.Module):
    """GEGLUtext:text"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        """Forward pass"""
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network:textlinear transformationtext"""
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
        """Forward pass"""
        return self.net(x)


# Attention mechanism module
class Attention(nn.Module):
    """Self-attention mechanism:textquery,key,valuetext"""
    def __init__(
            self,
            dim,
            dim_head=DEFAULT_DIM_HEAD,
            heads=8,
            causal=False,
            mask=None,
            talking_heads=False,
            sparse_topk=None,
            use_entmax15=False,
            num_mem_kv=0,
            dropout=0.,
            on_attn=False
    ):
        super().__init__()
        if use_entmax15:
            raise NotImplementedError("Please use entmax activation instead of softmax activation!")

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        inner_dim = dim_head * heads

        # query,key,valuetextlinear transformation
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # talking headstext
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))

        # Sparse top-k attention
        self.sparse_topk = sparse_topk

        # text
        self.attn_fn = F.softmax

        # textkeyvaluetext
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # text
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim * 2), nn.GLU()) if on_attn else nn.Linear(inner_dim, dim)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            rel_pos=None,
            sinusoidal_emb=None,
            prev_attn=None,
            mem=None
    ):
        """Forward pass"""
        b, n, _, h, talking_heads, device = *x.shape, self.heads, self.talking_heads, x.device
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)

        if exists(sinusoidal_emb):
            # textshortformertext,querytext
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        # textquery,key,value
        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        # text
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # text
        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device=device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device=device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask

        # textkeyvaluetext
        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)

        # Compute attention scores
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots

        # talking headstext
        if talking_heads:
            dots = einsum('b h i j, h k -> b k i j', dots, self.pre_softmax_proj).contiguous()

        # Relative positional encoding
        if exists(rel_pos):
            dots = rel_pos(dots)

        # Apply mask
        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        # Causal mask(text)
        if self.causal:
            i, j = dots.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, 'i -> () () i ()') < rearrange(r, 'j -> () () () j')
            mask = F.pad(mask, (j - i, 0), value=False)
            dots.masked_fill_(mask, mask_value)
            del mask

        # Sparse top-k attention
        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask

        # textattention weights
        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn

        attn = self.dropout(attn)

        # talking headstext
        if talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj).contiguous()

        # textvaluetext
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Storetext
        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn,
            post_softmax_attn=post_softmax_attn
        )

        return self.to_out(out), intermediates


class AttentionLayers(nn.Module):
    """Attention layer stack:text"""
    def __init__(
            self,
            dim,
            depth,
            heads=8,
            causal=False,
            cross_attend=False,
            only_cross=False,
            use_scalenorm=False,
            use_rmsnorm=False,
            use_rezero=False,
            rel_pos_num_buckets=32,
            rel_pos_max_distance=128,
            position_infused_attn=False,
            custom_layers=None,
            sandwich_coef=None,
            par_ratio=None,
            residual_attn=False,
            cross_residual_attn=False,
            macaron=False,
            pre_norm=True,
            gate_residual=False,
            **kwargs
    ):
        super().__init__()
        # textFeed-forward networktextparameters
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        # Positional encodingtext
        self.has_pos_emb = position_infused_attn
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else always(None)
        self.rotary_pos_emb = always(None)

        # Relative positional encodingtext
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'text'
        self.rel_pos = None

        self.pre_norm = pre_norm

        # text
        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn

        # text
        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        # text
        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')  # text,text,text
        elif cross_attend and only_cross:
            default_block = ('c', 'f')  # text,text
        else:
            default_block = ('a', 'f')  # text,text

        # Macarontext
        if macaron:
            default_block = ('f',) + default_block

        # text
        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            # PAR(Product Key)text
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'partext'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # PARtext2/3text
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'textpartext'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            # text
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'text'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # Build layers
        for layer_type in self.layer_types:
            if layer_type == 'a':
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)

            # Residual connectiontext
            if gate_residual:
                residual_fn = GRUGating(dim)
            else:
                residual_fn = Residual()

            self.layers.append(nn.ModuleList([
                norm_fn(),
                layer,
                residual_fn
            ]))

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            mems=None,
            return_hiddens=False
    ):
        """Forward pass"""
        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        # text
        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == 'a':
                hiddens.append(x)
                layer_mem = mems.pop(0)

            residual = x

            # text
            if self.pre_norm:
                x = norm(x)

            # text
            if layer_type == 'a':
                out, inter = block(x, mask=mask, sinusoidal_emb=self.pia_pos_emb, rel_pos=self.rel_pos,
                                   prev_attn=prev_attn, mem=layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask, prev_attn=prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            # Residual connection
            x = residual_fn(out, residual)

            # Storetext
            if layer_type in ('a', 'c'):
                intermediates.append(inter)

            # text
            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            # text
            if not self.pre_norm and not is_last:
                x = norm(x)

        # text
        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates
            )
            return x, intermediates

        return x


class Encoder(AttentionLayers):
    """text:textAttention layer stack"""
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'encoder cannot set causality'
        super().__init__(causal=False, **kwargs)


class TransformerWrapper(nn.Module):
    """Transformer wrapper:text,Positional encodingtextOutput layer"""
    def __init__(
            self,
            *,
            num_tokens,
            max_seq_len,
            attn_layers,
            emb_dim=None,
            max_mem_len=0.,
            emb_dropout=0.,
            num_memory_tokens=None,
            tie_embedding=False,
            use_pos_emb=True
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'textEncodertextDecoder'

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.num_tokens = num_tokens

        # Token embedding layer
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        # Positional encoding
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) if (
                use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        # text
        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        # Output layer
        self.to_logits = nn.Linear(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()

        # memory token(text[cls])
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

            # text,textfunneltextmemory tokentext
            if hasattr(attn_layers, 'num_memory_tokens'):
                attn_layers.num_memory_tokens = num_memory_tokens

    def init_(self):
        """Initialize weights"""
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(
            self,
            x,
            return_embeddings=False,
            mask=None,
            return_mems=False,
            return_attn=False,
            mems=None,
            **kwargs
    ):
        """Forward pass"""
        b, n, device, num_mem = *x.shape, x.device, self.num_memory_tokens

        # text
        x = self.token_emb(x)
        x += self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)

        # textmemory token
        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=b)
            x = torch.cat((mem, x), dim=1)

            # textmemory tokentext
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        # text
        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        # textmemory tokentexttoken
        mem, x = x[:, :num_mem], x[:, num_mem:]

        # text
        out = self.to_logits(x) if not return_embeddings else x

        # text
        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))
            return out, new_mems

        # Return attention maps
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out