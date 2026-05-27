"""Caduceus model for Hugging Face.

"""

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
#from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.modules.mamba_simple import Mamba
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, MaskedLMOutput, SequenceClassifierOutput

# import mamba_ssm
# print('mamba#,',mamba_ssm.__file__)
import mamba_ssm
import os
print("Loaded mamba_ssm from:", os.path.dirname(mamba_ssm.__file__))

# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# 用mamba-ssm
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from .configuration_caduceus import CaduceusConfig
from .modeling_rcps import RCPSAddNormWrapper, RCPSEmbedding, RCPSLMHead, RCPSMambaBlock
from torch import nn, Tensor
class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        rcps=False,
        device=None,
        dtype=None,
):
    """Create Caduceus block.

    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
        "bidirectional_weight_tie": bidirectional_weight_tie,
    }
    mixer_cls = partial(BiMambaWrapper, layer_idx=layer_idx, **ssm_cfg, **bidirectional_kwargs, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block_cls = RCPSMambaBlock if rcps else Block  #rcps false
    block = block_cls(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: Optional[str] = "add",
            bidirectional_weight_tie: bool = True,
            **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional:
            self.mamba_rev = Mamba(
                d_model=d_model,
                **mamba_kwargs
            )
            #true
            if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None
            
        # 门控融合 前向和反向 mamba
        # self.gate = nn.Sequential(
        #     nn.Linear(2*d_model, d_model),
        #     nn.Sigmoid()
        # )
        #self.proj = nn.Linear(2 * d_model, d_model)
        #self.dropout=nn.Dropout(0.1)
        # self.strand_gate = nn.Sequential(
        #     nn.Linear(8+2*d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, 1)  # 输出标量 gate
        # )
    def forward(self, hidden_states, inference_params=None,**mixer_kwargs):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                # 门控融合
                #gate=self.gate(torch.cat([out, out_rev], dim=-1))
                #out=gate * out + (1 - gate) * out_rev
                strand = mixer_kwargs.get('strand', None)
                #strand = mixer_kwargs['strand']
                #print("wrapper strand#,",strand)
                # if strand is not None:
                #     # 执行 strand-aware 逻辑
                # # 例如门控、embedding等
                #     strand = strand.unsqueeze(1).expand(-1, out.shape[1], -1) 
                #     #print('all shape#,',out.shape,out_rev.shape,strand.shape)
                #     gate_input = self.strand_gate(torch.cat([out, out_rev,strand], dim=-1) )
                # else:
                #     # 不存在 strand，就按原始逻辑处理
                #     pass
                
                # out = gate_input * out + (1 - gate_input) *  out_rev
                out=out + out_rev
                #print('gate#,',gate)
                #out = self.dropout(out + out_rev)
                #out = self.proj(torch.cat([out, out_rev], dim=-1))
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"`{self.bidirectional_strategy}` for bi-directionality not implemented!")
        return out


class CaduceusEmbeddings(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if config.rcps:
            self.word_embeddings = RCPSEmbedding(
                config.vocab_size, config.d_model, config.complement_map, **factory_kwargs
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)

    def forward(self, input_ids):
        """
            input_ids: (batch, seqlen)
        """
        return self.word_embeddings(input_ids)

class RelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_relative_position=16, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_relative_position = max_relative_position

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 相对位置嵌入（双向）
        self.relative_position_embeddings = nn.Embedding(2 * max_relative_position + 1, self.head_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        输入与 nn.MultiheadAttention 兼容：
        - query, key, value: [L, B, D] 或 [B, L, D]，这里统一使用 [B, L, D]
        - attn_mask: [L, L] 
        - key_padding_mask: [B, L]
        返回:
        - attn_output: [B, L, D]
        - attn_weights: [B, H, L, L]
        """
        B, L, D = query.shape
        H = self.num_heads

        # 投影
        Q = self.q_proj(query).view(B, L, H, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        K = self.k_proj(key).view(B, L, H, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, L, H, self.head_dim).transpose(1, 2)

        # 相对位置偏移索引矩阵
        positions = torch.arange(L, device=query.device)
        rel_pos = positions[None, :] - positions[:, None]  # i - j
        rel_pos = rel_pos.clamp(-self.max_relative_position, self.max_relative_position)
        rel_pos = rel_pos + self.max_relative_position  # shift to positive range
        rel_emb = self.relative_position_embeddings(rel_pos)  # [L, L, d]

        # 注意力打分
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, L, L]

        # 相对位置加权
        Q_rel = Q.permute(2, 0, 1, 3)  # [L, B, H, d]
        # einsum改写保证维度兼容
        #rel_scores = torch.einsum('lbhm,lld->bhll', Q_rel, rel_emb)  # [B, H, L, L]
        # Q: [B, H, L, d], rel_emb: [L, L, d]
        # 输出: [B, H, L, L]
        rel_scores = torch.einsum('bhld,lmd->bhlm', Q, rel_emb)  # 正确

        attn_scores = (attn_scores + rel_scores) * self.scale

        # mask处理
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, None, :], float('-inf')
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [B, H, L, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(attn_output)
        return output, attn_weights


class TransformerLayer(torch.nn.TransformerEncoderLayer):
    # Pre-LN structure
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,batch_first=True,max_relative_position=256):
        super(TransformerLayer, self).__init__(d_model, nhead, dim_feedforward, dropout)
    #    encoder_layers = TransformerLayer(hidden, 
    #                                       nhead = 8,
    #                                       dropout = 0.1,
    
    #                                       dim_feedforward = 512,
    #                                       batch_first = True)
        # Use the custom attention layer instead of the default one
        self.self_attn = RelativeMultiheadAttention(embed_dim=d_model, num_heads=nhead, max_relative_position=max_relative_position, dropout=0.1)
        #self.self_attn = MultiheadAttentionWithRoPE(d_model=d_model, n_heads=nhead, base=10000.0)

   
   
    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        # MHA section
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(src_norm, src_norm, src_norm, 
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        # src_side, attn_weights = self.self_attn(src_norm,  
        #                             attn_mask=src_mask,
        #                             key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src_side)

        # MLP section
        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        return src, attn_weights
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    """
    Compute sinusoidal rotary embeddings for RoPE
    """
    def __init__(self, dim_head, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        #positions = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        
        positions = torch.arange(seq_len, device=device).float() / float(64)  # 127 rope_scale
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        # [seq_len, dim_head/2] -> expand to [seq_len, dim_head]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply RoPE to query and key
    q, k: [batch, n_head, seq_len, head_dim]
    cos, sin: [seq_len, head_dim]
    """
    # broadcast
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    # split even and odd
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    q_rot = torch.stack([-q2, q1], dim=-1).reshape_as(q)
    k_rot = torch.stack([-k2, k1], dim=-1).reshape_as(k)

    q_out = q * cos + q_rot * sin
    k_out = k * cos + k_rot * sin
    return q_out, k_out


class MultiheadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, base=10000,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(self.d_head, base=base)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        x: [batch, seq_len, d_model]
        """
        B, L, _ = query.size()
        Q = self.q_proj(query).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # ---- Rotary position encoding ----
        cos, sin = self.rope(L, query.device)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # ---- Attention ----
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)

        # ---- Output projection ----
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.o_proj(out)
        return out, attn_weights
class AttnModule(nn.Module):
    # 8 layer transformer
    def __init__(self, hidden = 128, layers = 4, record_attn = False, inpu_dim = 256):
        super(AttnModule, self).__init__()
        #self.dropout = nn.Dropout(p=0.1)
        self.record_attn = record_attn
      
        encoder_layers = TransformerLayer(hidden, 
                                          nhead = 8,
                                          dropout = 0.1,
                                          dim_feedforward = 512,
                                          batch_first = True)
        self.module = TransformerEncoder(encoder_layers, 
                                         layers, 
                                         record_attn = record_attn)

    def forward(self, x):
        output = self.module(x)
        return output
    
class AttnMambaModule(nn.Module):
    # 8 layer transformer
    def __init__(self, config):
        super(AttnMambaModule, self).__init__()
        #self.dropout = nn.Dropout(p=0.1)
        self.record_attn = config.record_attn

        encoder_layers = TransformerLayer(config.d_model, 
                                          nhead = 8,
                                          dropout = 0.1,
                                          dim_feedforward = 512,
                                          batch_first = True)
        # self.module = TransformerEncoder(encoder_layers, 
        #                                  config.n_layer, 
        #                                  record_attn = self.record_attn)
        self.layers = self._get_clones(encoder_layers, config.n_layer)
        self.caduceus = Caduceus(config.caduceus_cfg, **{'ignore_embed_layer': True})
    def forward(self, hidden_states):
        # outputs = self.caduceus(
        #         input_ids=None,
        #         inputs_embeds=inputs_embeds,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        attn_weight_list = []
       
        # if self.norm is not None:
        #     output = self.norm(output)

        count=0
        for layer in self.layers:
            # 在2层mamba后插入一层attn
            if count==2:
                #self.attn.self_attn.max_relative_position=seq_len
                hidden_states = self.caduceus(
                    input_ids=None,
                    inputs_embeds=hidden_states,
                    output_hidden_states=False,
                    return_dict=False,
                )
                #print("attn:",hidden_states.shape)
            
            hidden_states, attn_weights = layer(hidden_states)
            attn_weight_list.append(attn_weights.unsqueeze(0).detach())
            count=count+1
        if self.record_attn:
            return hidden_states, torch.cat(attn_weight_list)
        else:
            return hidden_states
        #output = self.module(x)
        return hidden_states
    def _get_clones(self, module, N):
        return torch.nn.modules.ModuleList([copy.deepcopy(module) for i in range(N)])
import copy
class TransformerEncoder(torch.nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, norm=None, record_attn = False):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers)
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.record_attn = record_attn

    def forward(self, src, mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        attn_weight_list = []

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weight_list.append(attn_weights.unsqueeze(0).detach())
        if self.norm is not None:
            output = self.norm(output)

        if self.record_attn:
            return output, torch.cat(attn_weight_list)
        else:
            return output

    def _get_clones(self, module, N):
        return torch.nn.modules.ModuleList([copy.deepcopy(module) for i in range(N)])


#最优传输
#输入 和 输出




class CaduceusMixerModel(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
            ignore_embed_layer=False,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.fused_add_norm = config.fused_add_norm
        self.rcps = config.rcps
        self.residual_in_fp32 = config.residual_in_fp32

        if not ignore_embed_layer:
            self.embeddings = CaduceusEmbeddings(config, **factory_kwargs)

        # Mamba changes the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        print('configd_model#,',config.d_model)
        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    bidirectional=config.bidirectional,
                    bidirectional_strategy=config.bidirectional_strategy,
                    bidirectional_weight_tie=config.bidirectional_weight_tie,
                    rcps=config.rcps,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )
        #self.pos_encoder = RelativePositionEmbedding(config.d_model, dropout = 0.1)
        #2025 11 8 添加 一层transformer
        # self.attn=TransformerLayer(config.d_model,   # 256
        #                                   nhead = 8,
        #                                   dropout = 0.1,
        #                                   dim_feedforward = 512,
        #                                   batch_first = True,max_relative_position=157)

        norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)( #rms_norm true
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )
        self.norm_f = norm_f if (config.fused_add_norm or not config.rcps) else RCPSAddNormWrapper(norm_f)

    def forward(self, input_ids, inputs_embeds=None,output_hidden_states=False,strand=None):
        """Mixer forward."""
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        residual = None
        count=0
        seq_len=hidden_states.shape[1]
        #print('mixer strand#,',strand) # 输出
        mixer_kwargs = {}
        if strand is not None:
            mixer_kwargs['strand'] = strand
        for layer in self.layers:  # 0
            # 在2层mamba后插入一层attn
            # if count==2:
            #     self.attn.self_attn.max_relative_position=seq_len
            #     hidden_states, attn_weights = self.attn(hidden_states)
            #     print("attn:",hidden_states.shape)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            # TODO: Add support for gradient checkpointing
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None,**mixer_kwargs
            )
            count=count+1

        if not self.fused_add_norm: #true 
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states = self.norm_f(hidden_states, residual=residual, prenorm=False)
            else:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype)) #RMSNorm
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn # rms_norm_fn
            if self.rcps: #false
                # Set prenorm=False here since we don't need the residual
                hidden_states_fwd = fused_add_norm_fn(
                    hidden_states[..., :hidden_states.shape[-1] // 2],
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., :hidden_states.shape[-1] // 2],
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states_rc = fused_add_norm_fn(
                    hidden_states[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states = torch.cat([hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1)
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states

from src.tasks.encoders import EncoderSplitMore
class CaduceusCNNMixerModel(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
            ignore_embed_layer=False,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.fused_add_norm = config.fused_add_norm
        self.rcps = config.rcps
        self.residual_in_fp32 = config.residual_in_fp32

        if not ignore_embed_layer:
            self.embeddings = CaduceusEmbeddings(config, **factory_kwargs)

        # Mamba changes the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    bidirectional=config.bidirectional,
                    bidirectional_strategy=config.bidirectional_strategy,
                    bidirectional_weight_tie=config.bidirectional_weight_tie,
                    rcps=config.rcps,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )
        #self.pos_encoder = RelativePositionEmbedding(config.d_model, dropout = 0.1)
        #2025 11 8 添加 一层transformer
        self.attn=TransformerLayer(config.d_model,   # 256
                                          nhead = 8,
                                          dropout = 0.1,
                                          dim_feedforward = 512,
                                          batch_first = True,max_relative_position=157)
        self.cnn=EncoderSplitMore(config.signal_size)
        norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)( #rms_norm true
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )
        self.norm_f = norm_f if (config.fused_add_norm or not config.rcps) else RCPSAddNormWrapper(norm_f)

    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=False):
        """Mixer forward."""
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        residual = None
        count=0
        seq_len=hidden_states.shape[1]
        for layer in self.layers:
            # 在2层mamba后插入一层attn
            if count==2:
                # self.attn.self_attn.max_relative_position=seq_len
                # hidden_states, attn_weights = self.attn(hidden_states)
                # print("attn:",hidden_states.shape)
                self.cnn(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            # TODO: Add support for gradient checkpointing
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )
            count=count+1

        if not self.fused_add_norm: #true 
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states = self.norm_f(hidden_states, residual=residual, prenorm=False)
            else:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype)) #RMSNorm
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn # rms_norm_fn
            if self.rcps: #false
                # Set prenorm=False here since we don't need the residual
                hidden_states_fwd = fused_add_norm_fn(
                    hidden_states[..., :hidden_states.shape[-1] // 2],
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., :hidden_states.shape[-1] // 2],
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states_rc = fused_add_norm_fn(
                    hidden_states[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states = torch.cat([hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1)
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states

def cross_entropy(logits, y, ignore_index=-100):
    """Cross entropy loss."""
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y, ignore_index=ignore_index)


def weighted_cross_entropy(logits, y, loss_weights, ignore_index=-100):
    """Weighted cross entropy loss (discounts certain tokens, e.g., repeated base pairs in genome)."""
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="none")
    loss_weights = loss_weights.view(-1)
    loss_weights[y == ignore_index] = 0.0
    # TODO: Follows GPN implementation, but should we remove weight normalization?
    return (ce * (loss_weights / loss_weights.sum())).sum()


class CaduceusPreTrainedModel(PreTrainedModel):
    """PreTrainedModel wrapper for Caduceus backbone."""
    config_class = CaduceusConfig
    base_model_prefix = "caduceus"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BiMambaWrapper"]

    def _init_weights(
            self,
            module,
            initializer_range=0.02,  # Now only used for embedding layer.
            **kwargs,
    ):
        """Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py"""

        n_layer = self.config.n_layer
        initialized_cfg = self.config.initializer_cfg if self.config.initializer_cfg is not None else {}
        rescale_prenorm_residual = initialized_cfg.get("rescale_prenorm_residual", True)
        initializer_range = initialized_cfg.get("initializer_range", initializer_range)
        n_residuals_per_layer = initialized_cfg.get("n_residuals_per_layer", 1)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth.
            #   > Scale the weights of residual layers at initialization by a factor of 1/√N where N is the # of
            #   residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


class Caduceus(CaduceusPreTrainedModel):
    """Caduceus model that can be instantiated using HF patterns."""
    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config)

        if config.rcps:
            assert config.complement_map is not None, "Complement map must be provided for RCPS."

        # Adjust vocab size and complement maps if vocab padding is set.
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)
        if config.complement_map is not None and config.vocab_size > len(config.complement_map):
            for i in range(len(config.complement_map), config.vocab_size):
                config.complement_map[i] = i

        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = CaduceusMixerModel(config, **factory_kwargs, **kwargs)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            strand: Optional[torch.Tensor] = None,   # 新增参数
    ) -> Union[torch.Tensor, Tuple, BaseModelOutputWithNoAttention]:
        """HF-compatible forward method."""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, all_hidden_states = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            strand=strand
        )
        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None
            )
        elif output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class CaduceusForMaskedLM(CaduceusPreTrainedModel):
    """HF-compatible Caduceus model for masked language modeling."""

    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        if config.rcps:
            self.lm_head = RCPSLMHead(
                complement_map=self.config.complement_map,  # Use caduceus config as it might have been updated
                vocab_size=self.config.vocab_size,  # Use caduceus config as it might have been updated
                true_dim=config.d_model,
                dtype=dtype
            )
        else:
            self.lm_head = nn.Linear(
                config.d_model,
                self.config.vocab_size,  # Use caduceus config as it might have been updated
                bias=False,
                **factory_kwargs
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if self.config.rcps:
            raise NotImplementedError("Setting input embeddings for RCPS LM is not supported.")
        self.caduceus.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Overrides output embeddings."""
        if self.config.rcps:
            raise NotImplementedError("Setting output embeddings for RCPS LM is not supported.")
        self.lm_head = new_embeddings

    def tie_weights(self):
        """Tie weights, accounting for RCPS."""
        if self.config.rcps:
            self.lm_head.set_weight(self.get_input_embeddings().weight)
        else:
            super().tie_weights()

    def get_decoder(self):
        """Get decoder (backbone) for the model."""
        return self.caduceus

    def set_decoder(self, decoder):
        """Set decoder (backbone) for the model."""
        self.caduceus = decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """HF-compatible forward method."""

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.caduceus(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            if loss_weights is not None:
                loss = weighted_cross_entropy(logits, labels, loss_weights, ignore_index=self.config.pad_token_id)
            else:
                loss = cross_entropy(logits, labels, ignore_index=self.config.pad_token_id)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class CaduceusForSequenceClassification(CaduceusPreTrainedModel):
    def __init__(
            self,
            config: CaduceusConfig,
            pooling_strategy: str = "mean",
            conjoin_train: bool = False,
            conjoin_eval: bool = False,
            device=None,
            dtype=None,
            **kwargs):
        super().__init__(config, **kwargs)
        if pooling_strategy not in ["mean", "max", "first", "last"]:
            raise NotImplementedError(f"Pooling strategy `{pooling_strategy}` not implemented.")
        self.pooling_strategy = pooling_strategy
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)

        self.conjoin_train = conjoin_train
        self.conjoin_eval = conjoin_eval

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if self.config.rcps:
            raise NotImplementedError("Setting input embeddings for RCPS LM is not supported.")
        self.caduceus.backbone.embeddings.word_embeddings = value

    def pool_hidden_states(self, hidden_states, sequence_length_dim=1):
        """Pools hidden states along sequence length dimension."""
        if self.pooling_strategy == "mean":  # Mean pooling along sequence length dimension
            return hidden_states.mean(dim=sequence_length_dim)
        if self.pooling_strategy == "max":  # Max pooling along sequence length dimension
            return hidden_states.max(dim=sequence_length_dim).values
        if self.pooling_strategy == "last":  # Use embedding of last token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[-1, ...]
        if self.pooling_strategy == "first":  # Use embedding of first token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[0, ...]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get hidden representations from the backbone
        if self.config.rcps:  # Hidden states have 2 * d_model channels for RCPS
            transformer_outputs = self.caduceus(
                input_ids,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = torch.stack(
                [
                    transformer_outputs[0][..., :self.config.d_model],
                    torch.flip(transformer_outputs[0][..., self.config.d_model:], dims=[1, 2])
                 ],
                dim=-1
            )
        elif self.conjoin_train or (self.conjoin_eval and not self.training):  # For conjoining / post-hoc conjoining
            assert input_ids is not None, "`input_ids` must be provided for conjoining."
            assert input_ids.ndim == 3, "`input_ids` must be 3D tensor: channels corresponds to forward and rc strands."
            transformer_outputs = self.caduceus(
                input_ids[..., 0],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            transformer_outputs_rc = self.caduceus(
                input_ids[..., 1],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # Stack along channel dimension (dim=-1)
            hidden_states = torch.stack([transformer_outputs[0], transformer_outputs_rc[0]], dim=-1)
        else:
            transformer_outputs = self.caduceus(
                input_ids,
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

        # Pool and get logits
        pooled_hidden_states = self.pool_hidden_states(hidden_states)
        # Potentially run `score` twice (with parameters shared) for conjoining
        if hidden_states.ndim == 4:  # bsz, seq_len, hidden_dim, 2 where last channel has the stacked fwd and rc reps
            logits_fwd = self.score(pooled_hidden_states[..., 0])
            logits_rc = self.score(pooled_hidden_states[..., 1])
            logits = (logits_fwd + logits_rc) / 2
        else:
            logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )
