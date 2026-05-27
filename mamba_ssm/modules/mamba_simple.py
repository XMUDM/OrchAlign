# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
causal_conv1d_fn=None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
print('mamba RMSNorm#,',RMSNorm)
class MultiScaleCausalGatedConv1D(nn.Module):
    def __init__(self, total_width, kernel_sizes):
        """
        total_width: 通道数 = embedding 维度
        kernel_sizes: list[int], 比如 [3, 5, 7]
        """
        super().__init__()
        self.total_width = total_width
        self.kernel_sizes = kernel_sizes

        # 多尺度 depthwise 因果卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=total_width,
                out_channels=total_width,
                kernel_size=k,
                groups=total_width,
                padding=k - 1  # 只左 padding，保证因果性
            )
            for k in kernel_sizes
        ])

        # 门控权重，用于融合每个尺度的输出：输出维度为 (B, len(kernel_sizes), L)
        self.gate_mlp = nn.Sequential(
            nn.Conv1d(total_width, len(kernel_sizes), kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, C, L] 输入
        return: [B, C, L] 输出
        """
        conv_outputs = []
        input_dtype = x.dtype
        for conv, k in zip(self.convs, self.kernel_sizes):
            out = conv(x)  # [B, C, L + k - 1]
            out = out[:, :, :x.size(-1)]  # 截断成 [B, C, L]
            conv_outputs.append(out)

        # [B, C, L] × len → [B, len, C, L]
        conv_stack = torch.stack(conv_outputs, dim=1)  # [B, K, C, L]

        # 生成门控权重（每个位置对每个 kernel size 有一个权重）
        gates = self.gate_mlp(x)  # [B, K, L]
        # print('gate weights:',gates.shape,gates[0]) # [64, 3, 1023]
        gates = gates.unsqueeze(2)  # [B, K, 1, L]

        # 加权求和
        out = (conv_stack * gates).sum(dim=1)  # [B, C, L]
        return out.to(input_dtype)

# class LinearAttention(nn.Module):
#     """
#     Linear attention using kernel feature map.
#     Paper inspiration: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
#     Uses ϕ(x)=elu(x)+1 feature map (simple).
#     - x: [B, L, D]
#     - returns: [B, L, D] approximate global attention
#     """
#     def __init__(self, dim, n_heads=8, kernel_fn=None, eps=1e-6, dropout=0.0):
#         super().__init__()
#         assert dim % n_heads == 0
#         self.dim = dim
#         self.n_heads = n_heads
#         self.d_head = dim // n_heads
#         self.qkv = nn.Linear(dim, dim*3, bias=False)
#         self.out = nn.Linear(dim, dim)
#         self.eps = eps
#         self.dropout = nn.Dropout(dropout)
#         # default kernel feature map
#         self.kernel_fn = kernel_fn if kernel_fn is not None else (lambda x: F.elu(x) + 1.0)

#     def forward(self, x):
#         # x: [B, L, D]
#         B, L, D = x.shape
#         #print("LA#,",B,L,D) #4 2000 256
#         #RuntimeError: mat1 and mat2 shapes cannot be multiplied (1024x2000 and 128x384)
#         qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head).permute(2,0,3,1,4)
#         # qkv: [3, B, heads, L, d]
#         q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, L, d]
#         # apply kernel map on last dim
#         q = self.kernel_fn(q)  # [B,H,L,d]
#         k = self.kernel_fn(k)

#         # compute KV = sum_k (k * v)  -> [B,H,d,d?] but we do per head: sum over sequence pos
#         # we want: out_i = q_i @ (sum_j k_j^T v_j) / (q_i @ sum_j k_j)
#         # compute S = sum_j k_j * v_j (note v has last dim d, so we want S shape [B,H,d])
#         # careful with dims: k: [B,H,L,d], v: [B,H,L,d]
#         kv = torch.einsum('bhld,bhld->bhld', k, v)  # elementwise (we need outer?), actually we need k_j^T v_j -> but using feature map get vector form
#         # Standard efficient formula:
#         # numerator: q_i @ (K^T V) = q_i * (sum_j k_j * v_j) (with broadcasting)
#         # denominator: q_i @ sum_j k_j
#         sum_k = k.sum(dim=2)  # [B,H,d]
#         sum_kv = (k.unsqueeze(-1) * v.unsqueeze(-2)).sum(dim=2)  # [B,H,d,d]? that's heavy
#         # Simplify with per-dimension weighted sum:
#         # We follow common approximation: treat v as a vector in same dim, compute sum_j k_j * v_j (elementwise multiply then sum over seq) -> [B,H,d]
#         sum_kv = (k * v).sum(dim=2)  # [B,H,d]
#         # numerator for each position: (q_i * sum_kv).sum(dim=-1, keepdim=True)? That would give scalar per head, not vector.
#         # Instead we compute elementwise: out_i = q_i * (sum_kv / (sum_k + eps))
#         denom = (q * sum_k.unsqueeze(2)).sum(dim=-1, keepdim=True)  # shape [B,H,L,1]
#         # However above is not correct shape, let's use the common simpler formula used in practice:
#         # out = (q @ (k^T v)) / (q @ sum_k)
#         # Implement by broadcasting:
#         # first compute context = sum_j k_j * v_j  -> [B,H,d]
#         context = sum_kv  # [B,H,d]
#         # broadcast multiply:
#         out = q * context.unsqueeze(2)  # [B,H,L,d]  (elementwise)
#         denom = (q * sum_k.unsqueeze(2)).sum(dim=-1, keepdim=True)  # [B,H,L,1]
#         out = out / (denom + self.eps)
#         out = out.view(B, self.n_heads, L, self.d_head).permute(0,2,1,3).contiguous().view(B, L, D)
#         out = self.out(out)
#         return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearAttention(nn.Module):
    """
    标准Linear Attention实现，复杂度O(N)
    """
    def __init__(self, d_model, n_heads=8, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        #self.attention_dropout = nn.Dropout(0.1)
    def elu_feature_map(self, x):
        """ELU特征映射，确保特征非负"""
        return F.elu(x) + 1
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 重形状为多头
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 应用特征映射
        q = self.elu_feature_map(q)
        k = self.elu_feature_map(k)
        
        # Linear Attention计算
        k_v = torch.einsum('bhid,bhio->bhdo', k, v)  # (K^T V)
        Z = k.sum(dim=2, keepdim=True)  # 归一化因子
        
        # 注意力输出: Q (K^T V) / Z
        attn_out = torch.einsum('bhid,bhdo->bhio', q, k_v) / (Z + self.eps)
        #attn_out = self.attention_dropout(attn_out)
        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_out)

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        #self.att=LinearAttention(self.d_model*2)
        #self.att=None

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        #print('zx#,',batch)
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            print('use_fast_path#,',self.use_fast_path)
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            #print('z#,',z.shape,self.d_model) #z#, z#, torch.Size([4, 512, 2000]) 256
            # z 进行 注意力机制 B,D
            # if self.d_model==256:
            #     if self.att is None:
            #         self.att=LinearAttention(self.d_model*2).to(z.device)
            #     z1=self.att(z.transpose(1,2))
            #     #z=z1.transpose(1,2)+z  # 残差
            #     z=z1.transpose(1,2)  # 残差
            #z1=self.att(z.transpose(1,2))
            #z=z1.transpose(1,2)+z  # 残差
            #z=z1.transpose(1,2)
            #print('z#,',z.shape)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)
        print('zx#,',x.shape) # 2
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
