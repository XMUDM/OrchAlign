import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import re
import torch.distributions as dist
from collections import namedtuple

from transformers.modeling_outputs import MaskedLMOutput
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MixerModel

from caduceus.modeling_caduceus import CaduceusPreTrainedModel, Caduceus,AttnModule,AttnMambaModule

from caduceus.configuration_caduceus import CaduceusConfig, ExtendedMambaConfig, AttnConfig,AttnMambaConfig

from src.models.sequence.EPInformer import EPInformer_v2
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.models.sequence.long_conv_lm import LMBackbone
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
def local_entropy(T_sub):
    mass = T_sub.sum(dim=(1,2), keepdim=True) + 1e-12
    Tn = T_sub / mass
    H = -(Tn * (Tn + 1e-12).log()).sum(dim=(1,2))
    return H



def orthogonal_loss(shared: torch.Tensor, specific: torch.Tensor, reduction='mean'):
    """
    计算 shared 和 specific 特征的正交约束损失
    输入:
        shared: [B, L, D]
        specific: [B, L, D]
    输出:
        scalar loss
    """
    shared_norm = F.normalize(shared, dim=-1)  # [B, L, D]
    specific_norm = F.normalize(specific, dim=-1)  # [B, L, D]

    
    dot = torch.sum(shared_norm * specific_norm, dim=-1)

    loss = dot.pow(2) 

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  

class DecouplerKeepDim(nn.Module):
   
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.shared_proj = nn.Linear(dim, dim//2)
        self.specific_proj = nn.Linear(dim, dim//2)

    def forward(self, x):
        shared = F.layer_norm(self.shared_proj(x), (self.dim//2,))
        specific = F.layer_norm(self.specific_proj(x), (self.dim//2,))
        return shared, specific
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.shared_proj = nn.Linear(dim, dim)
        self.specific_proj = nn.Linear(dim, dim)

    def forward(self, x):
        shared = F.layer_norm(self.shared_proj(x), (self.dim,))
        specific = F.layer_norm(self.specific_proj(x), (self.dim,))
        return shared, specific



import torch
def entropy(mask, eps=1e-10):
    entropy = -mask * torch.log(mask + eps) - (1 - mask) * torch.log(1 - mask + eps)
    average_entropy = entropy.mean()
    return average_entropy


def smooth_max(tensor, window_size):
    """take max value of signals over window size"""
    tensor = tensor.unsqueeze(1)
    smoothed_tensor = F.max_pool1d(tensor, kernel_size=window_size, stride=1, padding=(window_size - 1) // 2)
    smoothed_tensor = smoothed_tensor.squeeze(1)
    return smoothed_tensor


def moving_average_cal(logits, mv_kernel=1, stride=1, padding_value=0.5, padding_mode='same'):
    # moving average
    kernel = torch.ones((logits.shape[-1], 1, mv_kernel)) / mv_kernel
    kernel = kernel.to(logits.device, dtype=logits.dtype)
    kernel.requires_grad = False

    logits = logits.permute(0, 2, 1)
    if padding_mode == 'same':
        right_pad = (mv_kernel - 1) // 2 if (mv_kernel - 1) % 2 == 0 else (mv_kernel - 1) // 2 + 1
        left_pad = (mv_kernel - 1) // 2
    elif padding_mode == 'no_pad':
        left_pad = right_pad = 0
    logits_padded = F.pad(logits, (left_pad, right_pad), mode='constant', value=padding_value)
    logits = F.conv1d(logits_padded, kernel, stride=stride, padding=0, groups=logits.shape[1])
    logits = logits.permute(0, 2, 1)

    return logits


def gumbel_softmax_threshold(logits, tau: float = 1, hard: bool = False, dim: int = -1, threshold=0.5, mv_kernel=1,
                             bio_mask=None, bio_mask_weight=0.0, counter_zero=True, merge_mask=False, subseq_size=1000,
                             node_merge_mask=False, node_merge_range=1, is_training=False):
    if node_merge_mask:
        logits = moving_average_cal(logits, mv_kernel=node_merge_range, stride=node_merge_range, padding_mode='no_pad')

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if node_merge_mask:
        y_soft = torch.repeat_interleave(y_soft, repeats=node_merge_range, dim=1)

    # moving average
    y_soft = moving_average_cal(y_soft, mv_kernel=mv_kernel, stride=1, padding_value=0.5, padding_mode='same')

    if bio_mask is not None and bio_mask_weight != 0.0:
        bio_mask_true = bio_mask[..., 0]
        bio_mask_complement = torch.zeros_like(bio_mask_true) if counter_zero else 1 - bio_mask_true
        bio_mask = torch.concat((bio_mask_complement.unsqueeze(-1), bio_mask_true.unsqueeze(-1)), dim=-1)
        y_soft = (1 - bio_mask_weight) * y_soft + bio_mask_weight * bio_mask

    # make sure the middle 2k part is 1
    promo_mask = torch.zeros_like(y_soft[:,:,1])
    start = (promo_mask.shape[1] - 2000) // 2
    promo_mask[:, start:start + 2000] = 1.1
    y1 = torch.max(y_soft[:, :, 1], promo_mask)
    y_soft = torch.concat((y_soft[:, :, 0:1], y1.unsqueeze(-1)), dim=-1)

    if hard:
        # Straight through.
        max_vals, index = y_soft.max(dim, keepdim=True)
        y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        if not is_training:
            mask = max_vals > threshold
            y_hard = y_hard * mask.to(dtype=y_soft.dtype)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    # if merge_mask:
    #     ret_repeat = torch.repeat_interleave(ret, repeats=subseq_size, dim=1)
    #     tensor1 = F.pad(ret_repeat, (0, 0, subseq_size, 0))  # (bs, 1000, dim)
    #     tensor2 = F.pad(ret_repeat, (0, 0, 0, subseq_size))  # (bs, 1000, dim)
    #     ret = torch.max(tensor1, tensor2)
    return ret


def Beta_fn(a, b):
    return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))


def kldivergence_kuma(distribution, prior_alpha, prior_beta):
    distribution.a = distribution.concentration1
    distribution.b = distribution.concentration0
    # prior_alpha = torch.tensor([1.0], device=distribution.a.device)
    # prior_beta = torch.tensor([4.0], device=distribution.a.device)
    kl = 1. / (1 + distribution.a * distribution.b) * Beta_fn(distribution.a.reciprocal(), distribution.b)
    kl += 1. / (2 + distribution.a * distribution.b) * Beta_fn(2.0 * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (3 + distribution.a * distribution.b) * Beta_fn(3. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (4 + distribution.a * distribution.b) * Beta_fn(4. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (5 + distribution.a * distribution.b) * Beta_fn(5. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (6 + distribution.a * distribution.b) * Beta_fn(6. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (7 + distribution.a * distribution.b) * Beta_fn(7. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (8 + distribution.a * distribution.b) * Beta_fn(8. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (9 + distribution.a * distribution.b) * Beta_fn(9. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (10 + distribution.a * distribution.b) * Beta_fn(10. * distribution.a.reciprocal(), distribution.b)
    kl *= (prior_beta - 1) * distribution.b

    # use another taylor approx for Digamma function
    psi_b_taylor_approx = torch.log(distribution.b) - 1. / (2 * distribution.b) - 1. / (12 * distribution.b ** 2)
    kl += (distribution.a - prior_alpha) / distribution.a * (
                -0.57721 - psi_b_taylor_approx - 1 / distribution.b)  # T.psi(self.posterior_b)

    # add normalization constants
    kl += torch.log(distribution.a * distribution.b) + torch.log(Beta_fn(prior_alpha, prior_beta))

    # final term
    kl += -(distribution.b - 1) / distribution.b

    return kl


class GeneExpHyena(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_inner: int,
        vocab_size: int,
        process_group=None,
        layer=None,
        attn_layer_idx=None,
        attn_cfg=None,
        max_position_embeddings=0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        dropout_cls=nn.Dropout,
        layer_norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        residual_in_fp32=False,
        pad_vocab_size_multiple: int = 1,
        sequence_parallel=True,
        checkpoint_mlp=False,
        checkpoint_mixer=False,
        device=None,
        dtype=None,
        interact='',
        use_bio_mask=False,
        base_size=4,
        signal_size=3,
        center_len=2000,
        rna_feat_dim=9,
        useRNAFeat=True,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.center_len = center_len
        self.useRNAFeat = useRNAFeat

        self.model = LMBackbone(
            d_model=d_model,
            n_layer=n_layer,
            d_inner=d_inner,
            vocab_size=vocab_size,
            process_group=process_group,
            layer=layer,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            dropout_cls=dropout_cls,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg,
            fused_mlp=fused_mlp,
            fused_dropout_add_ln=fused_dropout_add_ln,
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel,
            checkpoint_mlp=checkpoint_mlp,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs,
            **kwargs,
        )

        self.pToExpr = nn.Sequential(
            nn.Linear(d_model + rna_feat_dim if useRNAFeat else d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        seqs,
        signals,
        rna_feat=None,
        bio_mask=None,
        mask_regions=None,
        peak_mask=None,
        output_hidden_states=False,
        return_dict=False,
    ):
        hidden_states = self.model(
            seqs, position_ids=None, inference_params=None
        )

        if self.center_len:
            start_index = (hidden_states.shape[1] - self.center_len) // 2
            end_index = start_index + self.center_len
            hidden_states = hidden_states[:, start_index: end_index, :]

        hidden_states = torch.mean(hidden_states, dim=1)

        p_embed = torch.cat([hidden_states, rna_feat], dim=-1) if self.useRNAFeat else hidden_states
        logits = self.pToExpr(p_embed)

        logits = logits.float()

        return logits


class GeneExpMamba(nn.Module):
    def __init__(
            self,
            config: ExtendedMambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
    ):
        super().__init__()
        # if config.interact == 'concat':
        #     input_dim = config.base_size + config.signal_size
        # elif config.interact == 'no_signal':
        #     input_dim = config.base_size
        # self.input_layer = nn.Linear(input_dim, config.d_model)

        self.config = config
        self.model = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=config.vocab_size,
            ssm_cfg=config.ssm_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            **{"device": device, "dtype": dtype},
        )

        self.pToExpr = nn.Sequential(
            nn.Linear(config.d_model + config.rna_feat_dim if config.useRNAFeat else config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        seqs,
        signals,
        rna_feat=None,
        bio_mask=None,
        mask_regions=None,
        peak_mask=None,
        output_hidden_states=False,
        return_dict=False,
    ):

        hidden_states = self.model(input_ids=seqs)

        if self.config.center_len:
          
            start_index = (hidden_states.shape[1] - self.config.center_len) // 2
            end_index = start_index + self.config.center_len
            hidden_states = hidden_states[:, start_index: end_index, :]

        hidden_states = torch.mean(hidden_states, dim=1)

        p_embed = torch.cat([hidden_states, rna_feat], dim=-1) if self.config.useRNAFeat else hidden_states
        logits = self.pToExpr(p_embed)

        logits = logits.float()

        return logits

class GeneExpTransformer(nn.Module):
    def __init__(self, config: AttnConfig, device=None, dtype=None, **kwargs):
        super().__init__()
   
        self.config = config
        self.attn = AttnModule(hidden = config.d_model, layers=config.n_layer,record_attn = config.record_attn)
        signal_size=config.signal_size
        
        self.signal_size=signal_size
        self.record_attn=config.record_attn
        
        # 中间加一层mamba
        
        #self.caduceus = Caduceus(config, **{'ignore_embed_layer': True})
       
        self.pToExpr = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )
        # Initialize weights and apply final processing
        #self.post_init()

    def forward(
        self,
        seqs,
        signals,
        rna_feat=None,
        bio_mask=None,
        mask_regions=None,
        peak_mask=None,
        output_hidden_states=False,
        return_dict=False,
    ):
        
        inputs_embeds = torch.concat((seqs, signals), dim=-1)
        print('inputs#,',inputs_embeds.shape) #torch.Size([8, 625, 128]) torch.Size([8, 20000, 6])
        if self.record_attn:
            outputs, attn_weights = self.attn(inputs_embeds)
        else:
            #x = self.attn(x)
            outputs = self.attn(inputs_embeds)

        hidden_states = outputs
        if self.config.center_len:
            start_index = (hidden_states.shape[1] - self.config.center_len) // 2
            end_index = start_index + self.config.center_len
            hidden_states = hidden_states[:, start_index: end_index, :]

        hidden_states = torch.mean(hidden_states, dim=1)

        #p_embed = torch.cat([hidden_states, rna_feat], dim=-1) if self.config.useRNAFeat else hidden_states
        #logits = self.pToExpr(p_embed)
        logits = self.pToExpr(hidden_states)

        logits = logits.float()

        return logits,seqs,signals
    @property
    def n_epi(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "signal_size", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.signal_size

class GeneExpTransformerMamba(nn.Module):
    def __init__(self, config: AttnMambaConfig, device=None, dtype=None, **kwargs):
        super().__init__()
   
        self.config = config
        self.attn = AttnMambaModule(config)
        signal_size=config.signal_size
        
        self.signal_size=signal_size
        self.record_attn=config.record_attn
        
        # 中间加一层mamba
        self.pToExpr = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )
        # Initialize weights and apply final processing
        #self.post_init()

    def forward(
        self,
        seqs,
        signals,
        rna_feat=None,
        bio_mask=None,
        mask_regions=None,
        peak_mask=None,
        output_hidden_states=False,
        return_dict=False,
    ):
        
        inputs_embeds = torch.concat((seqs, signals), dim=-1)
        print('inputs#,',inputs_embeds.shape) #torch.Size([8, 625, 128]) torch.Size([8, 20000, 6])
        if self.record_attn:
            outputs, attn_weights = self.attn(inputs_embeds)
        else:
            #x = self.attn(x)
            outputs = self.attn(inputs_embeds)

        hidden_states = outputs
        if self.config.center_len:
            start_index = (hidden_states.shape[1] - self.config.center_len) // 2
            end_index = start_index + self.config.center_len
            hidden_states = hidden_states[:, start_index: end_index, :]

        hidden_states = torch.mean(hidden_states, dim=1)

        #p_embed = torch.cat([hidden_states, rna_feat], dim=-1) if self.config.useRNAFeat else hidden_states
        #logits = self.pToExpr(p_embed)
        logits = self.pToExpr(hidden_states)

        logits = logits.float()

        return logits,seqs,signals
    @property
    def n_epi(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "signal_size", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.signal_size
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hidden_states):
        scores = self.att_net(hidden_states)  # [B, L, 1]
        weights = F.softmax(scores, dim=1) # [B, L, 1] [B, L, d]
        return (weights * hidden_states).sum(dim=1)# [B, d]


from src.tasks.encoders import EncoderSplitMore,EncoderSplit2cov
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# utils
# -----------------------
def squared_euclid(a, b):
    # a: (B, n, D), b: (B, m, D)
    # returns cost matrix (B, n, m) with squared euclidean distances
    # Efficient: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    an = (a ** 2).sum(dim=2, keepdim=True)  # (B, n, 1)
    bm = (b ** 2).sum(dim=2, keepdim=True).transpose(1, 2)  # (B, 1, m)
    ab = torch.bmm(a, b.transpose(1, 2))  # (B, n, m)
    return torch.clamp(an + bm - 2.0 * ab, min=0.0)


def positional_cost_matrix(L, device, decay='abs'):
    # returns a (L, L) matrix with position-based cost, normalized to [0,1]
    # decay options: 'abs' -> |i-j| / (L-1); 'sq' -> ((i-j)/(L-1))^2
    idx = torch.arange(L, device=device, dtype=torch.float32)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)  # (L, L)
    if decay == 'abs':
        P = torch.abs(diff) / max(1.0, (L - 1))
    elif decay == 'sq':
        P = (diff / max(1.0, (L - 1))) ** 2
    else:
        P = torch.abs(diff) / max(1.0, (L - 1))
    return P  # not batched

import numpy as np
import torch

# Log-domain Sinkhorn 
# -----------------------
def sinkhorn_log_domain(M, a=None, b=None, epsilon=0.05, n_iters=50):
    """
    Log-domain Sinkhorn to solve:
        argmin_T <T, M> + epsilon * KL(T || a b^T)
    Inputs:
      M: (B, n, m) cost matrices
      a: (B, n) source marginals (if None -> uniform)
      b: (B, m) target marginals (if None -> uniform)
      epsilon: entropic regularization (float > 0)
      n_iters: sinkhorn iterations
    Returns:
      T: (B, n, m) transport matrices (rows/cols approx sum to a,b)
    """
    B, n, m = M.shape
    device = M.device
    dtype = M.dtype

    if a is None:
        a = torch.full((B, n), 1.0 / n, device=device, dtype=dtype)
    if b is None:
        b = torch.full((B, m), 1.0 / m, device=device, dtype=dtype)

    # log K = -M / epsilon
    logK = -M / (epsilon + 1e-30)  # (B, n, m)
    # initialize dual vectors log_u=0
    log_u = torch.zeros((B, n), device=device, dtype=dtype)
    log_a = torch.log(a + 1e-30)
    log_b = torch.log(b + 1e-30)
    for _ in range(n_iters):
        logK_plus_u = logK + log_u.unsqueeze(2)  # (B,n,m)
        log_Ku = torch.logsumexp(logK_plus_u, dim=1)  # (B, m)
        log_v = log_b - log_Ku
        logK_plus_v = logK + log_v.unsqueeze(1)  # (B,n,m)
        log_Kv = torch.logsumexp(logK_plus_v, dim=2)  # (B, n)
        log_u = log_a - log_Kv
        
    log_T = log_u.unsqueeze(2) + logK + log_v.unsqueeze(1)  # (B,n,m)
    T = torch.exp(log_T)
    T = T / (T.sum(dim=(1,2), keepdim=True) + 1e-30)
    return T

import numpy as np
# -----------------------
class CrossModalSinkhornLayer(nn.Module):
    """
    Inputs:
      dna: (B, L, D1)
      epi: (B, L, D2)
    Returns:
      fused_dna: (B, L, Df)
      fused_epi: (B, L, Df)
      T: (B, Ls, Ls) OT coupling used for mapping
    """
    def __init__(self,
                 D1,
                 D2,
                 proj_dim=64,
                 hidden_dim=None,
                 alpha=1.0,
                 beta=0.0,
                 pos_decay='abs',
                 epsilon=0.05,
                 sinkhorn_iters=40,
                 token_subsample=None,
                 use_layernorm=True):
        super().__init__()
        self.proj_dim = proj_dim
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.token_subsample = token_subsample  # int or None
        self.pos_decay = pos_decay
        # optional layernorm on projections
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln_x = nn.LayerNorm(D1)
            self.ln_y = nn.LayerNorm(D2)

        # gating fusion: merge original and mapped features
        merged_dim = D1 + D2
        if hidden_dim is None:
            hidden_dim = merged_dim // 2 if merged_dim >= 16 else merged_dim

        self.gate = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, merged_dim),
            nn.Sigmoid()
        )
    def forward(self, dna, epi, a=None, b=None):
        B, L, D1 = dna.shape
        device = dna.device
    
        if self.token_subsample is not None and self.token_subsample < L:
            Ls = self.token_subsample
          
            start_index = (L - self.token_subsample) // 2
        
            end_index = start_index + self.token_subsample
            dna_sub = dna[:, start_index:end_index, :]  # (B, Ls, D1)
            epi_sub = epi[:, start_index:end_index, :]  # (B, Ls, D2)
            
            idx_map = None
        else:
            dna_sub = dna
            epi_sub = epi
            Ls = L
            idx_map = None

  
        px=dna_sub
        py=epi_sub
        if self.use_layernorm:
            px = self.ln_x(px)
            py = self.ln_y(py)

        feat_cost = squared_euclid(px, py)  # (B, Ls, Ls)
        M = self.alpha * feat_cost # (B, Ls, Ls)
        if a is None:
            a = torch.full((B, Ls), 1.0 / Ls, device=device, dtype=dna.dtype)
        if b is None:
            b = torch.full((B, Ls), 1.0 / Ls, device=device, dtype=dna.dtype)

        T = sinkhorn_log_domain(M, a=a, b=b, epsilon=self.epsilon, n_iters=self.sinkhorn_iters)  # (B, Ls, Ls)
       
     
        row_sum = T.sum(dim=2, keepdim=True).clamp(min=1e-12)
        W_row = T / row_sum  # (B, Ls, Ls)
        dna_mapped = torch.bmm(W_row, epi_sub)  # (B, Ls, D2)  # dna mapped into epi space

        col_sum = T.sum(dim=1, keepdim=True).clamp(min=1e-12)

        epi_mapped = torch.bmm(T.transpose(1, 2) / (T.transpose(1, 2).sum(dim=2, keepdim=True).clamp(min=1e-12)), dna_sub)
      
        if idx_map is not None:
         
            px_all = self.f(dna) if idx_map is not None else px  # (B,L,proj_dim)
            py_all = self.g(epi) if idx_map is not None else py
            if self.use_layernorm:
                px_all = self.ln_x(px_all)
                py_all = self.ln_y(py_all)

       
            sim_x = torch.bmm(F.normalize(px_all, dim=2), F.normalize(px, dim=2).transpose(1, 2))
            nn_idx = sim_x.argmax(dim=2)  # (B, L)
           
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, L)  # (B, L)
            dna_mapped_full = dna_mapped[batch_idx, nn_idx]  # (B, L, D2)

           
            sim_y = torch.bmm(F.normalize(py_all, dim=2), F.normalize(py, dim=2).transpose(1, 2))
            nn_idx_y = sim_y.argmax(dim=2)
            epi_mapped_full = epi_mapped[batch_idx, nn_idx_y]  # (B, L, D1)
        else:
            dna_mapped_full = dna_mapped  # (B, L, D2)
            epi_mapped_full = epi_mapped  # (B, L, D1)
        fused = torch.cat([dna_mapped_full, epi_mapped_full], dim=-1)  # (B, L, D2 + D1)
   
        gate_vals = self.gate(fused)  # (B, L, D1+D2), values in (0,1)
        orig_cat = torch.cat([dna_sub, epi_sub], dim=-1)  # (B, L, D1+D2)
        fused_out = gate_vals * fused + (1.0 - gate_vals) * orig_cat
        
        return fused_out, T

def row_entropy(T):
    # T: (B, Lq, Lk)
        eps = 1e-12
        e = -(T * (T + eps).log()).sum(dim=-1)  # (B, Lq)
        return e.mean().item()
def topk_concentration(T, k=3):
     # T: (B, Lq, Lk)
    topk = T.topk(k, dim=-1).values  # (B, Lq, k)
    return topk.sum(dim=-1).mean().item()


def gini_coef(T):
    # T: (B, Lq, Lk)
    T_sorted = T.sort(dim=-1).values  # ascending
    L = T.size(-1)
    index = torch.arange(1, L + 1, device=T.device).float()
    gini = (2 * (index * T_sorted).sum(dim=-1) / (L * T.sum(dim=-1)) - (L + 1)) / (L - 1)
    return gini.mean().item()
def analyze_ot_distribution(T, name="OT"):
    print(f"--- {name} ---")
    print("entropy:", row_entropy(T))
    print("top1:", topk_concentration(T, k=1))
    print("top3:", topk_concentration(T, k=3))
    print("top5:", topk_concentration(T, k=5))
    print("gini:", gini_coef(T))

def gumbel_topk(scores, k, tau=1.0,eps = 0.01):
    """
    scores: (B, L)
    return:
        topk_idx (B, k)  -> 硬选择的位置
        soft_mask (B, L) -> 反向传播用的soft mask
    """
    if scores.dim() == 3:
        scores = scores.squeeze(-1)

    B, L = scores.shape
    device = scores.device

    
    # 1. Gumbel 噪声
    #gumbel = -torch.log(-torch.log(torch.rand_like(scores)))
    gumbel = -torch.log(-torch.log(torch.rand_like(scores).clamp(eps, 1 - eps)))
    #gumbel = torch.rand_like(scores).clamp(eps, 1 - eps)
    noisy = (scores + gumbel) / tau
    # print('gumbel#,',gumbel)
    # print('noisy#,',noisy)
    # print("gumbel min score:", gumbel.min().item())
    # print("gumbel max score:", gumbel.max().item())
    # print("gumbel any exactly 0:", (gumbel == 0).any().item())
    # print("gumbel any exactly 1:", (gumbel == 1).any().item())
    
    # print("noisy min score:", noisy.min().item())
    # print("noisy max score:", noisy.max().item())
    # print("noisy any exactly 0:", (noisy == 0).any().item())
    # print("noisy any exactly 1:", (noisy == 1).any().item())
    
    # 2. top-k (硬)
    topk_vals, topk_idx = noisy.topk(k, dim=-1)
    
    

    # 3. soft mask 作为梯度替代
    soft = F.softmax(noisy, dim=-1)

    hard = torch.zeros_like(soft)
    hard.scatter_(1, topk_idx, 1.0)

    # straight-through estimator
    soft_hard = hard + soft - soft.detach()

    return topk_idx, soft_hard

from src.models.sequence.OT import CrossModalGWLayer
from src.models.sequence.denoise import TwoStageDenoiseOTModel,TwoStageDenoiseOTModel2,CrossGuidedDenoiser
from src.models.sequence.gate import SharedToSpecGate
from src.models.sequence.KL import kl_token_alignment_loss_F
from src.models.sequence.distal import DistalGateNoBuffer
class GeneExpMambaCross(CaduceusPreTrainedModel):
    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        gen_config = copy.deepcopy(config)
        gen_config1 = copy.deepcopy(config)
        
        gen_config.n_layer = config.gen_n_layer//2
        gen_config1.n_layer = config.gen_n_layer//2
        if config.pretrained_model:
    
            state_dict = AutoModelForMaskedLM.from_pretrained(config.pretrained_model_name, trust_remote_code=True).state_dict()
            state_dict = {key.replace("caduceus.", ""): value for key, value in state_dict.items()}
            self.pre_model = Caduceus(config, **{'ignore_embed_layer': True})
            missing_keys, unexpected_keys = self.pre_model.load_state_dict(state_dict, strict=False)
            assert len(missing_keys) == 0

            if config.pretrained_freeze:
                for param in self.pre_model.parameters():
                    param.requires_grad = False
        else:
            gen_config.d_model = config.d_model//2
            self.caduceus_dna = Caduceus(gen_config, **{'ignore_embed_layer': True})
            self.caduceus_epi = Caduceus(gen_config, **{'ignore_embed_layer': True})
        signal_size=config.signal_size
        
        self.signal_size=signal_size
        self.promo_len = config.center_len
        self.seq_input_layer = nn.Linear(config.base_size, config.d_model//2)
        if config.interact == 'concat':  
            self.signal_input_layer = nn.Linear(config.signal_size, config.d_model//2)
        self.OT = CrossModalSinkhornLayer(config.d_model//4, config.d_model//4,
                    proj_dim=64,
                    alpha=1.0,
                    beta=0.0,
                    epsilon=0.01,
                    sinkhorn_iters=100,
                    token_subsample=2000)
        
        
        self.dna_decoupler = DecouplerKeepDim(config.d_model//2)
        self.epi_decoupler = DecouplerKeepDim(config.d_model//2)
        
        self.caduceus_all=Caduceus(gen_config1, **{'ignore_embed_layer': True})
        self.pToExpr = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )
        self.post_init()

    def forward(
        self,
        seqs,
        signals,
        strand=None,
        rna_feat=None,
        bio_mask=None,
        mask_regions=None,
        peak_mask=None,
        output_hidden_states=True,
        return_dict=False,
    ):
       
       
        seqs=self.seq_input_layer(seqs)
        signals=self.signal_input_layer(signals)

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seqs,all_seqs = self.caduceus_dna(
            input_ids=None,
            inputs_embeds=seqs,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        signals,all_signals = self.caduceus_epi(
            input_ids=None,
            inputs_embeds=signals,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # disentanglement
        dna_shared, dna_specific = self.dna_decoupler(seqs)
        epi_shared, epi_specific = self.epi_decoupler(signals)
        
        # original shared features
        oriL=torch.concat((dna_shared,epi_shared), dim=-1) 
        
        # focused ot
        inputs_embeds,T=self.OT(dna_shared,epi_shared)
        
        start_index = (dna_shared.shape[1] - self.config.center_len) // 2
        
        # replace the central region with the output of ot 
        oriL[:,start_index:start_index+self.config.center_len,:]=inputs_embeds
        
        
        othLoss=0.5*(orthogonal_loss(dna_shared,dna_specific)+orthogonal_loss(epi_shared,epi_specific))
        
        
        # a unified representation for fusion
        inputs_embeds=torch.concat((dna_specific, epi_specific,oriL), dim=-1) 
       
        hidden_states,_=self.caduceus_all(input_ids=None,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

       
        if self.config.center_len:
           
            start_index = (hidden_states.shape[1] - self.config.center_len) // 2
           
            end_index = start_index + self.config.center_len
            
            hidden_states = hidden_states[:, start_index: end_index, :]
       
        hidden_states = torch.mean(hidden_states, dim=1)
       
        logits = self.pToExpr(hidden_states)

        logits = logits.float()
        
      
        # kl
        dna_shared_OT=oriL[:,start_index:start_index+self.config.center_len,:dna_specific.shape[-1]]
        dna_specific=dna_specific[:,start_index:start_index+self.config.center_len,:]
        dna_kl=kl_token_alignment_loss_F(dna_specific,dna_shared_OT)
        epi_shared_OT=oriL[:,start_index:start_index+self.config.center_len,dna_specific.shape[-1]:]
        epi_specific=epi_specific[:,start_index:start_index+self.config.center_len,:]
        epi_kl=kl_token_alignment_loss_F(epi_specific,epi_shared_OT)
        
        
        kl_sum=dna_kl+epi_kl
        return logits,kl_sum,othLoss
       
    @property
    def n_epi(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "signal_size", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.signal_size

class GeneExpBiMamba(CaduceusPreTrainedModel):
    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        if config.pretrained_model:
            # self.pre_model = AutoModelForMaskedLM.from_pretrained(config.pretrained_model_name, trust_remote_code=True)
            state_dict = AutoModelForMaskedLM.from_pretrained(config.pretrained_model_name, trust_remote_code=True).state_dict()
            state_dict = {key.replace("caduceus.", ""): value for key, value in state_dict.items()}
            self.pre_model = Caduceus(config, **{'ignore_embed_layer': True})
            missing_keys, unexpected_keys = self.pre_model.load_state_dict(state_dict, strict=False)
            assert len(missing_keys) == 0

            if config.pretrained_freeze:
                for param in self.pre_model.parameters():
                    param.requires_grad = False

        else:
            self.caduceus = Caduceus(config, **{'ignore_embed_layer': True})

        if config.interact == 'concat':
            input_dim = config.base_size + config.signal_size
        elif config.interact == 'no_signal':
            input_dim = config.base_size
            
        self.input_layer = nn.Linear(input_dim, config.d_model) # 5*128

        self.pToExpr = nn.Sequential(
            nn.Linear(config.d_model + config.rna_feat_dim if config.useRNAFeat else config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        seqs,
        signals,
        rna_feat=None,
        bio_mask=None,
        mask_regions=None,
        peak_mask=None,
        output_hidden_states=False,
        return_dict=False,
    ):
        if self.config.interact == 'concat':
            if self.config.signal_size == 2:
                signals = signals[..., :2]
            inputs_embeds = torch.concat((seqs, signals), dim=-1)
        elif self.config.interact == 'no_signal':
            inputs_embeds = seqs

        if self.config.use_bio_mask:
            bio_mask_epinformer = bio_mask[..., 1]
            inputs_embeds = inputs_embeds * bio_mask_epinformer.unsqueeze(-1)

        inputs_embeds = self.input_layer(inputs_embeds)

        if self.config.pretrained_model:
            output_dict = self.pre_model(input_ids=None, inputs_embeds=inputs_embeds, output_hidden_states=True)
            last_hidden_states = output_dict.hidden_states[-1]  # dim=256
            if self.config.use_bio_mask:
                bio_mask_epinformer = bio_mask[..., 1]
                outputs = last_hidden_states * bio_mask_epinformer.unsqueeze(-1)
            else:
                outputs = last_hidden_states
        else:
            """HF-compatible forward method."""
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.caduceus(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = outputs
        if self.config.center_len:
            start_index = (hidden_states.shape[1] - self.config.center_len) // 2
            end_index = start_index + self.config.center_len
            hidden_states = hidden_states[:, start_index: end_index, :]

        hidden_states = torch.mean(hidden_states, dim=1)

        p_embed = torch.cat([hidden_states, rna_feat], dim=-1) if self.config.useRNAFeat else hidden_states
        logits = self.pToExpr(p_embed)

        logits = logits.float()

        return logits


class ModelMask(CaduceusPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.promo_len = config.center_len
        assert not (config.test_top and config.test_soft)
        assert not (config.use_bio_mask and config.use_peak_mask)

      
        self.seq_input_layer = nn.Linear(config.base_size, config.d_model)
        if config.interact == 'concat':  
            self.signal_input_layer = nn.Linear(config.signal_size, config.d_model)

        # generator
        gen_config = copy.deepcopy(config)
        gen_config.n_layer = config.gen_n_layer
        self.generator = Caduceus(gen_config, **{'ignore_embed_layer': True})

        # prior input weight - signal
        prior_weight = [float(x) for x in config.prior_weight.split(',')] # 2,2,1
        prior_weight = [x / sum(prior_weight) for x in prior_weight]
        prior_weight = torch.log(torch.tensor(prior_weight, dtype=torch.float32))
        self.prior_weights = nn.Parameter(prior_weight, requires_grad=config.dist_param_grad)

        # signal beta
        signal_beta = [float(x) for x in config.prior_beta.split(',')]
        signal_beta = torch.tensor(signal_beta, dtype=torch.float32)
        self.signal_betas = nn.Parameter(signal_beta, requires_grad=config.dist_param_grad)

        # mask output
        self.mask_output = nn.Linear(config.d_model, 2)
    
        self.post_init()

    def couple_post_dist(self):
      
        seq_alpha, seq_beta = self.post_distributions.concentration1, self.post_distributions.concentration0

        weights = F.softmax(self.prior_weights, dim=0)
        sig_alpha_total = torch.tensor(0.0, dtype=seq_alpha.dtype, device=seq_alpha.device)
        sig_beta_total = torch.tensor(0.0, dtype=seq_alpha.dtype, device=seq_alpha.device)
        for idx, prior_dist in enumerate(self.prior_dists):
            sig_alpha, sig_beta = prior_dist.concentration1, prior_dist.concentration0 # signal
            sig_alpha_total = sig_alpha_total + sig_alpha * weights[idx]
            sig_beta_total = sig_beta_total + sig_beta * weights[idx]

        # merge two distribution
        if self.config.only_x_sig:
            x_alpha = sig_alpha_total
            x_beta = sig_beta_total
        else:
            x_alpha = seq_alpha + sig_alpha_total  # make value valid, so no -1
            x_beta = seq_beta + sig_beta_total
        x_alpha = x_alpha * self.config.z_scale  #1 z_scale
        x_beta = x_beta * self.config.z_scale
        self.z_distribution = dist.Beta(x_alpha, x_beta)

    def posterior_dist(self, logit, eps=1e-8):
        print("logits#",logit.shape)
        alpha_logits = logit[..., 1]  # alpha, dist to 1
        beta_logits = logit[..., 0]  # beta, dist to 0
        alpha = F.softplus(alpha_logits) + self.config.beta_min
        beta = F.softplus(beta_logits) + self.config.beta_min
        #'beta'
        if self.config.post_dist == 'kuma':
            self.post_distributions = dist.kumaraswamy.Kumaraswamy(alpha, beta)
        elif self.config.post_dist == 'beta':
            self.post_distributions = dist.Beta(alpha + eps, beta + eps)
    def prior_dist(self, signals, eps=1e-8, peak_mask=None):
        if self.config.prior_signal == 'h3k27ac':
            prior_signal = signals[...,0].unsqueeze(-1)
        elif self.config.prior_signal == 'DHS':
            prior_signal = signals[...,1].unsqueeze(-1)
        elif self.config.prior_signal == 'hic':
            prior_signal = signals[...,2].unsqueeze(-1)
        elif self.config.prior_signal == 'all':
            prior_signal = signals
        signal_dim = prior_signal.shape[-1]
        self.prior_dists = []
        for i in range(signal_dim):
            cur_alpha = (prior_signal[...,i] + self.config.beta_min) * self.config.prior_scale_factor
            if self.config.max_pool_size > 0: #0
                smooth_max(cur_alpha, window_size=self.config.max_pool_size)
            # # merge peak mask
            # if self.config.merge_peak_mask:
            #     cur_alpha = torch.where(peak_mask < 0.1, 0.0, cur_alpha)
            distribution = dist.Beta(cur_alpha + eps, (self.signal_betas[i] + self.config.beta_min) * self.config.prior_scale_factor)
            self.prior_dists.append(distribution)

        # add the includelist, regard as a new signal
        if self.config.use_include_list and (not self.config.mask_region_hard): #config.use_include_list false mask_region_hard true
            include_alpha = self.includelist.to(signals.dtype) * self.config.include_alpha
            include_dist = dist.Beta(include_alpha + eps, eps)
            self.prior_dists.append(include_dist)

    def forward(
        self,
        seqs,
        signals,
        mask_regions=None,
        bio_mask=None,
        peak_mask=None,
        rna_feat=None,
    ):
        if self.config.merge_peak_mask:
            expanded_mask = peak_mask.unsqueeze(-1)
            signals = signals * expanded_mask

        self.includelist, self.blacklist = mask_regions[...,0], mask_regions[...,1]
        bs, seq_len, _ = seqs.shape
        seq_input_embeds = self.seq_input_layer(seqs)
        inputs_embeds = seq_input_embeds
        if self.config.interact == 'concat':
            signals = signals[..., :self.config.signal_size]
            signal_input_embeds = self.signal_input_layer(signals)
            if self.config.gen_signal:
                inputs_embeds = seq_input_embeds + signal_input_embeds

        assert not (self.config.use_bio_mask and self.config.use_peak_mask)
        # generator
        outputs = self.generator(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            return_dict=False,
        )
        logits = self.mask_output(outputs)
        # calculate the mask distribution by seqs & signals
        self.posterior_dist(logits) #self.post_distributions
        self.prior_dist(signals, peak_mask=peak_mask)

        if self.config.decouple_x:
            self.couple_post_dist() 
            z_dist = self.z_distribution 
        else:
            z_dist = self.post_distributions

        if self.config.post_sample:
            soft_mask = z_dist.mean
            if self.config.test_top:
                top_num = int(seq_len * self.config.test_top_percent)
                _, indices = torch.topk(soft_mask, top_num, dim=-1)
                hard_mask = torch.zeros_like(soft_mask)
                batch_indices = torch.arange(bs).unsqueeze(-1).expand_as(indices)
                hard_mask[batch_indices, indices] = 1.0
                mask = hard_mask
            elif self.config.test_soft:
                mask = soft_mask
            else:
                if self.config.pool_mask != 0:
                    soft_mask = smooth_max(soft_mask, self.config.pool_mask)
                hard_mask = (soft_mask >= self.config.sample_threshold).float()
                mask = hard_mask
        else:
            mask = F.gumbel_softmax(logits, tau=self.config.gumbel_temp, hard=True, dim=-1)[:,:,1]
        # includelist and blacklist
        mask[self.blacklist] = 0.0
        if self.config.use_include_list and self.config.mask_region_hard:
            mask[self.includelist] = 1.0

        # middle promoter length = 1
        start = (seq_len - self.promo_len) // 2
        mask[:, start:start + self.promo_len] = 1.0
        return mask
import pandas as pd
class GeneBiMambaMIRNP(CaduceusPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.promo_len = config.center_len
        assert not (config.test_top and config.test_soft)
        assert not (config.use_bio_mask and config.use_peak_mask)
        assert (config.test_soft + config.test_hard + config.test_top) <= 1

        # input layer
        self.seq_input_layer = nn.Linear(config.base_size, config.d_model)
        if config.interact == 'concat':
            print('config.signal_size#,',config.signal_size) # 3
            self.signal_input_layer = nn.Linear(config.signal_size, config.d_model) #signal_size: int = 3,

        # position encoding
        if config.pos_enc:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.seq_range, config.d_model))
        #print('use_mask_model#,',config.use_mask_model) # False
        if config.use_mask_model: #
            self.mask_model = ModelMask(config, **kwargs)
            checkpoint = torch.load(config.mask_model)
            new_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
            load = self.mask_model.load_state_dict(new_state_dict, strict=False)
            assert len(load.missing_keys) == 0
            for param in self.mask_model.parameters():
                param.requires_grad = False

        elif (not self.config.use_bio_mask) and (not self.config.use_peak_mask): #false
            # generator
            gen_config = copy.deepcopy(config)
            gen_config.n_layer = config.gen_n_layer  # 4
            self.generator = Caduceus(gen_config, **{'ignore_embed_layer': True})
            prior_weight = [float(x) for x in config.prior_weight.split(',')]
            prior_weight = [x / sum(prior_weight) for x in prior_weight]
            prior_weight = torch.log(torch.tensor(prior_weight, dtype=torch.float32))
            self.prior_weights = nn.Parameter(prior_weight, requires_grad=config.dist_param_grad)

            # signal beta
            signal_beta = [float(x) for x in config.prior_beta.split(',')] #1.5,0.2,1.0
            signal_beta = torch.tensor(signal_beta, dtype=torch.float32)
            self.signal_betas = nn.Parameter(signal_beta, requires_grad=config.dist_param_grad)

#             # mask output
            self.mask_output = nn.Linear(config.d_model, 2)  # 输出mask

            # remove grad
            if config.only_x_sig:
                for param in self.generator.parameters():
                    param.requires_grad = False
                for param in self.mask_output.parameters():
                    param.requires_grad = False

        # encoder
        if config.enc_prd_ps: #false
            # parameter sharing between encoder and predictor
            self.encoder = self.generator
        else:
            enc_config = copy.deepcopy(config)
            enc_config.n_layer = config.enc_n_layer
            self.encoder = Caduceus(enc_config, **{'ignore_embed_layer': True})

        self.pToExpr = nn.Sequential(
            nn.Linear(config.d_model + config.rna_feat_dim if config.useRNAFeat else config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )
        # marginal distribution p(z)
        marginal_mean = torch.tensor([config.marginal_mean])
        self.marginal_beta = (1 - marginal_mean) / marginal_mean
        self.marginal_alpha = torch.tensor([1.0])
        self.post_init()

    def couple_post_dist(self):
        seq_alpha, seq_beta = self.post_distributions.concentration1, self.post_distributions.concentration0
        weights = F.softmax(self.prior_weights, dim=0)
        sig_alpha_total = torch.tensor(0.0, dtype=seq_alpha.dtype, device=seq_alpha.device)
        sig_beta_total = torch.tensor(0.0, dtype=seq_alpha.dtype, device=seq_alpha.device)
        print('prior_dists#,',len(self.prior_dists))
        for idx, prior_dist in enumerate(self.prior_dists): 
            sig_alpha, sig_beta = prior_dist.concentration1, prior_dist.concentration0
            sig_alpha_total = sig_alpha_total + sig_alpha * weights[idx]
            sig_beta_total = sig_beta_total + sig_beta * weights[idx]

        # merge two distribution
        if self.config.only_x_sig: #false
            x_alpha = sig_alpha_total
            x_beta = sig_beta_total  
        else:
            print('seq_alpha#,',seq_alpha.shape) #torch.Size([8, 20000]
            print('sig_alpha_total#,',sig_alpha_total.shape) #torch.Size([8, 20000,2]
            x_alpha = seq_alpha + sig_alpha_total  # make value valid, so no -1
            x_beta = seq_beta + sig_beta_total
        x_alpha = x_alpha * self.config.z_scale
        x_beta = x_beta * self.config.z_scale
        self.z_distribution = dist.Beta(x_alpha, x_beta)

    def posterior_dist(self, logit, eps=1e-8):
        alpha_logits = logit[..., 1]  # alpha, dist to 1
        beta_logits = logit[..., 0]  # beta, dist to 0

        alpha = F.softplus(alpha_logits) + self.config.beta_min
        beta = F.softplus(beta_logits) + self.config.beta_min

        if self.config.post_dist == 'kuma':
            self.post_distributions = dist.kumaraswamy.Kumaraswamy(alpha, beta)
        elif self.config.post_dist == 'beta':
            self.post_distributions = dist.Beta(alpha + eps, beta + eps)
    def prior_dist(self, signals, eps=1e-8, peak_mask=None):
        prior_signal=signals 
        signal_dim = prior_signal.shape[-1]
        self.prior_dists = []
        for i in range(signal_dim):
            cur_alpha = (prior_signal[...,i] + self.config.beta_min) * self.config.prior_scale_factor #10 prior_scale_factor
            if self.config.max_pool_size > 0:
                smooth_max(cur_alpha, window_size=self.config.max_pool_size)
            distribution = dist.Beta(cur_alpha + eps, (self.signal_betas[i] + self.config.beta_min) * self.config.prior_scale_factor)
            self.prior_dists.append(distribution)

        # add the includelist, regard as a new signal
        if self.config.use_include_list and (not self.config.mask_region_hard): #false
            include_alpha = self.includelist.to(signals.dtype) * self.config.include_alpha
            include_dist = dist.Beta(include_alpha + eps, eps)
            self.prior_dists.append(include_dist)

    def kl_divergence(self):
        if self.config.decouple_x:  #False
            marginal_alpha = self.marginal_alpha * self.config.marginal_scale
            marginal_beta = self.marginal_beta * self.config.marginal_scale
            marginal_alpha = marginal_alpha.to(self.prior_weights.device)
            marginal_beta = marginal_beta.to(self.prior_weights.device)
            marginal_z_dist = dist.Beta(marginal_alpha, marginal_beta)
            kl_loss = dist.kl_divergence(self.z_distribution, marginal_z_dist)
            kl_loss = torch.mean(torch.mean(kl_loss, dim=1))
            return kl_loss

        weights = F.softmax(self.prior_weights, dim=0)

        post_dist = self.post_distributions
        prior_dists = self.prior_dists
        kl_loss_total = torch.tensor(0.0, dtype=self.prior_weights.dtype, device=self.prior_weights.device)

        for idx, prior_dist in enumerate(prior_dists):
            if self.config.post_dist == 'beta':
                kl_loss = dist.kl_divergence(post_dist, prior_dist)
            elif self.config.post_dist == 'kuma':
                kl_loss = kldivergence_kuma(post_dist, prior_dist.concentration1, prior_dist.concentration0)
            else:
                raise NotImplementedError()
            kl_loss = torch.mean(torch.mean(kl_loss, dim=1))
            kl_loss_total = kl_loss_total + weights[idx] * kl_loss
        return kl_loss_total

    def aux_loss(self, mask):
        kl_loss = self.kl_divergence() if self.config.aux_loss_kl else None
        l_padded_mask = torch.cat([mask[:,0].unsqueeze(1), mask], dim=1)
        r_padded_mask = torch.cat([mask, mask[:,-1].unsqueeze(1)], dim=1)
        continuity_cost = torch.mean(torch.mean(torch.abs(l_padded_mask - r_padded_mask), dim=1)) if self.config.aux_loss_con else None
        aux_loss = {
            'kl_loss': kl_loss,
            'continuity_loss': continuity_cost,
        }
        return aux_loss

    def forward(
        self,
        seqs,
        signals,
        mask_regions=None,
        bio_mask=None,
        peak_mask=None,
        rna_feat=None,
    ):
        if self.config.merge_peak_mask:
            expanded_mask = peak_mask.unsqueeze(-1)
            signals = signals * expanded_mask

        self.includelist, self.blacklist = mask_regions[...,0], mask_regions[...,1]
        bs, seq_len, _ = seqs.shape
        seq_input_embeds = self.seq_input_layer(seqs)
        if self.config.pos_enc:
            # add positional embedding
            seq_input_embeds = seq_input_embeds + self.pos_emb
        inputs_embeds_enc = inputs_embeds = seq_input_embeds
        if self.config.interact == 'concat':
            
           
            signal_input_embeds = self.signal_input_layer(signals)
           
            inputs_embeds_enc = seq_input_embeds + signal_input_embeds 
           
            if self.config.gen_signal or self.config.enc_prd_ps: 
                inputs_embeds = seq_input_embeds + signal_input_embeds

        if self.config.use_bio_mask:
            mask = bio_mask[...,1]
            aux_infor = {
                'mask': mask,
            }
        elif self.config.use_peak_mask:
            mask = peak_mask
            aux_infor = {
                'mask': mask,
            }
        elif self.config.use_mask_model:
            mask = self.mask_model(seqs=seqs, signals=signals, mask_regions=mask_regions, bio_mask=bio_mask, peak_mask=peak_mask, rna_feat=rna_feat)
            top_num = int(self.config.top_mask_percent * seq_len)
            topk_indices = torch.topk(mask, top_num, dim=1).indices
            binary_mask = torch.zeros_like(mask)
            binary_mask.scatter_(1, topk_indices, 1)
            mask = binary_mask
            aux_infor = {
                'mask': mask,
            }
        else:
            outputs = self.generator(
                input_ids=None,
                inputs_embeds=inputs_embeds, # seqs only
                output_hidden_states=False,
                return_dict=False,
            )
            logits = self.mask_output(outputs) # 
           
            self.posterior_dist(logits) 
            self.prior_dist(signals, peak_mask=peak_mask)

            if self.config.decouple_x:
                self.couple_post_dist() 
                z_dist = self.z_distribution
            else:
                z_dist = self.post_distributions

            if self.config.post_sample: 
                print('training#,',self.training) 
                if not self.training:
                    soft_mask = z_dist.mean
                    if self.config.test_top:
                        top_num = int(seq_len * self.config.test_top_percent)
                        _, indices = torch.topk(soft_mask, top_num, dim=-1)
                        hard_mask = torch.zeros_like(soft_mask)
                        batch_indices = torch.arange(bs).unsqueeze(-1).expand_as(indices)
                        if self.config.test_top_soft:
                            hard_mask[batch_indices, indices] = soft_mask[batch_indices, indices]
                        else:
                            hard_mask[batch_indices, indices] = 1.0
                        mask = hard_mask
                    elif self.config.test_soft: 
                        mask = soft_mask
                    elif self.config.test_hard:
                        if self.config.pool_mask != 0:
                            soft_mask = smooth_max(soft_mask, self.config.pool_mask)
                        hard_mask = (soft_mask >= self.config.sample_threshold).float() 
                        ratio = hard_mask.mean()
                        mask = hard_mask
                        print('ratio#,',ratio) 
                else:
                    soft_mask = z_dist.rsample() 
                    if self.config.pool_mask != 0: 
                        soft_mask = smooth_max(soft_mask, self.config.pool_mask)
                    if self.config.post_hard_dist: 
                        hard_mask = (soft_mask >= self.config.sample_threshold).float() 
                        mask = hard_mask - soft_mask.detach() + soft_mask 
                    else:
                        mask = soft_mask.clone()
            else:
                mask = F.gumbel_softmax(logits, tau=self.config.gumbel_temp, hard=True, dim=-1)[:,:,1]
            # includelist and blacklist
            mask[self.blacklist] = 0.0
            if self.config.use_include_list and self.config.mask_region_hard: #false 
                mask[self.includelist] = 1.0

            start = (seq_len - self.promo_len) // 2
            mask[:, start:start + self.promo_len] = 1.0 # TSS 上下1k
            # get aux loss
            aux_infor = self.aux_loss(mask=mask)
            aux_infor['mask'] = mask

        if self.config.pos_enc:
            valid_counts = mask.sum(dim=1)
            max_valid_len = valid_counts.max()
            padded_inputs_embeds_enc = torch.full((bs, int(max_valid_len), self.config.d_model), 0,
                                                  dtype=inputs_embeds_enc.dtype, device=inputs_embeds_enc.device)
            range_tensor = torch.arange(int(max_valid_len), device=inputs_embeds_enc.device).unsqueeze(0)
            valid_positions = range_tensor < valid_counts.unsqueeze(1)
            padded_inputs_embeds_enc[valid_positions] = inputs_embeds_enc[mask == 1]
            inputs_embeds_enc = padded_inputs_embeds_enc
         
        else:
            inputs_embeds_enc = inputs_embeds_enc * mask.unsqueeze(-1)
        outputs_enc = self.encoder(
            input_ids=None,
            inputs_embeds=inputs_embeds_enc,
            output_hidden_states=False,
            return_dict=False,
        )

        # output
        hidden_states = outputs_enc
        if self.config.pos_enc and self.config.center_len:
            valid_counts_left = mask[:,:seq_len//2].sum(dim=1) - self.promo_len // 2
            promo_indices = valid_counts_left.unsqueeze(1) + torch.arange(self.promo_len,
                                                                          device=outputs_enc.device).unsqueeze(0)
            batch_indices = promo_indices.unsqueeze(-1).expand(-1, -1, outputs_enc.shape[-1]).to(torch.int64)
            hidden_states = outputs_enc.gather(1, batch_indices)

        elif self.config.center_len: 
            start_index = (hidden_states.shape[1] - self.config.center_len) // 2
            end_index = start_index + self.config.center_len
            hidden_states = hidden_states[:, start_index: end_index, :]
     
        hidden_states = torch.mean(hidden_states, dim=1)
    
        p_embed = torch.cat([hidden_states, rna_feat], dim=-1) if self.config.useRNAFeat else hidden_states
        logits = self.pToExpr(p_embed)

        logits = logits.float()

        return logits, aux_infor, mask,seq_input_embeds,signal_input_embeds    #pred_expr, aux_infor, mask