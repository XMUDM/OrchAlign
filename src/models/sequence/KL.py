import torch
import torch.nn.functional as F
import math
def compute_token_similarity(x):
    """
    计算 token 间相似度矩阵（归一化为概率分布）
    x: [B, L, D]
    return: [B, L, L] 每个 token 对其他 token 的分布
    """
    x_norm = F.normalize(x, p=2, dim=-1)
  
    sim = torch.matmul(x_norm, x_norm.transpose(1, 2))
    mask = torch.eye(x_norm.shape[1], device=sim.device).bool()
    sim = sim.masked_fill(mask, float('-inf'))
    sim_dist = F.softmax(sim, dim=-1)
    return sim_dist

def kl_token_alignment_loss_F(shared, specific, eps=1e-8):
    
    shared_dist = compute_token_similarity(shared).clamp(min=eps)  # P
    specific_dist = compute_token_similarity(specific).clamp(min=eps)  # Q

    kl2 = F.kl_div(specific_dist.log(), shared_dist, reduction='batchmean')  # KL(Q||P)
    kl_loss =  kl2
    return kl_loss

