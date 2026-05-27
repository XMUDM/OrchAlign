from torch import nn

import src.models.nn.utils as U
import src.utils as utils


class Encoder(nn.Module):
    """Encoder abstraction

    Accepts a tensor and optional kwargs. Other than the main tensor, all other arguments should be kwargs.
    Returns a tensor and optional kwargs.
    Encoders are combined via U.PassthroughSequential which passes these kwargs through in a pipeline. The resulting
    kwargs are accumulated and passed into the model backbone.
    """

    def forward(self, x, **kwargs):
        """
        x: input tensor
        *args: additional info from the dataset (e.g. sequence lengths)

        Returns:
        y: output tensor
        *args: other arguments to pass into the model backbone
        """
        return x, {}


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)
class ConvBlock(nn.Module):
    def __init__(self, size, stride = 2, hidden_in = 64, hidden = 64):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out
    

class ConvBlockwithAttentionPool(nn.Module):
    def __init__(self, size, stride = 1, hidden_in = 64, hidden = 64):
        super(ConvBlockwithAttentionPool, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.att= AttentionPool(hidden, pool_size = 2)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        
        out = self.relu(res_out + identity)
        out=self.att(out)
        return out   

class ConvBlockwithMaxPool(nn.Module):
    def __init__(self, size, stride = 1, hidden_in = 64, hidden = 64):
        super(ConvBlockwithMaxPool, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.att= nn.MaxPool1d(kernel_size=2, stride=2) # (L-2)/2+1 
        #self.att= nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        
        out = self.relu(res_out + identity)
        out=self.att(out)
        return out   

class Encoder1(nn.Module):
    def __init__(self, in_channel, output_size = 256, filter_size = 5, num_blocks = 12):
        super(Encoder1, self).__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
                                    nn.Conv1d(in_channel, 32, 3, 2, 1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
 
 
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
 
# 改为6层卷积 # stride=2 625  stride=1 1250
class EncoderSplit(nn.Module):
    def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=1, num_blocks = 4):
        super(EncoderSplit, self).__init__()
        self.filter_size = filter_size
        
        # self.seq_gate_net = nn.Conv1d(4, 16, kernel_size=1)
        # self.seq_gate_net = nn.Conv1d(4, 16, kernel_size=1)
        
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(4, 16, 3, stride, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 64, 128,256]
        hidden_ins = [32, 32, 64, 128]
        
        # hiddens =        [32, 64,64,128,256]
        # hidden_ins = [32, 32, 64,64,128]
        
        # hiddens =        [32, 64, 128]
        # hidden_ins = [32, 32, 64]
        
        #6
        # hiddens =        [32, 64,64,128,256]
        # hidden_ins = [32, 32, 64,64,128]
        
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        
        #print('conv#,',self.conv_start_seq[0].weight.dtype)
        self.conv_start_epi = nn.Sequential(
                                        nn.Conv1d(signal_size, 16, 3, stride, 1),
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(),
                                        )
        #print('conv#,',self.conv_start_epi[0].weight.dtype) 
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(2, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        
        #self.para=nn.Linear(2,2)
        #self.conv_end = nn.Conv1d(256, output_size, 1)
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    # def forward(self, seq,epi):
    #     #print('x#,',x.shape) #torch.Size([64, 500, 5])
    #     seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
    #     epi=epi.transpose(1, 2).contiguous().float()
    #     seq = self.res_blocks_seq(self.conv_start_seq(seq))
    #     print('seq#,',seq.shape)
    #     epi = self.res_blocks_epi(self.conv_start_epi(epi))
    #     # x = torch.cat([seq, epi], dim = 1)
    #     print('epi#,',epi.shape)
    #     # out = self.conv_end(x)
    
    #     return seq,epi
    def forward(self, x):
        #print('x#,',x.shape) #torch.Size([64, 500, 5])
        seq=x[0]
        epi=x[1]
      
        seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
        epi=epi.transpose(1, 2).contiguous().float()
        # t1=self.conv_start_seq(seq)
        # t2=self.conv_start_epi(epi)
        # t1=t1.transpose(1, 2)
        # t2=t2.transpose(1, 2)
        # #seq#, torch.Size([8, 4, 20000])
        # #epi#, torch.Size([8, 2, 20000])
        # print('seq#,',t1.shape)
        # print('epi#,',t2.shape)
        # t1=t1*(1+self.weight_mlp(t2))
        # t1=t1.transpose(1, 2)
        # t2=t2.transpose(1, 2)
        # seq = self.res_blocks_seq(t1)
        # epi = self.res_blocks_epi(t2)
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        #print('seq#,',seq.shape)
        epi = self.res_blocks_epi(self.conv_start_epi(epi))
        # x = torch.cat([seq, epi], dim = 1)
        #print('epi#,',epi.shape)
        # out = self.conv_end(x)
        seq=seq.transpose(1, 2).contiguous()
        epi=epi.transpose(1, 2).contiguous()
        print('epi##,',epi.shape,seq.shape) #torch.Size([8, 40, 128]) torch.Size([8, 40, 128])
        out=[seq,epi]
        return out

class EncoderSplit3cov(nn.Module):
    def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=2, num_blocks = 3):
        super(EncoderSplit3cov, self).__init__()
        self.filter_size = filter_size
        
        # self.seq_gate_net = nn.Conv1d(4, 16, kernel_size=1)
        # self.seq_gate_net = nn.Conv1d(4, 16, kernel_size=1)
        
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(4, 32, 3, stride, 1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        hiddens =        [64, 128,256]
        hidden_ins = [64, 64, 128]
        
        # hiddens =        [32, 64,64,128,256]
        # hidden_ins = [32, 32, 64,64,128]
        
        # hiddens =        [32, 64,64,128,128,256]
        # hidden_ins = [32, 32, 64,64,128,128]
        
        # hiddens =        [32, 64, 128]
        # hidden_ins = [32, 32, 64]
        
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        
        #print('conv#,',self.conv_start_seq[0].weight.dtype)
        self.conv_start_epi = nn.Sequential(
                                        nn.Conv1d(signal_size, 32, 3, stride, 1),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(),
                                        )
        #print('conv#,',self.conv_start_epi[0].weight.dtype) 
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(2, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        
        #self.para=nn.Linear(2,2)
        #self.conv_end = nn.Conv1d(256, output_size, 1)
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    # def forward(self, seq,epi):
    #     #print('x#,',x.shape) #torch.Size([64, 500, 5])
    #     seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
    #     epi=epi.transpose(1, 2).contiguous().float()
    #     seq = self.res_blocks_seq(self.conv_start_seq(seq))
    #     print('seq#,',seq.shape)
    #     epi = self.res_blocks_epi(self.conv_start_epi(epi))
    #     # x = torch.cat([seq, epi], dim = 1)
    #     print('epi#,',epi.shape)
    #     # out = self.conv_end(x)
    
    #     return seq,epi
    def forward(self, x):
        #print('x#,',x.shape) #torch.Size([64, 500, 5])
        seq=x[0]
        epi=x[1]
      
        seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
        epi=epi.transpose(1, 2).contiguous().float()
        # t1=self.conv_start_seq(seq)
        # t2=self.conv_start_epi(epi)
        # t1=t1.transpose(1, 2)
        # t2=t2.transpose(1, 2)
        # #seq#, torch.Size([8, 4, 20000])
        # #epi#, torch.Size([8, 2, 20000])
        # print('seq#,',t1.shape)
        # print('epi#,',t2.shape)
        # t1=t1*(1+self.weight_mlp(t2))
        # t1=t1.transpose(1, 2)
        # t2=t2.transpose(1, 2)
        # seq = self.res_blocks_seq(t1)
        # epi = self.res_blocks_epi(t2)
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        #print('seq#,',seq.shape)
        epi = self.res_blocks_epi(self.conv_start_epi(epi))
        # x = torch.cat([seq, epi], dim = 1)
        #print('epi#,',epi.shape)
        # out = self.conv_end(x)
        seq=seq.transpose(1, 2).contiguous()
        epi=epi.transpose(1, 2).contiguous()
        print('epi##,',epi.shape,seq.shape) #torch.Size([8, 40, 128]) torch.Size([8, 40, 128])
        out=[seq,epi]
        return out

class EncoderSplit2cov(nn.Module):
    def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=2, num_blocks = 3):
        super(EncoderSplit2cov, self).__init__()
        self.filter_size = filter_size
        
     
        hiddens =        [128, 128,128]
        hidden_ins = [128, 128,128]
        #print('conv#,',self.conv_start_epi[0].weight.dtype) 
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        
       
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
   
    def forward(self, x):
        #print('x#,',x.shape) #torch.Size([64, 500, 5])
        seq=x[0]
        epi=x[1]
      
        seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
        epi=epi.transpose(1, 2).contiguous().float()
       
        seq = self.res_blocks_seq(seq)
        #print('seq#,',seq.shape)
        epi = self.res_blocks_epi(epi)
      
        seq=seq.transpose(1, 2).contiguous()
        epi=epi.transpose(1, 2).contiguous()
        print('epi##,',epi.shape,seq.shape) #torch.Size([8, 40, 128]) torch.Size([8, 40, 128])
        out=[seq,epi]
        return out

class EncoderSplitEpiMseq(nn.Module):
    def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=2, num_blocks = 4):
        super(EncoderSplitEpiMseq, self).__init__()
        self.filter_size = filter_size
        
        # self.seq_gate_net = nn.Conv1d(4, 16, kernel_size=1)
        # self.epi_gate_net = nn.Conv1d(2, 16, kernel_size=1)
        # 不压缩 1*1 
        self.seq_gate_net = nn.Sequential(
                                    nn.Conv1d(4, 16, kernel_size=1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        
        self.epi_gate_net = nn.Sequential(
                                    nn.Conv1d(2, 16, kernel_size=1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 64, 128,256]
        hidden_ins = [32, 32, 64, 128]
        
        # hiddens =        [32, 64,64,128,256]
        # hidden_ins = [32, 32, 64,64,128]
        
        # hiddens =        [32, 64, 128]
        # hidden_ins = [32, 32, 64]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        
        #print('conv#,',self.conv_start_seq[0].weight.dtype)
        self.conv_start_epi = nn.Sequential(
                                        nn.Conv1d(signal_size, 16, 3, stride, 1),
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(),
                                        )
        #print('conv#,',self.conv_start_epi[0].weight.dtype) 
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(2, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        self.weight_mlp = nn.Sequential(
            nn.Linear(16, 16),
            nn.Sigmoid()
        )
        
        #self.para=nn.Linear(2,2)
        #self.conv_end = nn.Conv1d(256, output_size, 1)
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    # def forward(self, seq,epi):
    #     #print('x#,',x.shape) #torch.Size([64, 500, 5])
    #     seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
    #     epi=epi.transpose(1, 2).contiguous().float()
    #     seq = self.res_blocks_seq(self.conv_start_seq(seq))
    #     print('seq#,',seq.shape)
    #     epi = self.res_blocks_epi(self.conv_start_epi(epi))
    #     # x = torch.cat([seq, epi], dim = 1)
    #     print('epi#,',epi.shape)
    #     # out = self.conv_end(x)
    
    #     return seq,epi
    def forward(self, x):
        #print('x#,',x.shape) #torch.Size([64, 500, 5])
        seq=x[0]
        epi=x[1]   
        seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
        epi=epi.transpose(1, 2).contiguous().float()
        
        seq=self.seq_gate_net(seq)
        epi=self.epi_gate_net(epi)
        seq=seq.transpose(1, 2)
        epi=epi.transpose(1, 2)
        seq=seq*(1+self.weight_mlp(epi))
        seq=seq.transpose(1, 2)
        epi=epi.transpose(1, 2)
        
      
        print('seq#,',seq.shape)
        print('epi#,',epi.shape)
     
        seq = self.res_blocks_seq(seq)
        epi = self.res_blocks_epi(epi)
        #seq = self.res_blocks_seq(self.conv_start_seq(seq))
        #print('seq#,',seq.shape)
        #epi = self.res_blocks_epi(self.conv_start_epi(epi))
        # x = torch.cat([seq, epi], dim = 1)
        #print('epi#,',epi.shape)
        # out = self.conv_end(x)
        seq=seq.transpose(1, 2).contiguous()
        epi=epi.transpose(1, 2).contiguous()
        print('epi##,',epi.shape,seq.shape) #torch.Size([8, 40, 128]) torch.Size([8, 40, 128])
        out=[seq,epi]
        return out
class EncoderSplitMoreEpi(nn.Module):
    def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=2, num_blocks = 6):
        super(EncoderSplitMoreEpi, self).__init__()
        self.filter_size = filter_size
        self.conv_start_epi = nn.Sequential(
                                        nn.Conv1d(signal_size, 32, 3, stride, 1),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(),
                                        )
            
        hiddens =        [32, 64,64,128,128,256]
        hidden_ins = [32, 32, 64,64,128,128]

        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(2, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        
        #self.para=nn.Linear(2,2)
        #self.conv_end = nn.Conv1d(256, output_size, 1)
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    def forward(self, x):
        #print('x#,',x.shape) #torch.Size([64, 500, 5])
        seq=x[0]
        epi=x[1]
        epi=epi.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
        epi = self.res_blocks_epi(self.conv_start_epi(epi))
        epi=epi.transpose(1, 2).contiguous()
       
        #print('epi##,',epi.shape,seq.shape) #torch.Size([8, 40, 128]) torch.Size([8, 40, 128])
        out=[seq,epi]
        return out
class EncoderSplitMoreDNA(nn.Module):
    def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=2, num_blocks = 6):
        super(EncoderSplitMoreDNA, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(4, 32, 3, stride, 1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        # hiddens =        [32, 64, 128,256]
        # hidden_ins = [32, 32, 64, 128]
        
        hiddens =        [32, 64,64,128,128,256]
        hidden_ins = [32, 32, 64,64,128,128]
        
        # hiddens =        [32, 64, 128]
        # hidden_ins = [32, 32, 64]
        #hiddens_half = (np.array(hiddens) / 2).astype(int)
        #hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        
       
        
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(2, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        
        #self.para=nn.Linear(2,2)
        #self.conv_end = nn.Conv1d(256, output_size, 1)
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    def forward(self, x):
        #print('x#,',x.shape) #torch.Size([64, 500, 5])
        seq=x[0]
        epi=x[1]
        seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        seq=seq.transpose(1, 2).contiguous()
       
        #print('epi##,',epi.shape,seq.shape) #torch.Size([8, 40, 128]) torch.Size([8, 40, 128])
        out=[seq,epi]
        return out
    

class EncoderSplitMore(nn.Module):
    def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=2, num_blocks = 6):
        super(EncoderSplitMore, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(4, 16, 3, stride, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        # hiddens =        [32, 64, 128,256]
        # hidden_ins = [32, 32, 64, 128]
        
        hiddens =        [32, 64,64,128,128,256]
        hidden_ins = [32, 32, 64,64,128,128]
        
        # hiddens =        [32, 64, 128]
        # hidden_ins = [32, 32, 64]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        
        print('conv#,',self.conv_start_seq[0].weight.dtype)
        self.conv_start_epi = nn.Sequential(
                                        nn.Conv1d(signal_size, 16, 3, stride, 1),
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(),
                                        )
        print('conv#,',self.conv_start_epi[0].weight.dtype) 
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(2, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        
        #self.para=nn.Linear(2,2)
        #self.conv_end = nn.Conv1d(256, output_size, 1)
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    def forward(self, x):
        #print('x#,',x.shape) #torch.Size([64, 500, 5])
        seq=x[0]
        epi=x[1]
        print('seq#,',seq.shape) #torch.Size([8, 20000, 4])
        seqvar = seq.var(dim=(0,1)).mean()  # 全局方差
        seqkurt = ((seq - seq.mean())**4).mean() / (seqvar**2)
        epivar = epi.var(dim=(0,1)).mean()  # 全局方差
        epikurt = ((epi - epi.mean())**4).mean() / (epivar**2)
        
        #print('rr#,',self.conv_start_seq(seq).shape)
  
        
        print('one-hot seq noise:', seqvar, seqkurt,'epi noise:', epivar, epikurt)
     
      
        # print('seq#,',seq.shape)
        # print('epi#,',epi.shape)
        seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
        epi=epi.transpose(1, 2).contiguous().float()
     
        seq1=self.conv_start_seq(seq).transpose(1, 2)
        epi1=self.conv_start_epi(epi).transpose(1, 2)
        seq1var = seq1.var(dim=(0,1)).mean()  # 全局方差
        seq1kurt = ((seq1 - seq1.mean())**4).mean() / (seq1var**2)
        epi1var = epi1.var(dim=(0,1)).mean()  # 全局方差
        epi1kurt = ((epi1 - epi1.mean())**4).mean() / (epi1var**2)
        print('covstart seq noise:', seq1var, seq1kurt,'epi noise:', epi1var, epi1kurt)
        #print('epi noise:', var, kurt,'seq noise:', var1, kurt1)
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        epi = self.res_blocks_epi(self.conv_start_epi(epi))

        seq=seq.transpose(1, 2).contiguous()
        epi=epi.transpose(1, 2).contiguous()
        print('epi##,',epi.shape,seq.shape) #torch.Size([8, 40, 128]) torch.Size([8, 40, 128])
        out=[seq,epi]
        return out
class EncoderSplitMoreDim(nn.Module):
    def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=2, num_blocks = 6):
        super(EncoderSplitMoreDim, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(4, 32, 5, stride, 2),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        # hiddens =        [32, 64, 128,128]
        # hidden_ins = [32, 32, 64, 128]
        
        hiddens =        [64, 64,64,128,128,128]
        hidden_ins = [32, 64, 64,64,128,128]
        
        # hiddens =        [32, 64, 128]
        # hidden_ins = [32, 32, 64]
        # hiddens_half = (np.array(hiddens) / 2).astype(int)
        # hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        
        print('conv#,',self.conv_start_seq[0].weight.dtype)
        self.conv_start_epi = nn.Sequential(
                                        nn.Conv1d(signal_size, 32, 5, stride, 2),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(),
                                        )
        print('conv#,',self.conv_start_epi[0].weight.dtype) 
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(2, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        # self.weight_mlp = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.Sigmoid()
        # )
        
        #self.para=nn.Linear(2,2)
        #self.conv_end = nn.Conv1d(256, output_size, 1)
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    def forward(self, x):
        #print('x#,',x.shape) #torch.Size([64, 500, 5])
        seq=x[0]
        epi=x[1]
        print('seq#,',seq.shape) #torch.Size([8, 20000, 4])
        seqvar = seq.var(dim=(0,1)).mean()  # 全局方差
        seqkurt = ((seq - seq.mean())**4).mean() / (seqvar**2)
        epivar = epi.var(dim=(0,1)).mean()  # 全局方差
        epikurt = ((epi - epi.mean())**4).mean() / (epivar**2)
        
        #print('rr#,',self.conv_start_seq(seq).shape)
  
        
        print('one-hot seq noise:', seqvar, seqkurt,'epi noise:', epivar, epikurt)
     
      
        # print('seq#,',seq.shape)
        # print('epi#,',epi.shape)
        seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
        epi=epi.transpose(1, 2).contiguous().float()
     
        seq1=self.conv_start_seq(seq).transpose(1, 2)
        epi1=self.conv_start_epi(epi).transpose(1, 2)
        seq1var = seq1.var(dim=(0,1)).mean()  # 全局方差
        seq1kurt = ((seq1 - seq1.mean())**4).mean() / (seq1var**2)
        epi1var = epi1.var(dim=(0,1)).mean()  # 全局方差
        epi1kurt = ((epi1 - epi1.mean())**4).mean() / (epi1var**2)
        print('covstart seq noise:', seq1var, seq1kurt,'epi noise:', epi1var, epi1kurt)
        #print('epi noise:', var, kurt,'seq noise:', var1, kurt1)
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        epi = self.res_blocks_epi(self.conv_start_epi(epi))

        seq=seq.transpose(1, 2).contiguous()
        epi=epi.transpose(1, 2).contiguous()
        print('epi##,',epi.shape,seq.shape) #torch.Size([8, 40, 128]) torch.Size([8, 40, 128])
        out=[seq,epi]
        return out



# class EncoderSplit(nn.Module):
#     def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=1, num_blocks = 4):
#         super(EncoderSplit, self).__init__()
#         self.filter_size = filter_size
#         self.conv_start_seq = nn.Sequential(
#                                     nn.Conv1d(4, 16, 3, stride, 1),
#                                     nn.BatchNorm1d(16),
#                                     nn.ReLU(),
#                                     )
#         hiddens =        [32, 64, 128,256]
#         hidden_ins = [32, 32, 64, 128]
        
#         # hiddens =        [32, 64, 128]
#         # hidden_ins = [32, 32, 64]
#         hiddens_half = (np.array(hiddens) / 2).astype(int)
#         hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        
#         print('conv#,',self.conv_start_seq[0].weight.dtype)
#         # self.epigenetic_conv_layers = nn.Sequential(
#         #     # 第一层: 处理连续的信号值
#         #     # nn.Conv1d(signal_size, 16, kernel_size=51, padding=25),
#         #     # nn.ReLU(),
#         #     # nn.AvgPool1d(kernel_size=10, stride=10),  # 平均池化平滑信号
#         #     nn.Conv1d(signal_size, 16, 5, stride, 1),
#         #     nn.BatchNorm1d(16),
#         #     nn.ReLU(),
#         #     nn.AvgPool1d(kernel_size=2, stride=2),  # 平均池化平滑信号                           
#         #     # 第二层: 捕捉局部模式
#         #     nn.Conv1d(16, 32, kernel_size=25, padding=12),
#         #     nn.BatchNorm1d(32),
#         #     nn.ReLU(),
#         #     nn.AvgPool1d(kernel_size=2, stride=2),  # 进一步下采样
            
#         #     # 第三层: 精细特征
#         #     nn.Conv1d(conv_filters*2, conv_filters*4, kernel_size=11, padding=5),
#         #     nn.ReLU(),
#         #     nn.AdaptiveAvgPool1d(25)  # 统一长度便于融合
#         # )
#         print('conv#,',self.conv_start_epi[0].weight.dtype) 
#         self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
#         self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
#         #self.para=nn.Linear(2,2)
#         #self.conv_end = nn.Conv1d(256, output_size, 1)
#     def get_res_blocks(self, n, his, hs):
#         blocks = []
#         for i, h, hi in zip(range(n), hs, his):
#             blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
#         res_blocks = nn.Sequential(*blocks)
#         return res_blocks 
 
# 改为6层卷积
class EncoderSplitPool(nn.Module):
    def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=1, num_blocks = 4):
        super(EncoderSplitPool, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(4, 16, 15, stride, 7),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 64, 128,256]
        hidden_ins = [32, 32, 64, 128]
        
        # hiddens =        [32, 64, 128]
        # hidden_ins = [32, 32, 64]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        
        print('conv#,',self.conv_start_seq[0].weight.dtype)
        self.conv_start_epi = nn.Sequential(
                                        nn.Conv1d(signal_size, 16, 15, stride, 7),
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(),
                                        )
        print('conv#,',self.conv_start_epi[0].weight.dtype) 
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        #self.para=nn.Linear(2,2)
        #self.conv_end = nn.Conv1d(256, output_size, 1)
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlockwithAttentionPool(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    # def forward(self, seq,epi):
    #     #print('x#,',x.shape) #torch.Size([64, 500, 5])
    #     seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
    #     epi=epi.transpose(1, 2).contiguous().float()
    #     seq = self.res_blocks_seq(self.conv_start_seq(seq))
    #     print('seq#,',seq.shape)
    #     epi = self.res_blocks_epi(self.conv_start_epi(epi))
    #     # x = torch.cat([seq, epi], dim = 1)
    #     print('epi#,',epi.shape)
    #     # out = self.conv_end(x)
    
    #     return seq,epi
    def forward(self, x):
        #print('x#,',x.shape) #torch.Size([64, 500, 5])
        seq=x[0]
        epi=x[1]
        print('seq#,',seq.shape)
        print('epi#,',epi.shape)
        seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
        epi=epi.transpose(1, 2).contiguous().float()
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        #print('seq#,',seq.shape)
        epi = self.res_blocks_epi(self.conv_start_epi(epi))
        # x = torch.cat([seq, epi], dim = 1)
        #print('epi#,',epi.shape)
        # out = self.conv_end(x)
        seq=seq.transpose(1, 2).contiguous()
        epi=epi.transpose(1, 2).contiguous()
        print('epi##,',epi.shape,seq.shape) #torch.Size([8, 40, 128]) torch.Size([8, 40, 128])
        out=[seq,epi]
        return out
class EncoderSplitMaxPool(nn.Module):
    def __init__(self, signal_size,output_size = 256, filter_size = 5, stride=1, num_blocks = 6):
        super(EncoderSplitMaxPool, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(4, 16, 3, stride, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 64,64,128,128,256]
        hidden_ins = [32, 32, 64,64,128,128]
        
        # hiddens =        [32, 64, 128]
        # hidden_ins = [32, 32, 64]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        
        print('conv#,',self.conv_start_seq[0].weight.dtype)
        self.conv_start_epi = nn.Sequential(
                                        nn.Conv1d(signal_size, 16, 3, stride, 1),
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(),
                                        )
        print('conv#,',self.conv_start_epi[0].weight.dtype) 
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        #self.para=nn.Linear(2,2)
        #self.conv_end = nn.Conv1d(256, output_size, 1)
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlockwithMaxPool(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    # def forward(self, seq,epi):
    #     #print('x#,',x.shape) #torch.Size([64, 500, 5])
    #     seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
    #     epi=epi.transpose(1, 2).contiguous().float()
    #     seq = self.res_blocks_seq(self.conv_start_seq(seq))
    #     print('seq#,',seq.shape)
    #     epi = self.res_blocks_epi(self.conv_start_epi(epi))
    #     # x = torch.cat([seq, epi], dim = 1)
    #     print('epi#,',epi.shape)
    #     # out = self.conv_end(x)
    
    #     return seq,epi
    def forward(self, x):
        #print('x#,',x.shape) #torch.Size([64, 500, 5])
        seq=x[0]
        epi=x[1]
        print('seq#,',seq.shape)
        print('epi#,',epi.shape)
        seq=seq.transpose(1, 2).contiguous().float() # [batch_size, seq_len, in_channels] ->[batch_size, in_channels, seq_len]
        epi=epi.transpose(1, 2).contiguous().float()
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        #print('seq#,',seq.shape)
        epi = self.res_blocks_epi(self.conv_start_epi(epi))
        # x = torch.cat([seq, epi], dim = 1)
        #print('epi#,',epi.shape)
        # out = self.conv_end(x)
        seq=seq.transpose(1, 2).contiguous()
        epi=epi.transpose(1, 2).contiguous()
        print('epi##,',epi.shape,seq.shape) #torch.Size([8, 40, 128]) torch.Size([8, 40, 128])
        out=[seq,epi]
        return out
registry = {
    "stop": Encoder,
    "id": nn.Identity,
    "embedding": nn.Embedding,
    "linear": nn.Linear,
    'genome': EncoderSplitMore #EncoderSplit EncoderSplitMaxPool EncoderSplitPool EncoderSplitMore EncoderSplitEpiMseq EncoderSplit3cov EncoderSplitMore
    
    
}

dataset_attrs = {
    "embedding": ["n_tokens"],
    "linear": ["d_input"],  # TODO make this d_data?
    "class": ["n_classes"],
    "time": ["n_tokens_time"],
    "onehot": ["n_tokens"],
    "conv1d": ["d_input"],
    "patch2d": ["d_input"],
    
}

model_attrs = {
    "embedding": ["d_model"],
    "linear": ["d_model"],
    "position": ["d_model"],
    "class": ["d_model"],
    "time": ["d_model"],
    "onehot": ["d_model"],
    "conv1d": ["d_model"],
    "patch2d": ["d_model"],
    "timestamp_embedding": ["d_model"],
    "layer": ["d_model"],
    "genome": ["n_epi"]
}


def _instantiate(encoder, dataset=None, model=None):
    """Instantiate a single encoder"""
    if encoder is None:
        return None
    if isinstance(encoder, str):
        name = encoder
    else:
        name = encoder["_name_"]

    # Extract dataset/model arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    
    #print('model#,',model.signal_size)
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
    print('args#,',name,model_attrs,model_attrs.get(name, []),model_args) #[]
    # Instantiate encoder
    obj = utils.instantiate(registry, encoder, *dataset_args, *model_args)
    return obj


def instantiate(encoder, dataset=None, model=None):
    encoder = utils.to_list(encoder)
    return U.PassthroughSequential(
        *[_instantiate(e, dataset=dataset, model=model) for e in encoder]
    )
