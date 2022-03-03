import math
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 num_heads, 
                 kernel_size, 
                 use_ln=False, 
                 use_pos_encoding=False,
                 learn_pos_encoding=False):
        super().__init__()
        assert hidden_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.use_ln = use_ln
        self.use_pos_encoding = use_pos_encoding
        self.learn_pos_encoding = learn_pos_encoding
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.kv_proj = nn.Linear(input_dim, 2*hidden_dim, )
        self.q_proj = nn.Linear(input_dim, 1*hidden_dim, )
        self.o_proj = nn.Linear(hidden_dim, input_dim)
        self.l_norm = nn.LayerNorm(input_dim) # applies layer on last dim
        
        # Positional encodings
        self.kernel_size = kernel_size
        # relative pos embedding for each spatial dim of kernel
    
        if self.learn_pos_encoding:
            # 2D position encoding: patches share a col and row pos embedding, that are concatenated
            self.pos_h = nn.Parameter(torch.randn(1, 1, 1, kernel_size, hidden_dim // 2  // num_heads), requires_grad=True) 
            self.pos_w = nn.Parameter(torch.randn(1, 1, kernel_size, 1, hidden_dim // 2  // num_heads), requires_grad=True)
        else: 
            self.pos_enc = PositionalEncoding(d_hid=hidden_dim // num_heads,
                                              n_position=kernel_size*kernel_size)
            
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
        if self.learn_pos_encoding:
            nn.init.normal_(self.pos_h, 0, 1)
            nn.init.normal_(self.pos_w, 0, 1)
        
    def forward(self, q, kv, mask=None, return_attention=False):
        batch_size, seq_length, _ = kv.size()

        if self.use_ln:
            kv = self.l_norm(kv)
            q = self.l_norm(q)
        
        kv = self.kv_proj(kv)
        q = self.q_proj(q)
        hidden_dim = q.shape[-1]

        # Separate Q, K, V from linear output
        kv = kv.reshape(batch_size, seq_length, self.num_heads, 2*self.head_dim)
        kv = kv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        k, v = kv.chunk(2, dim=-1)
        
        q = q.reshape(batch_size, 1, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        
        if self.use_pos_encoding:
            if self.learn_pos_encoding:
                # 2D position learnable encoding
            
                # apply head-invariant relative position encoding
                k_h, k_w = k.split(hidden_dim // self.num_heads // 2, dim=-1)
                # break up kernel to apply spatial-wise pos encoding addition
                k_h = k_h.view((batch_size, self.num_heads , self.kernel_size, self.kernel_size, hidden_dim // self.num_heads // 2))
                k_w = k_w.view((batch_size, self.num_heads , self.kernel_size, self.kernel_size, hidden_dim // self.num_heads // 2))
                # add relative position encoding
                k = torch.cat((k_h + self.pos_h, k_w + self.pos_w), dim=-1)
                k = k.view((batch_size, self.num_heads , seq_length, hidden_dim // self.num_heads))
            else:
                # 1D siusoid positional encoding
                k = self.pos_enc(k)
            
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, 1, hidden_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
        
class PositionalEncoding(nn.Module):
    """
    From https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
    """

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table.clone().detach()

def position_encoding_init(n_position, d_hid):
    ''' Init the sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.from_numpy(sinusoid_table).type(torch.FloatTensor)

        
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention