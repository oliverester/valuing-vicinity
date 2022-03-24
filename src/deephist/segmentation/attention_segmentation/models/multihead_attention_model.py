import math

import torch 
import torch.nn as nn
import torch.nn.functional as F

from src.deephist.segmentation.attention_segmentation.models.position_encoding import PositionalEncoding, _2DPositionalEmbedding

class MultiheadAttention(nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 num_heads, 
                 kernel_size, 
                 use_ln=False, 
                 use_pos_encoding=False,
                 learn_pos_encoding=False):
        """_summary_

        Args:
            input_dim (_type_): Dimension of patch embeddings
            hidden_dim (_type_): Hidden dimension of internal representation (over all heads).
            num_heads (_type_): Number of heads
            kernel_size (_type_): Neighbourhood length
            use_ln (bool, optional): Use layer normalization for k, q, v. Defaults to False.
            use_pos_encoding (bool, optional): Use sinusiodal position encoding. Defaults to False.
            learn_pos_encoding (bool, optional): Use learnable 2-d position embeddings. Defaults to False.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.use_ln = use_ln
        self.use_pos_encoding = use_pos_encoding
        self.learn_pos_encoding = learn_pos_encoding
        
        assert(use_pos_encoding + learn_pos_encoding <= 1), "Either use (sin.) position encoding or learn (2d) embeddings." 
        
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
            self._2d_pos_emb = _2DPositionalEmbedding(d_hid=hidden_dim // num_heads,
                                                      n_position=kernel_size)
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
               # 2D position learnable embedding
               k = self._2d_pos_emb(k)
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
        
        
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention