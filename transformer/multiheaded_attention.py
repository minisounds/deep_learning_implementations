import math
import torch
import torch.nn as nn

class MultiHeadedAttention(nn.Module): 
    def __init__(self, d_model=512, heads=8): 
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_head = self.d_model // self.heads 
        
        self.W_q = nn.Linear(self.d_model, self.d_head * self.heads) # (512, 512)
        self.W_k = nn.Linear(self.d_model, self.d_head * self.heads)
        self.W_v = nn.Linear(self.d_model, self.d_head * self.heads)
        self.W_o = nn.Linear(self.d_model, self.d_model)

    # Attention:
    def forward(self, X): 
        # Accepts tensor X (batch, seq_len, d_model) as input embeddings 
        batch, seq_len, _ = X.shape
        Q = self.W_q(X).view(batch, seq_len, self.heads, self.d_head).transpose(-2, -3) # (batch, seq_len, d_model) -> (batch, seq_len, heads, d_head) -> (batch, heads, seq_len, d_head)
        K = self.W_k(X).view(batch, seq_len, self.heads, self.d_head).transpose(-2, -3) # (batch, heads, seq_len, d_head)
        V = self.W_v(X).view(batch, seq_len, self.heads, self.d_head).transpose(-2, -3) # (batch, heads, seq_len, d_head)
        
        # Transpose K: 
        K_T = K.transpose(-1, -2) # (batch, heads, d_head, seq_len)
        
        # Compute Attention 
        compat_scores = torch.softmax(torch.matmul(Q, K_T) / (math.sqrt(self.d_head)), dim=-1) # (batch, heads, seq_len, seq_len)
        attn = torch.matmul(compat_scores, V).permute(0, 2, 1, 3).contiguous() #  (batch, heads, seq_len, d_head) -> (batch, seq_len, heads, d_head) 
        attn = attn.view(batch, seq_len, self.heads*self.d_head) # (batch, seq_len, heads, d_head) -> (batch, seq_len, heads*d_head)
        attn_out = self.W_o(attn) # (batch, seq_len, self.d_model)
        
        return attn_out