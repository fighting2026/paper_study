import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FuzzyLayer3(nn.Module):
    def __init__(self, fuzzynum, channel, dropout=0.2):
        super().__init__()
        self.n = fuzzynum
        self.channel = channel
        self.mu = nn.Parameter(torch.randn((1, channel, 1, fuzzynum)))
        self.sigma = nn.Parameter(torch.randn((1, channel, 1, fuzzynum)))
        self.bn = nn.BatchNorm1d(channel, affine=True)
        self.ln = nn.LayerNorm(channel)
        self.dropout = nn.Dropout(dropout)
        # 动态门控
        self.gate_fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.Sigmoid()
        )
    def forward(self, x_before_t, x_after_t):
        # [B, L, C]
        assert x_before_t.shape == x_after_t.shape
        x = rearrange(x_after_t, 'b l c -> b c l')
        x_expanded = x.unsqueeze(-1)
        sigma_safe = F.softplus(self.sigma) + 1e-6
        tmp = -((x_expanded - self.mu) / sigma_safe) ** 2
        tmp = torch.logsumexp(tmp, dim=-1)
        f_out = torch.exp(tmp)
        fNeural = self.bn(f_out)
        fNeural = F.relu(fNeural)
        fNeural_rearr = rearrange(fNeural, 'b c l -> b l c')
        # 动态门控
        fusion_gate = self.gate_fc(torch.mean(x_before_t, 1))  # [B, C]
        fusion_gate = fusion_gate.unsqueeze(1)                 # [B,1,C]
        out = fusion_gate * fNeural_rearr + (1-fusion_gate) * x_before_t
        out = self.ln(out)
        out = self.dropout(out)
        return out