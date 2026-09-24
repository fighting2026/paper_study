import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Function, Variable

_eps = 1e-10

class LorentzManifold:
    def __init__(self, eps=1e-8, norm_clip=0.8, max_norm=5.0):
        self.eps = eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm

    @staticmethod
    def ldot(u, v, keepdim=False):
        d = u.size(1) - 1
        uv = u * v
        uv = torch.cat((-uv.narrow(1, 0, 1), uv.narrow(1, 1, d)), dim=1)
        return torch.sum(uv, dim=1, keepdim=keepdim)

    def from_poincare_to_lorentz(self, x):
        # 加数值稳定：防止除以0
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        # 限制范数不超过1，避免映射后溢出
        x_norm_sq = torch.clamp(x_norm_sq, max=1 - self.eps)
        denom = 1 - x_norm_sq + self.eps
        t = (1 + x_norm_sq) / denom
        x_t = (2 * x) / denom
        return torch.cat((t, x_t), dim=-1)

    def from_lorentz_to_poincare(self, x):
        t = x.narrow(-1, 0, 1)
        x_t = x.narrow(-1, 1, x.size(-1) - 1)
        return x_t / (t + self.eps)

    def normalize(self, w):
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm:
            narrowed = torch.renorm(narrowed.view(-1, d), 2, 0, self.max_norm)
        first = torch.sqrt(1 + torch.sum(narrowed ** 2, dim=-1, keepdim=True))
        return torch.cat((first, narrowed), dim=-1)

class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return LorentzManifold.ldot(u, v)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u

class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = torch.sqrt(torch.clamp(x * x - 1 + eps, _eps))
        ctx.save_for_backward(z)
        ctx.eps = eps
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = torch.clamp(z, min=ctx.eps)
        z = g / z
        return z, None