# ==================== videomamba.py 模型文件 ====================
# 这是 Vision Mamba 模型的实现代码
# Vision Mamba 是一种结合了 Mamba 状态空间模型和 Transformer 的深度学习模型
# 用于处理视频/图像的高光谱数据分类

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# 版权声明，Facebook拥有此代码版权

# ==================== 第一部分：导入各种工具库 ====================
#导入双曲
from shuangqu import LorentzManifold
#导入模糊
from fuzzy import FuzzyLayer3

# 导入os模块，用于操作文件和目录
import os

# 导入PyTorch深度学习框架
import torch

# 导入torch.nn模块，里面有各种神经网络层（卷积层、全连接层、归一化层等）
import torch.nn as nn

# 从functools导入partial，用于创建带有默认参数的函数
from functools import partial

# 从torch导入Tensor类型，表示PyTorch的张量（多维数组）
from torch import Tensor

# 导入Optional类型，用于类型提示
from typing import Optional

# 导入torch.utils.checkpoint，用于减少显存占用（检查点机制）
import torch.utils.checkpoint as checkpoint

# 从einops导入rearrange和repeat，用于灵活地重排张量形状
# einops是一个操作张量的库，让代码更易读
from einops import rearrange, repeat

# 再次导入rearrange（重复导入，可能是为了保险）
from einops import rearrange

# 从timm库导入_cfg和register_model：
# - _cfg: 模型的默认配置
# - register_model: 注册模型的装饰器
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model

# 从timm库导入trunc_normal_，用于初始化权重（截断正态分布）
from timm.models.layers import trunc_normal_

# 从timm库导入DropPath和to_2tuple：
# - DropPath: 随机丢弃路径的正则化技术
# - to_2tuple: 把数字转成2元组
from timm.models.layers import DropPath, to_2tuple

# 从timm库导入_load_weights，用于加载预训练权重
from timm.models.vision_transformer import _load_weights

# 导入math模块，用于数学运算
import math

# 导入torch_dct模块，用于DCT变换（离散余弦变换）
import torch_dct as dct

# 再次导入torch（重复导入）
import torch

# 导入matplotlib.pyplot，用于画图（可视化）
import matplotlib.pyplot as plt

# ==================== 第二部分：导入Mamba相关模块 ====================

# 从mamba_ssm模块导入Mamba类
# mamba_ssm是一种状态空间模型（State Space Model）
# Mamba是核心的状态空间层，用于处理长序列数据
from mamba_ssm.modules.mamba_simple import Mamba

# 从models.csms6s导入三个选择性扫描Mamba变体：
# - SelectiveScanMamba: 选择性扫描Mamba
# - SelectiveScanCore: 核心版本
# - SelectiveScanOflex: 灵活版本
from models.csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex

# ==================== 尝试导入RMSNorm（层归一化）====================

# 尝试从mamba_ssm库导入RMSNorm（均方根归一化）
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# 如果导入失败（triton没安装），就把这三个变量设为None
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# ==================== 第三部分：模型路径配置 ====================

# 定义模型路径（需要改成你实际的模型权重路径）
MODEL_PATH = '自己实际的权重路径'

# _MODELS字典：预定义了一些预训练模型的路径
# 格式：{模型名: 权重文件路径}
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),  # 小型模型
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),  # 小型模型
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),  # 中型模型
}


# ==================== 第四部分：mamba_init 类 ====================

# mamba_init 类：包含各种初始化Mamba模型参数的方法
# 类似于"工厂"，生产各种Mamba需要的"零部件"
class mamba_init:
    """
    Mamba模型参数初始化器
    包含多个静态方法，用来初始化不同类型的参数
    """

    # 方法1：dt_proj 初始化 - 初始化时间步投影矩阵
    #初始化了dt_proj，的权重和偏置值
    #静态方法可以直接调用，不需要有self
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        """
        初始化时间步（delta）投影层
        输入：
            dt_rank: 时间步投影的秩（维度）：这个线性层的输入大小
            d_inner: 内部维度               这个线性层的输出大小
            dt_scale: 缩放因子              缩放系数，用来调节初始化范围
            dt_init: 初始化方式（"constant"或"random"）
            dt_min/dt_max: 时间的最小/最大值    初始化值的最小/最大值
            dt_init_floor: 初始化下界       防止梯度消失的下限值
        """
        # 创建一个线性层：从dt_rank维度映射到d_inner维度
        # 这个线性层的作用是把"时间步"投影到"内部空间"
        #**factory_kwargs，把额外参数（如 device、dtype）传给层
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        #权重初始化标准差：让神经网络初始化时既不爆炸也不消失，训练更稳
        #dt_rank ** -0.5等价于1 / sqrt(dt_rank)
        # dt_rank ** -0.5 是为了保持方差（让梯度更稳定）
        #eg:dt_rank=16,dt_scale=1.0  sqrt(16)=4,1/4=0.25,0.25*1.0=0.25,所以权重会从均值0、标准差0.25的正态分布里采样
        dt_init_std = dt_rank ** -0.5 * dt_scale

        # 根据dt_init选择初始化方式
        if dt_init == "constant":
            # 常数初始化：用dt_init_std填充权重
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            # 随机初始化：在[-dt_init_std, dt_init_std]范围内均匀分布
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            # 如果不是这两种方式，就报错
            raise NotImplementedError

        # 初始化dt bias（偏置）
        # 生成随机时间步，然后在dt_min和dt_max之间
        dt = torch.exp(# exp:自然指数e的多少次方，保证算出来的结果一定是正数
            torch.rand(d_inner, **factory_kwargs)  #生成d_inner个随机数，每个数都在0-1之间
            *(math.log(dt_max) - math.log(dt_min))+ math.log(dt_min) #把 0~1 的随机数 → 拉伸到 log(dt_min) ~ log(dt_max) 这个区间。
        ).clamp(min=dt_init_floor)#如果算出来的数 太小了（比如接近 0），强行把它拉到 dt_init_floor，防止 dt 变成 0，导致模型训练崩溃！

        #-torch.expm1(-dt)=-(e⁻ᵈᵗ - 1)= 1 - e⁻ᵈᵗ  
        #torch.log(...)= 取自然对数 ln
        #inv_dt = dt + ln(1 - e^(-dt))
        #作用：把dt转成模型能稳定训练的参数
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        # 把 inv_dt 的值，直接复制放进 dt_proj 这个线性层的偏置 b 里
        #告诉接下来这段代码只是手动赋值，不是模型自己学习的，不要算梯度
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        # 返回初始化好的dt_proj层
        return dt_proj


    # 方法2：A_log_init 初始化 - 初始化状态空间模型的A矩阵，专门用来做状态衰减
    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):#创建一组 “不会乱变、稳定衰减” 的模型状态矩阵 A，并且存成对数形式 A_log，让训练超级稳。
        """
        初始化A矩阵的对数（用于状态空间模型）
        输入：
            d_state: 模型状态维度
            d_inner: 内部维度，模型内部通道数
            copies: 复制次数，需要复制几份
            device: 设备（CPU或GPU）
            merge: 复制后要不要压平
        """

        # 创建一个从1到d_state的序列：[1, 2, 3, ..., d_state]
        #torch.arange(1, d_state+1)→ 生成 [1, 2, 3, ..., d_state]
        #repeat(..., "n -> d n", d=d_inner)→ 把这一行复制 d_inner 次！变成形状（d_inner,d_state）
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",  # 把n复制d份
            d=d_inner,    # d是d_inner
        ).contiguous()   # 变成连续内存,跑的更快

        # 对A取对数（保持float32精度），因为真正用的时候，A = -exp(A_log)，这样就能保证。A永远是负数，A永远稳定不爆炸，训练永远不会梯度消失
        A_log = torch.log(A)

        # 如果需要复制
        if copies > 0:
            # 把A_log复制copies份
            A_log = repeat(A_log, "d n -> r d n", r=copies)#变成(copies,d_inner,d_state)
            if merge:
                # 把复制的维度展平
                A_log = A_log.flatten(0, 1)#压平(copies*d_inner, d_state)

        # 转换成可学习的参数（PyTorch的Parameter），告诉模型参数，要跟着一起更新
        A_log = nn.Parameter(A_log)

        # 设置这个参数不参与权重衰减（防止被正则化）
        A_log._no_weight_decay = True

        return A_log


    # 方法3：D_init 初始化 - 初始化D矩阵（跳跃连接参数）
    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        """
        初始化D矩阵（"跳跃"参数，用于残差连接）
        输入：
            d_inner: 内部维度
            copies: 复制次数
            device: 设备
            merge: 是否合并
        """
        # D是一个"跳跃"参数，初始化为全1
        D = torch.ones(d_inner, device=device)

        # 如果需要复制
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)

        # 转换成可学习的参数（保持float32）
        D = nn.Parameter(D)

        # 设置不参与权重衰减
        D._no_weight_decay = True

        return D

# ==================== 第五部分：基础模块类 ====================

# -------------------- 类1：Residual（残差连接）--------------------
class Residual(nn.Module):
    """
    残差连接模块
    作用：把上一层的输出和这一层的输出相加，形成"残差连接"
    
    比喻：就像做菜时尝味道，先尝一口原材料（输入x），再尝尝做好的菜（fn(x)），
    然后把两个味道混在一起看看。这样梯度更容易传回去，不容易"忘"掉前面的东西
    
    公式：输出 = fn(x) + x

    它是深度网络不会崩、能训练很深的核心原因！
    作用（超级重要）：
    让梯度可以直接跳层传递，不会消失
        梯度不会因为网络太深而没了
    保留原始信息，不会被洗掉
        让模型 “记住原来的输入，再加上新学到的东西”
    让模型可以做几百上千层
        没有它，深层网络根本训练不动！
    """
    
    def __init__(self, fn):
        """
        初始化函数
        输入：fn - 一个神经网络层（要包裹起来的"黑盒"）
        """
        # 调用父类的初始化
        super().__init__()
        
        # 把传入的神经网络层保存起来
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        前向传播
        输入：x - 输入数据
        返回：fn(x) + x （输入+输出的和）
        """
        # 调用fn处理x，然后把结果和原始x相加
        return self.fn(x, **kwargs) + x


# -------------------- 类2：LayerNormalize（层归一化）--------------------
class LayerNormalize(nn.Module):
    """
    层归一化模块
    作用：先把数据做归一化（让它变得更"规矩"），然后再传给下一层
    
    比喻：就像把一团乱糟糟的面条整理整齐，然后再进行下一步处理
    这样可以让训练更稳定，不会出现某些数值特别大或特别小的情况
    """
    
    #创建一个“先归一化、再处理”的包装器，让神经网络训练更稳定
    def __init__(self, dim, fn):
        """
        初始化函数
        输入：
            dim - 数据的维度
            fn - 要包裹的神经网络层
        """
        super().__init__()
        
        # 创建LayerNorm层（层归一化）
        # 它会对数据进行归一化，让数据的均值变成0，方差变成1
        self.norm = nn.LayerNorm(dim)
        
        # 保存要处理的神经网络层
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        前向传播：先归一化，再传给fn处理
        """
        # 先对x做归一化，然后用fn处理
        return self.fn(self.norm(x), **kwargs)


# ==================== 第六部分：Attention（注意力机制）====================

class Attention(nn.Module):
    """
    自注意力机制模块
    作用：让模型学会"关注"重要的信息
    
    比喻：就像我们看一张图时，会自动关注重要的部分，忽略不重要的背景
    这个模块让神经网络也能做到这一点
    
    具体实现：
    - 把输入分成多个"头"（head），每个头学习不同的特征
    - 计算Query（查询）、Key（键）、Value（值）三个矩阵
    - 通过Q和K的相似度来决定V的权重
    """
    
    def __init__(self, dim, heads=8, dropout=0.1):
        """
        初始化函数
        输入：
            dim - 输入数据的维度
            heads - 注意力头的数量（默认8个）
            dropout - Dropout比例（防止过拟合）
        """
        super().__init__()
        
        # 保存注意力头数量
        self.heads = heads
        
        # 计算缩放因子（用于缩放注意力分数，防止数值过大）
        # 1/sqrt(dim)，这是Transformer论文中的标准做法
        self.scale = dim ** -0.5

        # to_qkv: 用来同时生成Q、K、V三个矩阵
        # 输入dim维，输出dim*3维（因为Q、K、V各一份）
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        
        # to_kv: 另一个生成K、V的线性层（用于特殊模式）
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)

        # nn1: 注意力输出后的线性变换层
        self.nn1 = nn.Linear(dim, dim)
        
        # do1: Dropout层，随机丢弃一些神经元
        self.do1 = nn.Dropout(dropout)
        
        # sr: 下采样卷积层（用于空间缩减）
        # kernel_size=2, stride=2 相当于把图像缩小一半
        self.sr = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        
        # norm: 层归一化
        self.norm = nn.LayerNorm(dim)
        
        # act: GELU激活函数（比ReLU更平滑）
        self.act = nn.GELU()

    def forward(self, x, mask=None, use_SR=False):
        """
        前向传播：计算注意力
        输入：
            x - 输入数据，形状 (batch, num_patches, dim)
            mask - 可选的注意力掩码
            use_SR - 是否使用空间缩减
        返回：注意力处理后的结果
        """
        # 解包数据：b=batch数量，n=token数量，d=维度，h=头数量
        b, n, d, h = *x.shape, self.heads
        
        # 计算图像的边长（假设是方形的）
        #把展平的像素数量n，还原成原来的图像高度/宽度
        # s = sqrt(n-1)，因为有一个CLS token
        s = int((n-1) ** 0.5)
        
        # 分离CLS token和特征tokens
        #在PyTorch中，引用数字0-》维度消失，索引用切片-》维度保留
        c = x[:,0,:].reshape(b,1,d)  # 第一个是CLS token（分类标记）
        f = x[:,1:,:]                # 剩下的都是特征tokens

        # 如果使用空间缩减（SR）
        if use_SR==True:
            # 把Q reshape成多头形式：(batch, head, num, dim/head)，x=【批次，token数，维度】
            q = x.reshape(b, n, h, d // h).permute(0, 2, 1, 3)
            
            # 把特征tokens reshape成2D图像形式：(batch, dim, height, width)
            f_ = f.permute(0, 2, 1).reshape(b, d, s, s)
            
            # 用卷积下采样，把图像缩小一半
            f_ = self.sr(f_)#f_:(b,d,s,s)
            
            # 继续reshape和permute
            #返回的f_=（b,n,d）
            f_ = rearrange(f_, 'b d s s -> b d (s s)').permute(0, 2, 1)
            
            # 把CLS token和下采样后的特征拼接
            f_ = torch.cat((c, f_), dim=1)
            
            # 归一化 + 激活
            f_ = self.norm(f_)
            f_ = self.act(f_)
            
            # 用to_kv生成K和V
            #dim=-1,对最后一维进行操作
            kv = self.to_kv(f_).chunk(2, dim = -1)
            
            # 把K和V也reshape成多头形式
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)
        
        # 如果不使用空间缩减
        else:
            # 用to_qkv生成Q、K、V
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            
            # 把Q、K、V都reshape成多头形式
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # 计算注意力分数（Q和K的点积）
        # einsum是爱因斯坦求和约定，写起来很简洁
        # 'bhid,bhjd->bhij' 意思是：batch, head, i, dim 点乘 batch, head, j, dim
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        # 掩码值的默认值（负无穷，这样softmax后就是0）
        mask_value = -torch.finfo(dots.dtype).max

        # 如果传入了mask
        if mask is not None:
            # 填充mask（在前面加一个1，因为CLS token不需要mask）
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            
            # 检查mask维度是否正确
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            
            # 扩展mask成二维
            mask = mask[:, None, :] * mask[:, :, None]
            
            # 用掩码把注意力分数设为负无穷
            dots.masked_fill_(~mask, float('-inf'))
            
            # 删除mask节省内存
            del mask

        # 对注意力分数做softmax（归一化）
        # 这样所有分数加起来等于1，表示"关注程度"的概率分布
        attn = dots.softmax(dim=-1)

        # 用注意力权重乘以V，得到输出
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        # 把多头的结果合并成一个
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # 经过一个线性层
        out = self.nn1(out)
        
        # 经过Dropout
        out = self.do1(out)
        
        return out


# ==================== 第七部分：Transformer模块 ====================

class Transformer(nn.Module):
    """
    Transformer编码器模块
    作用：由多个注意力层和MLP层交替组成，是模型的核心结构
    
    结构：每个block包含两部分
    1. Attention（自注意力层）：让token之间互相"看"对方
    2. MLP（前馈神经网络层）：进一步处理特征
    
    就像一个"流水线"，数据一遍遍经过这些层，每遍都学点新东西
    """
    
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        """
        初始化函数
        输入：
            dim - 特征维度
            depth - 有多少个Transformer块（层数）
            heads - 注意力头数量
            mlp_dim - MLP的隐藏层维度，MLP做的事：升维-》激活函数-》降维回来
            dropout - Dropout比例
        """
        super().__init__()
        
        # 创建空的层列表
        self.layers = nn.ModuleList([])
        
        # 循环创建depth个Transformer块
        for _ in range(depth):
            # 每个块包含：
            # 1. 残差+归一化+注意力
            # 2. 残差+归一化+MLP
            # 用ModuleList保存，这样PyTorch能识别所有参数
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        """
        前向传播：依次通过所有层
        输入：
            x - 输入数据
            mask - 可选的注意力掩码
        返回：处理后的数据
        """
        # 遍历每一对（attention, mlp）
        for attention, mlp in self.layers:
            # 先过注意力层
            x = attention(x, mask=mask)
            
            # 再过MLP层
            x = mlp(x)
        
        return x

# ==================== 第八部分：Block类 ====================

class Block(nn.Module, mamba_init):
    """
    Mamba状态空间模型块（核心模块）
    作用：结合了Mamba状态空间模型和神经网络的前向传播
    
    比喻：这是整个模型的"发动机"
    - 它不是简单地传递数据，而是用状态空间模型来处理
    - 状态空间模型就像一个"记忆机器"，能记住长序列的信息
    - 比普通的Transformer更省显存，适合处理长序列
    
    结构：
    1. 输入投影 (in_proj)：把输入分成两路
    2. 卷积层：提取局部特征
    3. 状态空间扫描：选择性扫描，捕捉长距离依赖
    4. 输出投影 (out_proj)：把特征变回原始维度
    """
    
    def __init__(self,
                 scan_type=None,         # 扫描类型
                 group_type = None,       # 分组类型
                 k_group = None,         # 分组数量，Mamba的扫描分组，通常是2，双向扫描
                 #dim是输入输出维度的通用叫法，用来描述数据的形状
                 dim=None,               # 输入维度和embed_dim一样，它是模型的“输入口径”，mamba的输入，必须是dim维的特征，高光谱传来的数据如果波段是200，dim=64时，需要把200个波段映射成64维
                 dt_rank="auto",         # 时间步投影的秩
                 d_state = None,          # 状态维度：Mamba状态空间的状态数（通常是16）
                 d_model = None,         # 模型维度和embed_dim一样
                 ssm_ratio = None,        # SSM扩展比例
                 bimamba=None,            # 是否使用双向Mamba
                 seq=False,               # 是否顺序处理
                 force_fp32=True,         # 强制使用32位浮点
                 dropout=0.0,             # Dropout比例
                 **kwargs):               # 其他参数
        """
        初始化函数：创建Mamba块的所有组件
        """
        super().__init__()
        
        # 选择激活函数：SiLU（平滑的SiLU，比ReLU更好）
        act_layer = nn.SiLU
        
        # 下面是一些超参数的默认值
        dt_min = 0.001    # 时间步的最小值
        dt_max = 0.1      # 时间步的最大值
        dt_init = "random"  # 初始化方式
        dt_scale = 1.0    # 缩放因子
        dt_init_floor = 1e-4  # 初始化下界
        bias = False     # 是否有偏置
        
        # 保存一些属性
        self.force_fp32 = force_fp32  # 强制32位浮点
        self.seq = seq                # 是否顺序处理
        self.k_group = k_group        # 分组数量
        self.group_type = group_type  # 分组类型
        self.scan_type = scan_type    # 扫描类型
        
        # 计算内部维度：d_inner = ssm_ratio * d_model
        # 就像膨胀系数，ssm_ratio=2表示维度扩大2倍
        d_inner = int(ssm_ratio * d_model)
        
        #dt_rank=一个很小的辅助通道数，专门用来计算“光谱该怎么一步一步走”，走路快慢，不需要很大，只需要是模型维度的1/16就足够了
        #ceil:向上取整
        # 如果dt_rank是"auto"，就自动计算：d_model / 16，向上取整
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # ==================== 1. 输入投影层 ====================
        # in_proj：把dim维的输入变成 d_inner*2 维
        # 为什么要*2？因为后面要 chunk成 两路（x和z）
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias, **kwargs)
        
        # 激活函数
        #: nn.Module，这是类型注解，告诉代码：self.act 是一个神经网络层（PyTorch 模块）
        #相当于self.act = nn.GELU()
        self.act: nn.Module = act_layer()
        
        # 1D卷积：用于处理序列数据（batch,channels,sequence_length）
        self.forward_conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner, kernel_size=1
        )
        
        # 2D卷积：用于处理2D图像数据（空间维度）,给每个通道做独立通道增强
        self.conv2d = nn.Conv2d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,  # groups=d_inner是深度可分离卷积，每个通道独立处理
            bias=True, kernel_size=(1, 1), **kwargs,
        )
        
        # 3D卷积：用于处理3D视频/光谱数据，对每个 3D 点做轻量化特征增强
        self.conv3d = nn.Conv3d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias = True, kernel_size=(1, 1, 1), ** kwargs,
        )

        # ==================== 2. X投影层 ====================
        # 作用：把处理后的特征投影到状态空间需要的维度
        # x_proj：生成B（离散化参数）和C（输出权重）
        #输出 = dt_rank      +    d_state    +    d_state
        #       (delta参数)   +   (B矩阵)    +   (C矩阵)
        #dt_rank:光谱从波段1-》2-》3...时，每一步跨多大，算步长用的中间维度
        #B:当前波段，要往记忆里存多少新内容
        #C:当前波段，要把过去记忆拿出多少来用
        self.x_proj = [
            nn.Linear(d_inner, #输入：模型处理后的高光谱特征
                      (dt_rank + d_state * 2),#输出：三个参数拼在一起
                        bias=False, **kwargs)  # 输出: dt_rank + d_state*2
            for _ in range(k_group)  # k_group个投影
        ]
        
        # 把所有x_proj的权重堆在一起，方便计算
        #x_proj有k_group个小线性层，每个线性层都有一个权重，将这些权重在第0维堆叠起来
        #nn.Parameter：这一大块权重是模型参数，要参与训练、要更新！
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        
        # 删除列表，释放内存
        del self.x_proj

        # ==================== 3. DT投影层 ====================
        # 作用：生成时间步参数delta
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **kwargs)
            for _ in range(k_group)
        ]
        
        # 把所有dt_proj的权重和偏置堆在一起
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        
        del self.dt_projs
        # ==================== 4. A和D参数 ====================
        # A：状态转移矩阵（对数形式），控制状态如何更新
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        
        # D：跳跃连接参数（"skip"连接），就像高速公路
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)

        # ==================== 5. 输出投影层 ====================
        # 层归一化
        self.out_norm = nn.LayerNorm(d_inner)
        
        # 把d_inner维变回dim维
        self.out_proj = nn.Linear(d_inner, dim, bias=bias, **kwargs)
        
        # Dropout：如果dropout>0就添加，否则什么都不做
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()


    def scan(self, x, scan_type=None, group_type=None, route=None):
        """
        选择性扫描函数
        作用：把输入按照不同方向扫描，生成多个"视角"
        
        输入：
            x - 输入数据
            scan_type - 扫描类型
            group_type - 分组类型（'Patch'或'Linear'）
            route - 路由（可选）
        
        返回：扫描后的结果
        """
        # 如果是Patch模式（图像块模式）
        if group_type == 'Patch':
            # 把x从 (B, H, W, D) 变成两种视图：
            # 1. 正常视图：(B, H*W, D)
            # 2. 转置视图：(B, W*H, D)
            #transpose转置
            #contiguous强行整理连续
            x_hwwh = torch.stack([
                x.view(self.B, -1, self.L),  # 正常view
                torch.transpose(x, dim0=1, dim1=2).contiguous().view(self.B, -1, self.L)  # 转置后view
            ], dim=1).view(self.B, 2, -1, self.L)
            
            # 加上翻转版本（前后反向），这样可以捕捉双向信息
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        
        # 如果是Linear模式（线性模式）
        elif group_type == 'Linear':
            # 加上翻转版本
            #flip翻转，dims=[-1] = 在最后一维翻转（就是光谱这一维）
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        
        return xs


    def forward(self, x: Tensor, route=None, SelectiveScan = SelectiveScanMamba):
        """
        前向传播：Mamba块的核心计算
        
        输入：
            x - 输入数据，形状 (batch, sequence, dim)
            route - 路由（可选）
            SelectiveScan - 选择性扫描函数（默认用SelectiveScanMamba）
        
        返回：处理后的结果
        """
        # 第1步：输入投影
        # in_proj把dim维变成 d_inner*2 维
        # 比如：dim=96, d_inner=192 → 输入[10,64,96] → 输出[10,64,384]
        x = self.in_proj(x)
        
        # 第2步：分裂成两路
        # chunk(2, dim=-1) 把最后一维分成两半
        # x: 主路径的特征，[10, 64, 192]
        # z: 门控信号，[10, 64, 192]
        x, z = x.chunk(2, dim=-1)
        
        # 对z做激活（门控机制，让模型自己决定信息是否通过）
        z = self.act(z)

        # 第3步：如果是Linear模式，用1D卷积处理
        if self.group_type == 'Linear':
            # rearrange: (b s d) -> (b d s)
            x1_rearranged = rearrange(x, "b s d -> b d s").contiguous()
            x = self.forward_conv1d(x1_rearranged)
            x = self.act(x)

        # 定义选择性扫描函数（调用C++/CUDA实现的高效版本）
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        # 获取数据维度信息
        if len(x.size()) == 3:
            B, D, L = x.shape  # B=batch, D=维度, L=序列长度
        
        self.B = B  # 保存到self
        self.L = L
        
        # 获取参数维度
        D, N = self.A_logs.shape  #（d_inner,d_state）
        K, D, R = self.dt_projs_weight.shape  # K=有几个组，D=d_inner,R=dt_rank
        # 第4步：扫描
        # 按照group_type进行不同方式的扫描
        xs = self.scan(x, scan_type=self.scan_type, group_type=self.group_type, route=route)

        # 第5步：投影到B、C、delta
        # einsum: 批量矩阵乘法
        # x_dbl形状：[B, K, 3, L]，包含delta、B、C
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        
        # 分割成三部分：delta、B、C
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        
        # delta再投影
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        # 调整维度，准备进入状态空间模型
        xs = xs.view(B, -1, L)              # [B, K*D, L]
        dts = dts.contiguous().view(B, -1, L)  # [B, K*D, L]
        Bs = Bs.contiguous()  # [B, K, N, L]
        Cs = Cs.contiguous()  # [B, K, N, L]

        # 获取A、D、delta_bias参数
        As = -torch.exp(self.A_logs.float())   # A的对数，取负号
        Ds = self.Ds.float()                   # D参数
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # 辅助函数：把所有张量转成float32
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        # 如果强制使用32位浮点
        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        # 第6步：选择性扫描
        if self.seq:
            # 顺序处理每个组
            out_y = []
            for i in range(self.k_group):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i],    # 第i个组的xs
                    dts.view(B, K, -1, L)[:, i],   # 第i个组的delta
                    As.view(K, -1, N)[i],          # 第i个组的A
                    Bs[:, i].unsqueeze(1),         # 第i个组的B
                    Cs[:, i].unsqueeze(1),         # 第i个组的C
                    Ds.view(K, -1)[i],              # 第i个组的D
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            # 并行处理所有组（更快）
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)

        # 第7步：合并双向结果
        if out_y.size(1) == 2:
            # 正向 + 反向 = 双向Mamba
            y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])
            y = y.transpose(dim0=1, dim1=2).contiguous()
            y = self.out_norm(y)

        # 第8步：门控输出
        # z是门控信号，和最终结果相乘（元素级别）
        # 就像给结果加一把"锁"，让模型自己决定输出什么
        y = y * z
        
        # 第9步：输出投影 + Dropout
        out = self.dropout(self.out_proj(y))

        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        为推理分配缓存
        用于在推理时预先分配显存，提高效率
        """
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# ==================== 第九部分：MLP_Block类 ====================

class MLP_Block(nn.Module):
    """
    多层感知机块（MLP）
    作用：在Transformer中做"前馈神经网络"
    
    比喻：就像Attention之后的"消化系统"
    - Attention告诉了模型要关注什么
    - MLP负责把这些信息"消化"成更高级的特征
    
    结构：Linear → GELU → Dropout → Linear → Dropout
    两层神经网络，中间有激活函数和Dropout
    """
    
    def __init__(self, in_features, hidden_features, dropout=0.1):
        """
        初始化函数
        输入：
            in_features - 输入特征维度
            hidden_features - 隐藏层特征维度（通常是输入的4倍）
            dropout - Dropout比例
        """
        super().__init__()
        
        # 创建Sequential容器，按顺序执行各层
        self.mlp = nn.Sequential(
            # 第1层：把in_features变到hidden_features
            # 比如：96 → 384（扩大4倍）
            nn.Linear(in_features, hidden_features),
            
            # 激活函数：GELU（比ReLU更平滑，效果更好）
            nn.GELU(),
            
            # Dropout：随机丢弃一些神经元（防止过拟合）
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            
            # 第2层：把hidden_features变回in_features
            # 96 ← 384
            nn.Linear(hidden_features, in_features)
        )

    def forward(self, x):
        """
        前向传播：按顺序执行MLP
        """
        return self.mlp(x)


# ==================== 第十部分：VisionMamba 主模型类 ====================

class VisionMamba(nn.Module):
    """
    Vision Mamba 主模型（核心模型）
    作用：整合所有组件，实现高光谱图像分类
    
    比喻：这是整个项目的"大脑"
    - 先用3D卷积提取光谱-空间特征
    - 再用Transformer学习全局依赖
    - 再用Mamba捕捉长距离依赖
    - 最后用分类头输出类别
    
    支持的模型类型（model_type）：
    1. 'Parallel MT': 并行Mamba-Transformer（同时运行，并行处理）
    2. 'Interval MT': 间隔Mamba-Transformer
    3. 'Series MT': 串行Mamba-Transformer（先Mamba后Transformer）
    4. 'Series TM': 串行Transformer-Mamba（先Transformer后Mamba）
    """
    
    def __init__(
            self,
            model_type=None,        # 模型类型，有并行，间隔，串行Mamba->Transformer,Transformer->mamba
            k_group=None,          # 分组数量:mamba的扫描分组，通常是2，双向扫描
            depth=None,            # 模型深度（层数）：有多少个Mamba+Transformer block叠加
            embed_dim=None,        # 嵌入维度：数据特征的“宽度”，输出特征的维度，嵌入维度越大，看到的越精细
            d_state: int = None,  # 状态维度：mamba状态空间的状态数，状态空间的记忆容量，越大记得越多，但算的越慢
            ssm_ratio: int = None,  # SSM扩展比例，控制mamba内部维度
            num_classes: int = None,  # 分类数量，要区分多少类
            drop_rate=0.,          # Dropout比例，训练时随机丢弃多少神经元
            drop_path_rate=0.1,     # 路径丢弃比例，随机深度比例，训练时随机跳过多少层
            fused_add_norm=False,  # 是否融合加法和归一化
            residual_in_fp32=True, # 是否用32位浮点做残差连接，更稳定
            bimamba=True,          # 是否使用双向Mamba
            # video相关参数
            fc_drop_rate=0.,       # 随机扔掉神经元的比例
            # checkpoint相关
            use_checkpoint=False,  # 是否使用检查点（省显存）
            checkpoint_num=0,      # 是否使用梯度检查点
            Pos_Cls = False,      # 是否使用位置分类
            pos: str = None,      # 位置编码类型
            cls: str = None,      # CLS类型
            conv3D_channel: int = None,  # 3D卷积通道数
            conv3D_kernel_1: int = None,  # 3D卷积核大小1
            conv3D_kernel_2: int = None,  # 3D卷积核大小2
            conv3D_kernel_3: int = None,  # 3D卷积核大小3
            dim_patch: int = None,  # Patch维度
            dim_linear_1: int = None,  # 线性层维度1
            dim_linear_2: int = None,  # 线性层维度2
            dim_linear_3: int = None,  # 线性层维度3
            use_hyperbolic=False,
            use_fuzzy=True,
            fuzzy_num: int=None,
            **kwargs,             # 其他参数
        ):
        """
        初始化函数：创建完整的Vision Mamba模型
        """
        super().__init__()
        
        # ==================== 新增：双曲模块（带开关）====================
        self.use_hyperbolic = use_hyperbolic
        self.use_fuzzy = use_fuzzy
        self.fuzzy_num=fuzzy_num
        if self.use_hyperbolic:
            # 降低max_norm，减少溢出风险
            self.hyper_manifold = LorentzManifold(eps=1e-8, norm_clip=0.5, max_norm=3.0)
            self.hyper_proj = nn.Linear(embed_dim, embed_dim)
            # 关键：用小方差初始化，避免初始特征范数过大
            nn.init.xavier_uniform_(self.hyper_proj.weight, gain=0.05)
            nn.init.zeros_(self.hyper_proj.bias)
        # ===============================================================

        # 保存各种配置参数
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.Pos_Cls = Pos_Cls
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # 保持一致性
        self.k_group = k_group
        self.depth = depth
        self.model_type = model_type

        # ==================== 第一部分：3D卷积特征提取 ====================
        # 作用：用3D卷积同时提取光谱维度和空间维度的特征
        # 3D卷积比2D卷积多一个"时间/光谱"维度，能更好地捕捉光谱信息
        
        # 第1个3D卷积分支：核大小conv3D_kernel_1（通常是5×5×5）
        self.conv3d_features_1 = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel_1),  # 3D卷积
            nn.BatchNorm3d(conv3D_channel),  # 批归一化
            nn.ReLU(),  # 激活函数
        )
        
        # 第2个3D卷积分支：核大小conv3D_kernel_2（通常是7×7×7）
        self.conv3d_features_2 = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel_2),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )
        
        # 第3个3D卷积分支：核大小conv3D_kernel_3（通常是9×9×9）
        self.conv3d_features_3 = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel_3),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )
        

        # ==================== 第二部分：空间嵌入层 ====================
        # 作用：把卷积后的特征图转换成向量形式（类似ViT的patch embedding）
        
        # 第1个分支的嵌入层
        #conv3D_channel：3D 卷积提取出来的通道数（特征数）,dim_linear_1：空间维度展开后的长度（比如像素数）
        self.embedding_spatial_1 = nn.Sequential(nn.Linear(conv3D_channel * dim_linear_1, embed_dim))
        
        # 第2个分支的嵌入层
        self.embedding_spatial_2 = nn.Sequential(nn.Linear(conv3D_channel * dim_linear_2, embed_dim))
        
        # 第3个分支的嵌入层
        self.embedding_spatial_3 = nn.Sequential(nn.Linear(conv3D_channel * dim_linear_3, embed_dim))

        # ==================== 第三部分：归一化和池化 ====================
        self.norm = nn.LayerNorm(embed_dim)  # 层归一化
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化（输出1×1）,把整个图浓缩为一个点
        self.flatten = nn.Flatten(1)  # 展平层

        # ==================== 第四部分：位置编码和Dropout ====================
        # CLS token：类似于BERT的[CLS]标记，用于分类
        #cls_token 就是专门用来吸收全局信息的专用向量！
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # 位置编码：让模型知道每个token的位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, 1792 + 1, self.embed_dim))
        
        # 时间位置编码
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, 28, embed_dim))
        
        # 位置Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ==================== 第五部分：分类头 ====================
        # 作用：把特征向量转换成类别概率
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # ==================== 第六部分：Drop Path（随机深度）====================
        # 作用：训练时随机丢弃一些路径，防止过拟合
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # ==================== 第七部分：投影层 ====================
        #融合信息
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # ==================== 第八部分：Transformer层 ====================
        # 作用：用Transformer的注意力机制学习全局依赖
        # 三个并行的Transformer分支，处理三种不同尺度的特征
        
        self.transformer_1 = nn.ModuleList([Residual(LayerNormalize(embed_dim, Attention(embed_dim, heads=8, dropout=drop_path_rate)))
                for i in range(depth)])
        self.transformer_2 = nn.ModuleList([Residual(LayerNormalize(embed_dim, Attention(embed_dim, heads=8, dropout=drop_path_rate)))
                for i in range(depth)])
        self.transformer_3 = nn.ModuleList([Residual(LayerNormalize(embed_dim, Attention(embed_dim, heads=8, dropout=drop_path_rate)))
                for i in range(depth)])

        # ==================== 第九部分：FFN层（前馈神经网络）====================
        self.FFN = nn.ModuleList([Residual(
                LayerNormalize(
                embed_dim, MLP_Block(embed_dim, embed_dim, dropout=drop_path_rate)))
                for i in range(depth)])

        # ==================== 第十部分：Mamba层 ====================
        # 作用：用Mamba的状态空间模型捕捉长距离依赖
        self.layers = nn.ModuleList([Block(
                group_type='Linear',  # 线性扫描模式
                k_group=2,           # 2个分组（双向）
                dim=embed_dim,
                d_state=d_state,
                d_model=embed_dim,
                ssm_ratio=ssm_ratio,
                bimamba=bimamba,
                **kwargs, )
                for i in range(depth)])
        
        # ==================== 初始化模糊融合层 ====================
        if self.use_fuzzy:
            self.fuzzy_fusion_1 = nn.ModuleList([FuzzyLayer3(fuzzynum=fuzzy_num, channel=embed_dim) for _ in range(depth)])
            self.fuzzy_fusion_2 = nn.ModuleList([FuzzyLayer3(fuzzynum=fuzzy_num, channel=embed_dim) for _ in range(depth)])
            self.fuzzy_fusion_3 = nn.ModuleList([FuzzyLayer3(fuzzynum=fuzzy_num, channel=embed_dim) for _ in range(depth)])
            self.fuzzy_fusion_mamba = nn.ModuleList([FuzzyLayer3(fuzzynum=fuzzy_num, channel=embed_dim) for _ in range(depth)])


    def get_num_layers(self):
        """
        获取模型的层数
        """
        return len(self.layers)


    def scan(self, x, scan_type=None, group_type=None):
        """
        扫描函数：重排数据的维度顺序
        作用：把(b,c,t,h,w)变成(b,h,w,c)的格式
        
        输入：x - 3D卷积的输出
        返回：重排后的数据
        """
        # 第一步：(b,c,t,h,w) → (b,(c*t),h,w)
        # 把通道和光谱维度合并
        x = rearrange(x, 'b c t h w -> b (c t) h w')
        
        # 第二步：(b,c,h,w) → (b,h,w,c)
        # 把通道维度移到最后，方便后续处理
        x = rearrange(x, 'b c h w -> b h w c')
        return x


    def mask_generate(self, img_size, num):
        """
        生成掩码（暂时未使用，可能是为了一些特殊功能）
        """
        out = []
        mask_none = torch.ones(img_size, img_size)
        for i in reversed(range(0, num - 1)):
            mask = torch.ones(img_size // (2 ** i), img_size // (2 ** i))
            mask = torch.triu(mask, diagonal=0)
            mask = torch.rot90(mask, k=1, dims=(0, 1))
            out.append(
                torch.nn.functional.pad(mask, (0, img_size - img_size // (2 ** i), 0, img_size - img_size // (2 ** i)),
                                        mode='constant', value=0))
        out.append(mask_none)
        return out


    def forward_features(self, x, inference_params=None):
        """
        特征提取函数：模型的核心前向传播
        
        输入：x - 原始输入数据，形状 (batch, 1, bands, height, width)
             例如：[10, 1, 30, 15, 15] = 10个样本，1个通道，30个波段，15×15像素
        
        返回：feature - 提取后的特征向量，形状 (batch, embed_dim)
        """
        # ==================== 第1步：3D卷积特征提取 ====================
        # 用三个不同核大小的3D卷积提取特征
        # 输入：[10, 1, 30, 15, 15] → 输出三个不同尺度的特征
        x_1 = self.conv3d_features_1(x)  # [10, 32, 28, 11, 11]
        x_2 = self.conv3d_features_2(x)  # [10, 32, 28, 9, 9]
        x_3 = self.conv3d_features_3(x)  # [10, 32, 28, 7, 7]
    
        # ==================== 第2步：扫描重排 ====================
        x_1 = self.scan(x_1)   # [64, 11, 11, 832]
        x_2 = self.scan(x_2)   # [64, 9, 9, 768]
        x_3 = self.scan(x_3)   # [64, 7, 7, 704]
        


        # ==================== 第3步：空间嵌入 ====================
        # 把卷积特征转换成token嵌入
        x_1 = self.embedding_spatial_1(x_1)  # [64, 11, 11, 32]
        x_2 = self.embedding_spatial_2(x_2)  # [64, 9, 9, 32]
        x_3 = self.embedding_spatial_3(x_3)  # [64, 7, 7, 32]
        
        # ==================== 第4步：位置Dropout ====================
        x_1 = self.pos_drop(x_1)
        x_2 = self.pos_drop(x_2)
        x_3 = self.pos_drop(x_3)

        # ==================== 第5步：根据模型类型选择处理方式 ====================
        if self.model_type == 'Parallel MT' or self.model_type == 'Interval MT' or self.model_type == 'Series MT' or self.model_type == 'Series TM':
            # 把三个分支的特征都flatten成序列
            LG_1_init = rearrange(x_1, 'b h w c-> b (h w) c')  # [64, 121, 32]
            LG_2_init = rearrange(x_2, 'b h w c-> b (h w) c')  # [64, 81, 32]
            LG_3_init = rearrange(x_3, 'b h w c-> b (h w) c')  # [64, 49, 32]
            
            # 拼接三个分支：[64, 251, 32]
            LG = torch.cat([LG_1_init, LG_2_init, LG_3_init], dim=1)

            
            # ==================== Parallel MT（并行Mamba-Transformer）====================
            if self.model_type == 'Parallel MT':
                # 同时使用Transformer和Mamba，结果相加
                for i in range(self.depth):
                    #保存Transformer之前的特征
                    LG_before_T_1 = LG[:, 0:LG_1_init.shape[1], :]
                    LG_before_T_2 = LG[:, LG_1_init.shape[1]:LG_1_init.shape[1] + LG_2_init.shape[1], :]
                    LG_before_T_3 = LG[:, LG_1_init.shape[1] + LG_2_init.shape[1]:, :]

                    # Transformer处理三个分支
                    LG_after_T_1 = self.transformer_1[i](LG_before_T_1, mask=None)
                    LG_after_T_2 = self.transformer_2[i](LG_before_T_2, mask=None)
                    LG_after_T_3 = self.transformer_3[i](LG_before_T_3, mask=None)

                    if self.use_fuzzy:
                        LG_fused_1 = self.fuzzy_fusion_1[i](LG_before_T_1, LG_after_T_1)
                        LG_fused_2 = self.fuzzy_fusion_2[i](LG_before_T_2, LG_after_T_2)
                        LG_fused_3 = self.fuzzy_fusion_3[i](LG_before_T_3, LG_after_T_3)
                    else:
                        LG_fused_1 = LG_before_T_1 + LG_after_T_1
                        LG_fused_2 = LG_before_T_2 + LG_after_T_2
                        LG_fused_3 = LG_before_T_3 + LG_after_T_3
                    
                    # 拼接Transformer结果
                    LG_T_fused = torch.cat([LG_fused_1, LG_fused_2, LG_fused_3], dim=1)
                    LG_T_fused = self.FFN[i](LG_T_fused)
                    
                    
                    # Mamba处理（残差连接）
                    LG_M = LG + self.drop_path(self.layers[i](self.norm(LG)))

                    # 两者相加
                    LG = LG_T_fused + LG_M


            # ==================== Interval MT（间隔Mamba-Transformer）====================
            elif self.model_type == 'Interval MT':
                for i in range(self.depth):
                    LG_before_T_1 = LG[:, 0:LG_1_init.shape[1], :]
                    LG_before_T_2 = LG[:, LG_1_init.shape[1]:LG_1_init.shape[1] + LG_2_init.shape[1], :]
                    LG_before_T_3 = LG[:, LG_1_init.shape[1] + LG_2_init.shape[1]:, :]

                    LG_1 = self.transformer_1[i](LG_before_T_1, mask=None)
                    LG_2 = self.transformer_2[i](LG_before_T_2, mask=None)
                    LG_3 = self.transformer_3[i](LG_before_T_3, mask=None)

                    if self.use_fuzzy:
                        LG_fused_1 = self.fuzzy_fusion_1[i](LG_before_T_1, LG_1)
                        LG_fused_2 = self.fuzzy_fusion_2[i](LG_before_T_2, LG_2)
                        LG_fused_3 = self.fuzzy_fusion_3[i](LG_before_T_3, LG_3)
                    else:
                        LG_fused_1 = LG_before_T_1 + LG_1
                        LG_fused_2 = LG_before_T_2 + LG_2
                        LG_fused_3 = LG_before_T_3 + LG_3

                    LG_T = torch.cat([LG_fused_1, LG_fused_2, LG_fused_3], dim=1)
                    LG_T = self.FFN[i](LG_T)
                    LG = LG_T + self.drop_path(self.layers[i](self.norm(LG_T)))

            # ==================== Series TM1（串行Transformer→Mamba）====================
            elif self.model_type == 'Series TM':
                for i in range(self.depth):
                    LG_before_T_1 = LG[:, 0:LG_1_init.shape[1], :]
                    LG_before_T_2 = LG[:, LG_1_init.shape[1]:LG_1_init.shape[1] + LG_2_init.shape[1], :]
                    LG_before_T_3 = LG[:, LG_1_init.shape[1] + LG_2_init.shape[1]:, :]

                    LG_1 = self.transformer_1[i](LG_before_T_1, mask=None)
                    LG_2 = self.transformer_2[i](LG_before_T_2, mask=None)
                    LG_3 = self.transformer_3[i](LG_before_T_3, mask=None)

                    if self.use_fuzzy:
                        LG_fused_1 = self.fuzzy_fusion_1[i](LG_before_T_1, LG_1)
                        LG_fused_2 = self.fuzzy_fusion_2[i](LG_before_T_2, LG_2)
                        LG_fused_3 = self.fuzzy_fusion_3[i](LG_before_T_3, LG_3)
                    else:
                        LG_fused_1 = LG_before_T_1 + LG_1
                        LG_fused_2 = LG_before_T_2 + LG_2
                        LG_fused_3 = LG_before_T_3 + LG_3

                    LG = torch.cat([LG_fused_1, LG_fused_2, LG_fused_3], dim=1)
                    LG = self.FFN[i](LG)

                for i in range(self.depth):
                    LG = LG + self.drop_path(self.layers[i](self.norm(LG)))

            # ==================== Series MT2（串行Mamba→Transformer）====================
            elif self.model_type == 'Series MT':
                for i in range(self.depth):
                    LG = LG + self.drop_path(self.layers[i](self.norm(LG)))

                for i in range(self.depth):
                    LG_before_T_1 = LG[:, 0:LG_1_init.shape[1], :]
                    LG_before_T_2 = LG[:, LG_1_init.shape[1]:LG_1_init.shape[1] + LG_2_init.shape[1], :]
                    LG_before_T_3 = LG[:, LG_1_init.shape[1] + LG_2_init.shape[1]:, :]

                    LG_1 = self.transformer_1[i](LG_before_T_1, mask=None)
                    LG_2 = self.transformer_2[i](LG_before_T_2, mask=None)
                    LG_3 = self.transformer_3[i](LG_before_T_3, mask=None)

                    if self.use_fuzzy:
                        LG_fused_1 = self.fuzzy_fusion_1[i](LG_before_T_1, LG_1)
                        LG_fused_2 = self.fuzzy_fusion_2[i](LG_before_T_2, LG_2)
                        LG_fused_3 = self.fuzzy_fusion_3[i](LG_before_T_3, LG_3)
                    else:
                        LG_fused_1 = LG_before_T_1 + LG_1
                        LG_fused_2 = LG_before_T_2 + LG_2
                        LG_fused_3 = LG_before_T_3 + LG_3

                    LG = torch.cat([LG_fused_1, LG_fused_2, LG_fused_3], dim=1)
                    LG = self.FFN[i](LG)

        # ==================== 第6步：全局平均池化 ====================
        # 对所有token取平均，得到一个全局特征向量
        feature = LG.mean(dim=1)  # [64, 32]

        # ==================== 双曲增强（开关控制）====================
        if self.use_hyperbolic:
            # 1. 先对特征做归一化，限制输入范围
            feature = nn.functional.normalize(feature, dim=-1)
            # 2. 投影（带温和初始化）
            feat = self.hyper_proj(feature)
            # 3. 限制投影后的特征范数，防止溢出
            feat_norm = torch.norm(feat, dim=-1, keepdim=True)
            feat = feat / (1 + feat_norm * 0.5)
            # 4. 映射到洛伦兹空间
            feat_l = self.hyper_manifold.from_poincare_to_lorentz(feat)
            # 5. 归一化约束
            feat_l = self.hyper_manifold.normalize(feat_l)
            # 6. 映射回Poincare盘
            feature = self.hyper_manifold.from_lorentz_to_poincare(feat_l)
            # 7. 输出再归一化，保持特征分布稳定
            feature = nn.functional.normalize(feature, dim=-1)
        # ==============================================================

        return feature


    def forward(self, x, inference_params=None):
        """
        完整的前向传播：特征提取 + 分类
        
        输入：x - 原始输入数据
        返回：
            x - 类别预测（logits）
            feature - 特征向量
        """
        # 第1步：提取特征
        feature = self.forward_features(x, inference_params)
        
        # 第2步：分类
        # 先过Dropout，再过线性层得到类别概率
        x = self.head(self.head_drop(feature))
        
        # 返回预测结果和特征
        return x, feature
