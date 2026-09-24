"""
=================================================================
项目说明：高光谱图像分类系统
=================================================================
核心流程：
1. 数据预处理：PCA降维 → 创建图像块 → 划分训练/测试集
2. 模型训练：使用3D卷积提取光谱-空间特征，结合Mamba状态空间模型
3. 模型测试：输出分类精度（OA、平均精度AA、Kappa系数等指标）
4. 可视化：生成分类地图

用途：对卫星拍摄的高光谱图像进行分类，识别不同地物类型
=================================================================
"""

# ==================== 第一部分：导入各种工具库 ====================

# 导入PyTorch深度学习框架（必选），用于构建神经网络
import torch

# 从models文件夹导入videomamba模块，这是Vision Mamba模型
import HMTF

# 导入os模块，用于操作文件和目录
import os

# 从config_PU.py导入config对象，里面存放所有可配置参数
import config

# 设置使用哪个GPU（0表示第1块显卡），config.gpus通常是'0'，表示用第0号显卡，如果有多个显卡也可以改成别的
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

# 导入numpy，用于处理数值数组（图像数据就是大数组）
import numpy as np

# 导入scipy的io模块，用于读取.mat格式的文件（MATLAB数据）
import scipy.io as sio

# 从sklearn导入PCA（主成分分析），用于降维
from sklearn.decomposition import PCA

# 从sklearn导入train_test_split，用于划分训练集和测试集
from sklearn.model_selection import train_test_split

# 从sklearn导入各种评估指标：
# - confusion_matrix: 混淆矩阵
# - accuracy_score: 准确率
# - classification_report: 分类报告
# - cohen_kappa_score: Kappa系数
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

# 再次导入torch（深度学习框架的核心）
import torch

# 导入torch.nn模块，里面有各种神经网络层（卷积层、全连接层等），nn是神经网络模块
import torch.nn as nn

# 导入torch.optim模块，里面有各种优化器（Adam、SGD等），optim是优化器模块
import torch.optim as optim

# 从operator模块导入truediv，用于精确除法
from operator import truediv

# 导入time模块，用于记录程序运行时间
import time

# 导入gc模块，用于手动释放内存（处理大数据集时很有用）
import gc

# 从utils.utils模块导入Logger和mkdirs：
# - Logger: 用于写日志（记录训练过程）
# - mkdirs: 用于创建目录
from utils.utils import Logger, mkdirs

# 导入get_cls_map_PU模块，用于生成分类地图
import get_cls_map_PU


# ==================== 第二部分：定义各种函数 ====================

# -------------------- 函数1：loadData() - 加载数据 --------------------
def loadData():
    """
    加载高光谱图像数据（.mat格式文件）
    根据config.data选择要加载的数据集
    返回：图像数据data和标签labels
    注意：自动将float64转为float32以节省一半内存
    """
    
    # 如果配置的数据集是'PaviaU'
    if config.data == 'PaviaU':
        # 使用scipy读取PaviaU.mat文件，['paviaU']是文件里的变量名
        data = sio.loadmat('./data/PaviaU/PaviaU.mat')['paviaU']
        # 读取对应的标签文件（ground truth）
        labels = sio.loadmat('./data/PaviaU/PaviaU_gt.mat')['paviaU_gt']

    # 如果配置的数据集是'LongKou'
    elif config.data == 'LongKou':
        data = sio.loadmat('./data/LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        labels = sio.loadmat('./data/LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']

    # 如果配置的数据集是'Houston2013'
    elif config.data == 'Houston2013':
        data = sio.loadmat('./data/hus/Houston.mat')['Houston']
        labels = sio.loadmat('./data/hus/Houston_gt.mat')['Houston_GT']

    # 如果配置的数据集是'Salinas'
    elif config.data == 'Salinas':
        data = sio.loadmat('./data/Salinas/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('./data/Salinas/Salinas_gt.mat')['salinas_gt']
    # 如果配置的数据集是'Salinas'
    elif config.data == 'PaviaC':
        data = sio.loadmat('./data/pc/Pavia.mat')['pavia']
        labels = sio.loadmat('./data/pc/Pavia_gt.mat')['pavia_gt']
    
    # 转float32以节省内存（.mat通常是float64，占内存2倍）
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    
    # 返回图像数据和标签
    return data, labels


# -------------------- 函数2：applyPCA() - PCA降维 --------------------
def applyPCA(X, numComponents):
    """
    对高光谱图像进行PCA主成分分析降维（内存优化版）
    输入：
        X: 原始高光谱图像（高×宽×波段数）
        numComponents: 保留的主成分数量（通常是30）
    返回：降维后的图像（高×宽×numComponents）
    """
    
    # 保持原始数据类型（float32节省一半内存）
    orig_dtype = X.dtype
    rows, cols, bands = X.shape
    
    # 把图像从3D数组变成2D数组
    newX = np.reshape(X, (-1, bands))
    # 确保内存连续（避免view导致后续操作创建意外副本）
    newX = np.ascontiguousarray(newX)
    
    # 创建PCA对象，用randomized SVD节约内存
    pca = PCA(n_components=numComponents, whiten=True, svd_solver='randomized')
    
    # 对数据进行PCA变换（保持原始dtype，不转float64）
    newX = pca.fit_transform(newX)
    
    # 确保输出是float32
    if newX.dtype != np.float32:
        newX = newX.astype(np.float32)
    
    # 把数据变回3D数组形状
    newX = np.reshape(newX, (rows, cols, numComponents))

    return newX


# -------------------- 函数3：padWithZeros() - 边缘填充 --------------------
def padWithZeros(X, margin=2):
    """
    给图像边缘加一圈零值padding（防止切图时边缘像素缺失）
    输入：
        X: 原始图像
        margin: 边缘宽度（默认2像素）
    返回：填充后的图像
    """
    
    # 创建一个新的空数组，比原图大2*margin
    # X.shape[0]是高度，X.shape[1]是宽度，X.shape[2]是波段数
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    
    # 计算偏移量
    x_offset = margin
    y_offset = margin
    
    # 把原图放到新图中间，周围一圈是0
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# -------------------- 函数4：createImageCubes() - 创建图像块 --------------------
def createImageCubes(X, y, windowSize, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    label_rows, label_cols = np.where(y > 0)
    patchesData = np.zeros((len(label_rows), windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros(len(label_rows))
    for i in range(len(label_rows)):
        r = label_rows[i] + margin
        c = label_cols[i] + margin
        patchesData[i] = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
        patchesLabels[i] = y[label_rows[i], label_cols[i]]
    patchesLabels -= 1
    return patchesData, patchesLabels


# -------------------- 新增：LazyPatchDataset（按需切图，防止OOM）--------------------
class LazyPatchDataset(torch.utils.data.Dataset):
    """
    按需加载的图像块数据集（不预存所有patch，节省内存）
    适用于大尺寸数据集（如LongKou），避免一次性创建所有patch导致OOM
    """
    def __init__(self, padded_image, coords, labels, patch_size, pca_components):
        self.padded = padded_image           # (H+2*margin, W+2*margin, pca_components)
        self.coords = coords                 # [(row, col), ...] 标签像素坐标
        self.labels = labels - 1             # 标签从0开始
        self.patch_size = patch_size
        self.pca_components = pca_components
        self.margin = (patch_size - 1) // 2

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        r, c = self.coords[idx]
        m = self.margin
        # 从padding后的图像中实时切出patch（不存中间大数组）
        patch = self.padded[r - m:r + m + 1, c - m:c + m + 1, :]  # (ps, ps, bands)
        patch = patch.reshape(1, self.patch_size, self.patch_size, self.pca_components)  # add channel dim
        patch = patch.transpose(0, 3, 1, 2)    # (1, bands, ps, ps)
        patch = torch.FloatTensor(patch)
        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return patch, label


# -------------------- 函数5：splitTrainTestSet() - 划分训练测试集 --------------------
def splitTrainTestSet(X, y, train_samples_per_class, randomState=345):
    """
    固定每类选取 train_samples_per_class 个样本作为训练集
    其余样本作为测试集
    """
    np.random.seed(randomState)
    train_indices = []
    test_indices = []
    
    # 遍历每一个类别
    for cls in range(config.num_classes):
        # 获取当前类别的所有样本索引
        cls_indices = np.where(y == cls)[0]
        # 随机打乱
        np.random.shuffle(cls_indices)
        
        # 选取前15个作为训练集
        train_idx = cls_indices[:train_samples_per_class]
        # 剩下的作为测试集
        test_idx = cls_indices[train_samples_per_class:]
        
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    # 生成训练集和测试集
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


# -------------------- 函数6：create_data_loader() - 创建数据加载器 --------------------
def create_data_loader():
    """
    整合所有数据处理步骤，创建DataLoader
    步骤：加载数据 → PCA降维 → 切图块 → 划分训练测试集 → 封装Dataset → 创建DataLoader
    返回：训练集loader、测试集loader、全部标签、索引、全部数据loader、真实标签
    """
    
    # 第1步：加载原始数据
    X, y = loadData()
    
    # 找出所有有标签的像素位置（标签不为0）
    # 把标签图拉直成一条线，并找出所有【不是0】的像素位置
    # 返回的是一个元组
    index = np.nonzero(y.reshape(y.shape[0]*y.shape[1]))
    # 取出位置数组
    index = index[0]
    
    # 从config读取参数
    train_samples_per_class=config.train_samples_per_class
    patch_size = config.patch_size          # 图像块大小（如15）
    pca_components = config.pca_components  # PCA主成分数（如30）
    
    # 打印原始数据信息
    print('原始高光谱图像形状: ', X.shape)
    print('标签形状: ', y.shape)
    groundtruth = y  # 保存真实标签
    
    # 第2步：PCA降维
    print('\n... ... 正在进行PCA主成分分析 ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('PCA降维后的数据形状: ', X_pca.shape)
    
    # 释放原始大数据（PCA之后不再需要）
    del X
    gc.collect()
    print('已释放原始高光谱数据内存')
    
    # 是否需要使用按需加载
    # 判断依据：原始图像的像素数或标签像素数较大时启用
    total_pixels = y.shape[0] * y.shape[1]
    total_labeled = np.count_nonzero(y)
    # 总像素超15万 或 标签像素超2万 都启用（LongKou约22万像素，必须用按需加载）
    use_lazy = (total_pixels > 150000 or total_labeled > 20000)
    if use_lazy:
        print(f'检测到大尺寸图像（总像素={total_pixels}，标签像素={total_labeled}）')
        print('已启用按需切图模式（LazyPatchDataset），避免内存溢出')
    
    if use_lazy:
        print('\n... ... 数据集较大，启用按需切图模式 ... ...')
        # 预计算padding图像和标签坐标坐标
        margin = (patch_size - 1) // 2
        zeroPaddedX = padWithZeros(X_pca, margin=margin)
        label_rows, label_cols = np.where(y > 0)
        
        # 获取标签值（原始）
        y_labels = y[label_rows, label_cols]
        
        # 划分训练/测试坐标
        np.random.seed(345)
        train_coords = []
        train_labels_list = []
        test_coords = []
        test_labels_list = []
        
        for cls in range(config.num_classes):
            cls_mask = (y_labels == cls + 1)  # 标签从1开始
            cls_indices = np.where(cls_mask)[0]
            np.random.shuffle(cls_indices)
            cls_labels = y_labels[cls_indices]
            cls_rows = label_rows[cls_indices]
            cls_cols = label_cols[cls_indices]
            
            train_idx = cls_indices[:train_samples_per_class]
            test_idx = cls_indices[train_samples_per_class:]
            
            train_coords.extend(list(zip(cls_rows[:train_samples_per_class], cls_cols[:train_samples_per_class])))
            train_labels_list.extend(cls_labels[:train_samples_per_class])
            test_coords.extend(list(zip(cls_rows[train_samples_per_class:], cls_cols[train_samples_per_class:])))
            test_labels_list.extend(cls_labels[train_samples_per_class:])
        
        train_labels_arr = np.array(train_labels_list, dtype=np.int64)
        test_labels_arr = np.array(test_labels_list, dtype=np.int64)
        # 训练集：使用按需加载（但样本少，也可以预加载，统一用Lazy即可）
        trainset = LazyPatchDataset(zeroPaddedX, train_coords, train_labels_arr, patch_size, pca_components)
        testset = LazyPatchDataset(zeroPaddedX, test_coords, test_labels_arr, patch_size, pca_components)
        
        # 所有标签像素（全量）用于生成分类图
        all_coords = list(zip(label_rows, label_cols))
        allset = LazyPatchDataset(zeroPaddedX, all_coords, np.array(list(y_labels), dtype=np.int64), patch_size, pca_components)
        
        print(f'按需加载模式 - 训练样本: {len(train_coords)}, 测试样本: {len(test_coords)}, 总样本: {len(all_coords)}')
    else:
        # 第3步：创建图像块（切小图）
        print('\n... ... 正在创建图像块 ... ...')
        X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
        print('图像块X的形状: ', X_pca.shape)
        print('图像块y的形状: ', y.shape)
        
        # 第4步：划分训练集和测试集
        print('\n... ... 正在划分训练集和测试集 ... ...')
        Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, train_samples_per_class)
        print('训练集Xtrain形状: ', Xtrain.shape)
        print('测试集Xtest形状: ', Xtest.shape)
        
        # 第5步：调整数据形状（为了PyTorch的3D卷积需要）
        X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
        Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
        Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
        
        print('转置前 - Xtrain形状: ', Xtrain.shape)
        print('转置前 - Xtest形状: ', Xtest.shape)
        
        # 第6步：转置数据
        X = X.transpose(0, 4, 3, 1, 2)
        Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
        Xtest = Xtest.transpose(0, 4, 3, 1, 2)
        
        print('转置后 - Xtrain形状: ', Xtrain.shape)
        print('转置后 - Xtest形状: ', Xtest.shape)
        
        # 第7步：封装成PyTorch的Dataset
        X = TestDS(X, y)
        trainset = TrainDS(Xtrain, ytrain)
        testset = TestDS(Xtest, ytest)
        
        allset = X
    
    # 大数据集使用较小的batch size，防止数据加载时内存溢出
    actual_batch_size = config.BATCH_SIZE_TRAIN
    if use_lazy:
        actual_batch_size = min(config.BATCH_SIZE_TRAIN, 32)  # 长序列数据集降为32
    
    # 第8步：创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=actual_batch_size,
        shuffle=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    all_data_loader = torch.utils.data.DataLoader(
        dataset=allset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    return train_loader, test_loader, y, index, all_data_loader, groundtruth


# ==================== 第三部分：定义Dataset类 ====================

""" 训练数据集类：继承自PyTorch的Dataset """
class TrainDS(torch.utils.data.Dataset):
    """
    用于存放训练数据的类
    PyTorch的DataLoader需要Dataset对象来提供数据
    """
    
    def __init__(self, Xtrain, ytrain):
        """
        初始化：把numpy数组转换成PyTorch张量
        输入：Xtrain（训练图像）, ytrain（训练标签）
        """
        # 记录样本数量
        self.len = Xtrain.shape[0]
        
        #numpy数组：只能在CPU上跑，用来处理数据、切图、计算
        #Pytorch Tensor:才能送入模型训练、在GPU上加速、计算梯度、反向传播
        #模型训练、向前传播、损失计算、梯度下降----全都只能用Tensor
        # 把numpy数组转成PyTorch的FloatTensor（浮点张量）
        self.x_data = torch.FloatTensor(Xtrain)
        
        # 把numpy数组转成PyTorch的LongTensor（长整型张量，用于分类标签）
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        """
        根据索引返回对应的数据和标签
        输入：index（索引）
        返回：第index个图像数据和对应的标签
        """
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        """
        返回数据集的大小（有多少个样本）
        """
        return self.len


""" 测试数据集类：继承自PyTorch的Dataset """
class TestDS(torch.utils.data.Dataset):
    """
    用于存放测试数据的类，功能和TrainDS类似
    """
    
    def __init__(self, Xtest, ytest):
        """初始化：把numpy数组转换成PyTorch张量"""
        # 记录样本数量
        self.len = Xtest.shape[0]
        
        # 转换成PyTorch张量
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        """根据索引返回对应的数据和标签"""
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """返回数据集的大小"""
        return self.len


# ==================== 第四部分：定义训练和测试函数 ====================

# -------------------- 函数7：train() - 训练模型 --------------------
def train(train_loader):
    """
    训练Vision Mamba模型
    输入：train_loader（训练数据加载器）
    返回：训练好的模型net、FLOPs、参数量
    """
    
    # 创建Vision Mamba模型，所有参数从config读取
    net = videomamba.VisionMamba(
        model_type=config.model_type,         # 模型类型（如'Parallel MT'）
        embed_dim=config.embed_dim,           # 嵌入维度（默认32）
        d_state=config.d_state,               # 状态维度（默认16）
        ssm_ratio=config.ssm_ratio,           # SSM扩展比例（默认1）
        num_classes=config.num_classes,        # 分类类别数（如PaviaU是9类）
        depth=config.depth,                    # 模型深度（默认3）
        pos=config.pos,                        # 是否使用位置编码
        cls=config.cls,                        # 是否使用CLS token
        conv3D_channel=config.conv3D_channel, # 3D卷积通道数
        conv3D_kernel_1=config.conv3D_kernel_1, # 3D卷积核大小1
        conv3D_kernel_2=config.conv3D_kernel_2, # 3D卷积核大小2
        conv3D_kernel_3=config.conv3D_kernel_3, # 3D卷积核大小3
        dim_patch=config.dim_patch,            # patch维度
        dim_linear_1=config.dim_linear_1,      # 线性层维度1
        dim_linear_2=config.dim_linear_2,      # 线性层维度2
        dim_linear_3=config.dim_linear_3,      # 线性层维度3
        use_hyperbolic=config.use_hyperbolic,
        use_fuzzy=config.use_fuzzy,
        fuzzy_num=config.fuzzy_num,
    ).cuda()  # 把模型放到GPU上
    
    # 定义损失函数：交叉熵损失（分类问题最常用）
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器：Adam，学习率0.001
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # 初始化总损失
    total_loss = 0
    
    # 取一个batch作为profile的输入样本
    profile_input = None
    for batch_data, _ in train_loader:
        profile_input = batch_data.cuda()
        break

    # 开始训练循环（训练config.train_epoch轮，默认100轮）
    for epoch in range(config.train_epoch):
        # 把模型设为训练模式（启用dropout、batch normalization等）
        net.train()
        
        # 从训练数据加载器中一批批读取数据
        for i, (data, target) in enumerate(train_loader):
            # 把数据放到GPU上
            data, target = data.cuda(), target.cuda()  # data形状: [64, 1, 30, 15, 15]
            
            # 把数据传入模型，得到输出
            outputs, _ = net(data)  # outputs形状: [64, 9]，64个样本，9个类别
            
            # 计算损失（模型预测与真实标签的差距）
            loss = criterion(outputs, target)
            
            if hasattr(net, 'use_kan') and net.use_kan:
                try:
                    reg = net.kan.regularization_loss()
                    loss += 0.0001 * reg  # 权重必须很小！0.005 最稳
                except:
                    pass
            # 清空梯度（上次计算的梯度不要）
            optimizer.zero_grad()
            
            # 反向传播，计算梯度
            loss.backward()
            
            # 更新模型参数
            optimizer.step()
            
            # 累加损失
            total_loss += loss.item()
        
        # 写入日志，记录这一轮的平均损失
        log.write('[Epoch: %d]   [loss avg: %.4f] \n' % (epoch + 1, total_loss / (epoch + 1)))
    
    # 训练完成，写入日志
    log.write('Finished Training')
    
    # 导入thop库，用于计算FLOPs和参数量
    from thop import profile
    
    # 计算模型的FLOPs（计算量，模型做了多少次计算）和参数量(模型有多少个权重)
    # data[0].unsqueeze(dim=0)是把一个样本扩展成batch=1，批次为1，里面只有一个图
    #print("data的形状",data[0].shape)

    flops, params = profile(net, inputs=(profile_input[0].unsqueeze(dim=0),))
    
    # 打印参数量（单位：M百万）
    print('Params = ' + format(str(params / 1000 ** 2), '.6') + 'M')#1000**2：1000²，'.6'保留6位小数，M：百万
    
    # 打印FLOPs（单位：G十亿）
    print('FLOPs = ' + format(str(flops / 1000 ** 3), '.6') + 'G')#G：十亿
    
    # 返回训练好的模型、FLOPs和参数量
    return net, flops, params


# -------------------- 函数8：mytest() - 测试模型 --------------------
def mytest(net, test_loader):
    """
    使用训练好的模型对测试集进行预测
    输入：
        net: 训练好的模型
        test_loader: 测试数据加载器
    返回：预测结果y_pred_test、真实标签y_test、特征y_feature
    """
    
    count = 0  # 标记是否是第一次处理
    net.eval()  # 把模型设为评估模式
    
    y_pred_test = 0  # 存放预测结果
    y_test = 0       # 存放真实标签
    y_feature = 0    # 存放特征
    
    # 遍历测试数据
    for inputs, labels in test_loader:
        # 把数据放到GPU上
        inputs = inputs.cuda()
        
        #训练时不存特征：因为模型在变，特征不稳定，而且数据太大没用
        #测试时必须存特征，因为模型固定了，特征稳定，可复用，能做高级任务
        # 输入模型，得到预测输出和特征
        outputs, features = net(inputs)
        
        """
        计算图有什么用？
        只在训练时有用，
        记录计算路径
        自动求梯度
        让模型学会调整权重w和b
        """
        # 把输出从GPU转到CPU，转成numpy数组
        # detach()是从计算图分离，把张量从计算图拆下来，变成普通数据，才能转numpy，保存、画图
        #cpu()是转到CPU，numpy()是转成numpy
        # argmax(axis=1)找出每行最大值的索引，作为预测类别
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        features = features.detach().cpu().numpy()
        
        # 第一次处理：直接赋值
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            y_feature = features
            count = 1
        # 后续处理：拼接到一起
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
            y_feature = np.concatenate((y_feature, features))
    
    return y_pred_test, y_test, y_feature


# -------------------- 函数9：AA_andEachClassAccuracy() - 计算每类准确率 --------------------
def AA_andEachClassAccuracy(confusion_matrix):
    """
    根据混淆矩阵计算每类的准确率和平均准确率
    输入：confusion_matrix（混淆矩阵）
    返回：each_acc（每类准确率数组）, average_acc（平均准确率）
    """
    
    # 取混淆矩阵的对角线元素（每个类正确分类的数量）
    list_diag = np.diag(confusion_matrix)
    
    # 计算每行的总和（每个类别的真实样本数）
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    
    # 计算每个类的准确率 = 正确数 / 总数
    # nan_to_num处理除以0的情况（把NaN变成0）
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    
    # 计算平均准确率（AA）
    average_acc = np.mean(each_acc)
    
    return each_acc, average_acc


# -------------------- 函数10：acc_reports() - 计算评估指标 --------------------
def acc_reports(y_test, y_pred_test):
    """
    综合计算各种评估指标
    输入：y_test（真实标签）, y_pred_test（预测标签）
    返回：oa（总体准确率）、confusion（混淆矩阵）、每类准确率、aa（平均准确率）、kappa（Kappa系数）
    """
    
    # 计算总体准确率（OA）
    oa = accuracy_score(y_test, y_pred_test)
    
    # 计算混淆矩阵
    confusion = confusion_matrix(y_test, y_pred_test)
    
    # 计算每类准确率和平均准确率
    each_acc, average_acc = AA_andEachClassAccuracy(confusion)
    
    # 计算Kappa系数
    kappa = cohen_kappa_score(y_test, y_pred_test)
    
    # 返回所有指标，乘以100变成百分比形式
    return oa*100, confusion, each_acc*100, average_acc*100, kappa*100


# ==================== 第五部分：主程序入口 ====================

if __name__ == '__main__':
    """
    主程序入口
    流程：创建目录 → 初始化日志 → 循环训练测试 → 保存结果
    """
    import numpy as np
    import time
    import os
    
    # 创建保存模型和日志的目录
    mkdirs(config.checkpoint_path, config.checkpoint_path, config.logs)
    
    # 创建日志记录器
    log = Logger()
    # 打开日志文件（追加模式）
    log.open(config.logs + config.data + '_log.txt', mode='a')
    
    # 创建空列表，用于存放每次测试的结果
    oa = []     # 总体准确率
    acc = []    # 每类准确率
    average_acc = []  # 平均准确率
    kappa = []  # Kappa系数

    
    # 循环test_epoch次（默认5次），每次重新训练和测试
    for num in range(config.test_epoch):
        
        # 每轮开始前释放上一轮的GPU缓存
        if num > 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # 第1步：准备数据（创建DataLoader）
        train_loader, test_loader, y_all, index, all_data_loader, y = create_data_loader()
        
        # 第2步：训练模型，记录开始时间
        tic1 = time.perf_counter()
        net, flops, params = train(train_loader)
        toc1 = time.perf_counter()
        
        # 打印训练时间
        print("训练时间: {:.2f} 秒".format(toc1 - tic1))
        
        # 第3步：测试模型，记录开始时间
        tic2 = time.perf_counter()
        y_pred_test, y_test, y_feature = mytest(net, test_loader)  # y_test形状: (42776,)
        toc2 = time.perf_counter()
        
        # 打印测试时间
        print("测试时间: {:.2f} 秒".format(toc2 - tic2))
        
        # 第4步：计算评估指标
        each_oa, confusion, each_acc, each_aa, each_kappa = acc_reports(y_test, y_pred_test)
        
        # 把结果添加到列表
        oa.append(each_oa)
        acc.append(each_acc)
        average_acc.append(each_aa)
        kappa.append(each_kappa)
        
        # 写入日志
        log.write('Test_Epoch: %.f: Each_OA: %.2f, Each_AA: %.2f, Each_kappa: %.2f \n' 
                  % (num+1, each_oa, each_aa, each_kappa))
        
        # 第5步：生成分类地图
        # 生成保存路径，包含测试轮数和各项指标
        save_path = config.checkpoint_path + 'TestEpoch%.f_OA%.3f_AA%.3f_Kappa%.3f/' % (num+1, each_oa, each_aa, each_kappa)
        
        # 创建保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 调用get_cls_map_PU生成分类地图图片
        get_cls_map_PU.get_cls_map(net, all_data_loader, y, save_path)
        
        # 把每类准确率写入文件
        with open(save_path + "acc.txt", "w") as file:
            for item in each_acc:
                file.write("%s\n" % item)
            file.write("%s\n" % each_oa)    # 写入OA
            file.write("%s\n" % each_aa)    # 写入AA
            file.write("%s\n" % each_kappa)  # 写入Kappa
        
        # 释放本轮的模型和数据，为下一轮腾出空间
        del net, train_loader, test_loader, all_data_loader, y_pred_test, y_test, y_feature
        gc.collect()
    
    acc_np = np.array(acc)  # 转成数组 shape: [test_epoch, num_classes]
    cls_mean_acc = np.mean(acc_np, axis=0)  # 每类均值
    cls_std_acc = np.std(acc_np, axis=0)    # 每类标准差
    cls_var_acc = np.var(acc_np, axis=0)    # 每类方差

    log.write("\n==================== 每类平均准确率 ====================\n")
    for i in range(config.num_classes):
        log.write(f"Class {i+1:2d}: Mean={cls_mean_acc[i]:.2f}, Std={cls_std_acc[i]:.2f}, Var={cls_var_acc[i]:.2f}\n")
    log.write("========================================================\n")

    # 循环结束后，计算平均结果
    log.write('   AVG:   OA: %.2f, std: %.2f, var: %.2f    AA: %.2f, std: %.2f, var: %.2f    Kappa: %.2f, std: %.2f, var: %.2f \n'
              % (np.mean(oa), np.std(oa), np.var(oa),   np.mean(average_acc), np.std(average_acc), np.var(average_acc),     np.mean(kappa), np.std(kappa), np.var(kappa)))

    # 把最终结果写入文件
    with open((config.logs + 'AVG_OA%.3f_AA%.3f_Kappa%.3f.txt' % (np.mean(oa), np.mean(average_acc), np.mean(kappa))), 'w') as file:
        file.write("OA: %.2f, std: %.2f, var: %.2f\n" % (np.mean(oa), np.std(oa), np.var(oa)))
        file.write("AA: %.2f, std: %.2f, var: %.2f\n" % (np.mean(average_acc), np.std(average_acc), np.var(average_acc)))
        file.write("Kappa: %.2f, std: %.2f, var: %.2f\n" % (np.mean(kappa), np.std(kappa), np.var(kappa)))
        file.write("Params: %.4f\n" % (params / 1000 ** 2))
        file.write("Flops: %.4f\n" % (flops / 1000 ** 3))
        file.write("Training time: %.2f\n" % (toc1 - tic1))
        file.write("Testing time: %.2f\n" % (toc2 - tic2))
        file.write("\n==================== 每类平均准确率 ====================\n")
        for i in range(config.num_classes):
            file.write(f"Class {i+1:2d}: Mean={cls_mean_acc[i]:.2f}, Std={cls_std_acc[i]:.2f}, Var={cls_var_acc[i]:.2f}\n")