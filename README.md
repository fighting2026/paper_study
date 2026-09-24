# HMTF — 高光谱图像分类

本仓库是论文 **HMTF**（Hyperspectral Image Classification）的官方实现代码。

方法以 **3D 卷积** 作为光谱-空间特征提取前端，主干结合 **Mamba 状态空间模型** 与 **Transformer**
两种建模方式（`Parallel MT` / `Interval MT` / `Series MT` / `Series TM` 等多种组合），
并引入 **双曲几何（Lorentz 流形）** 与 **模糊融合** 模块。

---

## 目录结构

```
.
├── main.py               # 主程序：数据加载 → PCA → 切块 → 训练 → 测试 → 出图
├── config.py             # 所有超参数与数据集开关
├── HMTF.py               # 模型实现（VisionMamba 主干 + 3D 卷积 + 双曲/模糊模块）
├── get_cls_map_PU.py     # 分类图（classification map）绘制
├── shuangqu.py           # 双曲几何：Lorentz 流形与相关算子
├── fuzzy.py              # 模糊层 FuzzyLayer3
├── models/
│   ├── __init__.py
│   └── csms6s.py         # 选择性扫描（selective scan）底层算子
├── utils/
│   └── utils.py          # Logger、mkdirs 等工具
└── data/                 # 数据集（通过 Git LFS 提供）
    ├── pu/               # Pavia University
    ├── pc/               # Pavia Centre
    ├── hus/              # Houston 2013
    └── LongKou/          # WHU-Hi-LongKou
```

---

## 环境准备

### 1. 必须先安装 Git LFS

数据集通过 **Git LFS** 存储。**不安装 git-lfs 直接 clone，`.mat` 文件只会得到 130 字节左右的指针文本，代码无法运行。**

```bash
git lfs install
git clone https://github.com/fighting2026/paper_study.git
cd paper_study
```

如果已经 clone 过但忘记装 LFS，在仓库目录下执行 `git lfs pull` 即可补下数据。

### 2. 安装依赖

需要 **NVIDIA GPU + CUDA**（代码中大量 `.cuda()` 调用，以及 `mamba_ssm` 的 CUDA 算子）。

```bash
pip install -r requirements.txt
```

> `mamba-ssm` 需要与你的 PyTorch / CUDA 版本匹配，编译时间较长。
> 若安装报错，请参考 [mamba-ssm 官方说明](https://github.com/state-spaces/mamba) 选择对应版本。

### 3. (可选) 预训练权重

`HMTF.py` 中 `MODEL_PATH` 用于加载 ImageNet 预训练权重，**本仓库的实验默认不使用预训练权重**，
保持 `MODEL_PATH = '自己实际的权重路径'` 原样即可。如需加载，请自行下载 VideoMamba 权重并修改该路径。

---

## 数据集

本仓库通过 Git LFS 提供 **4 个**公开高光谱数据集。切换数据集只需修改 `config.py` 中的
`data` 与 `num_classes`：

| `config.py` 的 `data` | `num_classes` | 数据目录 | 文件 | 图像尺寸 (H×W×Bands) |
|---|---|---|---|---|
| `'PaviaU'`   | 9  | `data/pu/`       | `PaviaU.mat` / `PaviaU_gt.mat`                     | 610 × 340 × 103 |
| `'PaviaC'`   | 9  | `data/pc/`       | `Pavia.mat` / `Pavia_gt.mat`                       | 1096 × 715 × 102 |
| `'Houston2013'` | 15 | `data/hus/`    | `Houston.mat` / `Houston_gt.mat`                   | 349 × 1905 × 144 |
| `'LongKou'`  | 9  | `data/LongKou/`  | `WHU_Hi_LongKou.mat` / `WHU_Hi_LongKou_gt.mat`     | 550 × 400 × 270 |

⚠️ **注意命名不一致**：`PaviaU` 的数据放在 `data/pu/`（不是 `data/PaviaU/`），
这是仓库的历史目录命名，`main.py` 中已按此路径写死，请勿重命名目录。

代码里还保留了 `'Salinas'` 分支，但 **Salinas 数据集未随仓库提供**，使用前需自行下载并放到
`data/Salinas/`（文件名：`Salinas_corrected.mat`、`Salinas_gt.mat`）。

### 数据来源

以上数据集均为公开基准数据，版权归原作者所有。本仓库只是把它们转成了 `.mat` 格式并随代码一起提供，
**如需引用请遵循各数据集的原始许可与引用要求**。也可以直接从官方来源自行下载：

| 数据集 | 官方来源 | 需要下载的文件 / 变量名 |
|---|---|---|
| Pavia University (`data/pu/`) | [UPV/EHU 高光谱数据集页](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | `PaviaU.mat`（变量 `paviaU`）、`PaviaU_gt.mat`（变量 `paviaU_gt`） |
| Pavia Centre (`data/pc/`) | 同上 | `Pavia.mat`（变量 `pavia`）、`Pavia_gt.mat`（变量 `pavia_gt`） |
| Salinas（代码支持、仓库未附带） | 同上 | `Salinas_corrected.mat`、`Salinas_gt.mat`，放到 `data/Salinas/` |
| Houston 2013 (`data/hus/`) | [2013 IEEE GRSS Data Fusion Contest](https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest)（旧地址：[hyperspectral.ee.uh.edu](https://hyperspectral.ee.uh.edu/?page_id=459)） | 需在官网填表免费申请；本仓库使用 **cloud-free** 版本，变量名为 `Houston` / `Houston_GT`（注意 `GT` 大写） |
| WHU-Hi-LongKou (`data/LongKou/`) | [WHU-Hi 数据集分享页](http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) | `WHU_Hi_LongKou.mat`（变量 `WHU_Hi_LongKou`）、`WHU_Hi_LongKou_gt.mat` |

> **在网页上浏览本仓库时，`.mat` 文件只会显示成 130 字节左右的 LFS 指针文本，看不到真实数据**，
> 这是 Git LFS 的正常行为（真实数据在 LFS 存储里，clone 时自动还原）。
> 如果你是通过在线匿名镜像（如 anonymous.4open.science）查看代码，请按上表从官方来源下载数据。

各数据集标签文件（ground truth）的类别编号与 `main.py` 中的处理逻辑一致，无需再做转换。

---

## 运行

```bash
python main.py
```

程序会按 `config.py` 的 `test_epoch` 次重复「训练 + 测试」，并输出：

- **日志**：`checkpoint/<数据集>/.../<数据集>_log.txt`
- **每类精度**：各测试轮次目录下的 `acc.txt`
- **分类图**：各测试轮次目录下的 `PU_SSPredictions.png`
- **汇总结果**：`checkpoint/<数据集>/.../AVG_OA<OA>_AA<AA>_Kappa<Kappa>.txt`

---

## 主要超参数（`config.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `patch_size` | 15 | 输入图像块边长 |
| `pca_components` | 30 | PCA 降维后的波段数 |
| `train_samples_per_class` | 15 | 每类训练样本数，其余作测试 |
| `train_epoch` | 150 | 训练轮数 |
| `test_epoch` | 5 | 重复训练/测试次数（用于统计均值与标准差） |
| `BATCH_SIZE_TRAIN` | 64 | 批大小（大尺寸数据集自动降到 32） |
| `model_type` | `'Parallel MT'` | 主干组合方式，可选 `Parallel MT` / `Interval MT` / `Series MT` / `Series TM` / `Series Mamba-Transformer` 等 |
| `depth` / `embed_dim` / `d_state` / `ssm_ratio` | 4 / 32 / 16 / 1 | 模型规模相关 |
| `use_hyperbolic` | `True` | 是否启用双曲几何模块 |
| `use_fuzzy` | `False` | 是否启用模糊融合模块 |
| `gpus` | `'0'` | 使用的 GPU 编号 |

图像较大时（总像素 > 15 万或标签像素 > 2 万，如 LongKou）程序会自动切换到
`LazyPatchDataset` 按需切图模式，避免内存溢出。

---

## 引用

如果本代码对你的研究有帮助，请引用：

```bibtex
@article{hmtf2026,
  title   = {TODO: 论文标题},
  author  = {TODO: 作者列表},
  journal = {TODO: 期刊 / 会议},
  year    = {2026}
}
```

---

## 致谢

模型实现参考并改写了以下开源项目，特此致谢：

- [VideoMamba](https://github.com/OpenGVLab/VideoMamba) — Mamba 主干与 `models/csms6s.py` 选择性扫描算子
- [HSI-MFormer](https://github.com/tubingnuist/HSI-MFormer) — 高光谱分类整体框架与 `get_cls_map_PU.py` 出图脚本
  （Y. He, B. Tu, B. Liu, J. Li and A. Plaza, "HSI-MFormer: Integrating Mamba and Transformer Experts for Hyperspectral Image Classification," *IEEE TGRS*, vol. 63, 2025, doi: 10.1109/TGRS.2025.3564167）
- [timm](https://github.com/huggingface/pytorch-image-models) / DeiT — 部分模块来自其实现

---

## 许可

本项目采用 [MIT License](LICENSE)。
