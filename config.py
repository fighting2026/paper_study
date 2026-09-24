
"""
参数	                    说明	                    示例值
data	                   数据集名称         'PaviaU', 'Indian', 'Houston2018', 'Houston2013'
num_classes	                类别数	                 PaviaU=9, Indian=16
patch_size	               图像块大小	                   15
pca_components	          PCA降维维度	                   30
train_epoch	               训练轮数	                      100
test_epoch	               测试轮数（重复训练次数）	        5
BATCH_SIZE_TRAIN	        批次大小	                   64
model_type	                模型架构	       'Parallel MT', 'Series Mamba-Transformer'
gpus	                    GPU编号	                        '0'
"""

#配置文件：所有参数都在这里改
class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 1e-2
    # training parameters
    train_epoch = 150
    test_epoch = 5
    BATCH_SIZE_TRAIN = 64
    norm_flag = True
    gpus = '0'
    train_samples_per_class =15
    # source data information
    # #pu
    # data = 'PaviaU'  # PaviaU-9-103-0.95 / Indian-16-200-0.9  / Houston2018-21  / Houston2013-15-0.95
    # num_classes = 9

    #houston
    data = 'Houston2013'  # PaviaU-9-103-0.95 / Indian-16-200-0.9  / Houston2018-21  / Houston2013-15-0.95
    num_classes = 15
  
    # #LongKou
    # data = 'LongKou'  # PaviaU-9-103-0.95 / Indian-16-200-0.9  / Houston2018-21  / Houston2013-15-0.95
    # num_classes = 9

    # data='PaviaC'
    # num_classes=9


    patch_size = 15
    pca_components = 30
    # model
    model_type = 'Parallel MT'   # 'Parallel MT'  'Interval MT'  'Series MT'  'Series TM'  'Parallel Transformer-Mamba'  'Series Transformer-Mamba'  'Series Mamba-Transformer'
    depth = 4
    embed_dim =32
    d_state = 16
    ssm_ratio = 1
    pos = False
    cls = False
    # 3DConv parameters
    conv3D_channel = 32
    conv3D_kernel_1 = (5, 5, 5)  #(5, 5, 5)
    conv3D_kernel_2 = (7, 7, 7)  #(7, 7, 7)
    conv3D_kernel_3 = (9, 9, 9)  #(9, 9, 9)
    dim_patch = patch_size - conv3D_kernel_1[1] + 1  # 8
    dim_linear_1 = pca_components - conv3D_kernel_1[0] + 1  # 28
    dim_linear_2 = pca_components - conv3D_kernel_2[0] + 1  # 28
    dim_linear_3 = pca_components - conv3D_kernel_3[0] + 1  # 28
    # paths information
    checkpoint_path = ('./' + "checkpoint/" + data + '/' + model_type + '_TrainEpoch' + str(train_epoch) + '_TestEpoch' + str(test_epoch) + '_Batch' + str(BATCH_SIZE_TRAIN)\
                      + '/PatchSize' + str(patch_size) \
                      + '/'  + 'Depth' + str(depth) + '_embed' + str(embed_dim) + '_dstate' + str(d_state) + '_ratio' + str(ssm_ratio)
                      + '_3Dconv' + str(conv3D_channel) + '&' + str(conv3D_kernel_1) + '&' + str(conv3D_kernel_2) + '&' + str(conv3D_kernel_3) + '/')
    logs = checkpoint_path

    # 双曲几何开关：True=开启，False=关闭（保持原来模型）
    use_hyperbolic =True
    use_fuzzy =False # True=开启模糊融合, False=关闭
    fuzzy_num=14# 推荐8~16
    # fuzzy_num=14

config = DefaultConfigs()

