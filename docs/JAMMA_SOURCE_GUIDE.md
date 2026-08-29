# JamMa 源码

本文按当前仓库源码整理，面向刚开始学习 Python、PyTorch 和计算机视觉的读者。重点回答三个问题：一个函数为什么存在、它接收什么形状的 Tensor、调用后形状和内容如何变化。

> 说明：文中的 `B` 是 batch size，`C` 是通道数，`H/W` 是高/宽，`L=H*W`，`M` 是实际选出的匹配数量。PyTorch 图像 Tensor 通常是 `[B,C,H,W]`，序列 Tensor 通常是 `[B,L,C]`。矩阵不是都表示线性代数矩阵，很多地方只是多维数组。

## 1. 先看总调用链

```text
demo/demo.py
  -> demo/utlis.py::JamMa.forward
      -> src/jamma/backbone.py::CovNextV2_nano.forward
          -> src/convnextv2/convnextv2.py::ConvNeXtV2.forward_features_8
      -> src/jamma/jamma.py::JamMa.forward
          -> coarse_match
              -> KeypointEncoder_wo_score
              -> JointMamba
              -> CoarseMatching
          -> fine_preprocess
              -> inter_fpn / up_conv4 / MLPMixerEncoderLayer
          -> FineSubMatching
```

`demo/utlis.py` 的外层 `JamMa` 只是“组装器”：先让 backbone 把两张图片变成特征，再让 matcher 在 `data` 字典中继续添加匹配结果。这个项目大量使用 `data.update(...)`，所以函数之间不是通过 `return` 传递所有中间结果，而是共同读写同一个字典。

## 2. 一个完整的具体例子

Demo 默认把图片处理成 `[B,3,832,832]`，通常 B=1。项目模型配置是 `resolution=[8,2]`，fine window 是 `5`。

配置有两个来源：正式训练/测试使用 `src/config/default.py` 中的 yacs 配置，其中 fine `D_MODEL=128`；当前 `demo/utlis.py` 为轻量 demo 单独写了字典配置，其中 fine `d_model=64`。因此阅读 demo 的 fine Tensor 时，以 64 为准；阅读训练配置时，以配置文件的 128 为准。coarse 在两处都是 256。

### 2.1 ConvNeXtV2 Nano 的标准结构

源码中的 Nano 参数是 `depths=[2,2,8,2]`、`dims=[80,160,320,640]`。

| 顺序 | 源码模块 | 832 输入时输出 | 224 输入时输出 | 作用 |
|---|---|---:|---:|---|
| 1 | `downsample_layers[0]` / stem | `[B,80,208,208]` | `[B,80,56,56]` | `kernel=4,stride=4`，空间缩小 4 倍 |
| 2 | `stages[0]`，2 个 Block | `[B,80,208,208]` | `[B,80,56,56]` | 保持尺寸，提取特征 |
| 3 | `downsample_layers[1]` | `[B,160,104,104]` | `[B,160,28,28]` | `kernel=2,stride=2` |
| 4 | `stages[1]`，2 个 Block | `[B,160,104,104]` | `[B,160,28,28]` | 保持尺寸 |
| 5 | `downsample_layers[2]` | `[B,320,52,52]` | `[B,320,14,14]` | 标准分类模型中的第三次下采样 |
| 6 | `stages[2]`，8 个 Block | `[B,320,52,52]` | `[B,320,14,14]` | 保持尺寸 |
| 7 | `downsample_layers[3]` | `[B,640,26,26]` | `[B,640,7,7]` | 标准分类模型中的第四次下采样；224 时卷积公式产生 7 |
| 8 | `stages[3]`，2 个 Block | `[B,640,26,26]` | `[B,640,7,7]` | 保持尺寸 |

因此，用户示例里的“第一层 224→112”不适用于这份源码：这里第一层是 224→56，因为 stem 的 stride 是 4，不是 2。用户示例中后面的“每次 stride=2”适用于中间 downsample 层。

### 2.2 JamMa 实际裁剪后的结构

`src/jamma/backbone.py::CovNextV2_nano.__init__` 将 `norm`、`head`、`downsample_layers[2/3]`、`stages[2/3]` 设为 `None`。所以实际只执行前两层：

1. 图片 `[B,3,832,832]`。
2. stem：`[B,3,832,832] -> [B,80,208,208]`。
3. `stages[0]`：保持 `[B,80,208,208]`。
4. `downsample_layers[1]`：`[B,80,208,208] -> [B,160,104,104]`。
5. `stages[1]`：保持 `[B,160,104,104]`。
6. `lin_4`：1x1 卷积 `[B,80,208,208] -> [B,128,208,208]`。
7. `lin_8`：1x1 卷积 `[B,160,104,104] -> [B,256,104,104]`。
8. 两张图片先沿 batch 拼接为 `[2B,3,832,832]`，CNN 一次处理，再用 `.split(B)` 拆回两张图片。

这里的 `feat_4` 是图片的 1/4 特征，`feat_8` 是 1/8 特征。对 832 图片，`h_8=w_8=104`、`hw_8=10816`，`grid_8` 是 `[B,10816,2]`，每行是一个 1/8 网格位置的 `(x,y)` 坐标。

## 3. ConvNeXtV2：每个函数是什么意思

文件：`src/convnextv2/convnextv2.py`，辅助层在 `src/convnextv2/utils.py`。

### `Block.__init__(dim, drop_path)`

创建一个残差块：7x7 depthwise 卷积、LayerNorm、通道扩张 4 倍的 Linear、GELU、GRN、通道缩回，以及残差连接。空间尺寸不变，通道最终仍是 `dim`。

### `Block.forward(x)`

输入 `[B,C,H,W]`。`dwconv` 后仍是 `[B,C,H,W]`；`permute` 变成 `[B,H,W,C]`，因为 Linear 作用在最后一个维度；`pwconv1` 变成 `[B,H,W,4C]`；`pwconv2` 变回 `[B,H,W,C]`；再 permute 回 `[B,C,H,W]`，最后与原输入相加。它不会改变最终形状。

### `ConvNeXtV2.__init__(in_chans, num_classes, depths, dims, ...)`

创建 4 个 downsample 层和 4 个 stage。`depths[i]` 是第 i 个 stage 中 Block 的数量，`dims[i]` 是该 stage 的通道数。最后的 `norm/head` 是分类用途，JamMa 将它们移除。

### `forward_features(x)`

依次执行四组 `downsample -> stage`，然后 `x.mean([-2,-1])` 做全局平均池化：`[B,640,H,W] -> [B,640]`，再交给分类 head。JamMa 不调用它。

### `forward_features_8(x)` / `forward_features_4(x)`

这是 JamMa 真正使用的中间特征接口。前者执行 stem+stage0，并保存 `feat[4]`；再执行 downsample1+stage1，并保存 `feat[8]`。后者只保存 `feat[4]`。注意：这里的数字表示相对原图的尺度，不是通道数。

### `forward(x)`

分类模型的完整入口：`forward_features -> head`，形状 `[B,3,H,W] -> [B,num_classes]`。在 JamMa backbone 中不使用。

### `convnextv2_atto/femto/pico/nano/tiny/base/large/huge(**kwargs)`

这些工厂函数只负责用不同的 `depths/dims` 创建 `ConvNeXtV2`。JamMa 选择 `convnextv2_nano()`。

### `LayerNorm.forward(x)`

支持两种布局。`channels_last` 输入 `[B,H,W,C]`，直接调用 `F.layer_norm`；`channels_first` 输入 `[B,C,H,W]`，沿 C 计算均值方差，输出形状不变。

### `GRN.forward(x)`

输入 `[B,H,W,C]`。对 H、W 求 L2 范数得到 `[B,1,1,C]`，再按通道平均归一化，最后与可学习的 `gamma/beta` 结合。形状不变。

## 4. Backbone 封装

文件：`src/jamma/backbone.py`。

### `CovNextV2_nano.__init__()`

创建 Nano、裁剪掉分类尾部和后两级、下载并加载 ConvNeXt 权重，再创建 `lin_4:80->128` 和 `lin_8:160->256` 两个 1x1 卷积。当前代码会联网下载权重；已有本地缓存时才可离线运行。

### `CovNextV2_nano.forward(data)`

读取 `data["imagec_0"]` 和 `data["imagec_1"]`，拼 batch 后提特征，拆分回 0/1 两张图片。创建 1/8 网格，并把 `bs,c,h_8,w_8,hw_8,feat_4_0/1,feat_8_0/1,grid_8` 写回 `data`。它没有显式 return，真正结果在字典里。

## 5. 主匹配网络

文件：`src/jamma/jamma.py`。

### `JamMa.__init__(config, profiler)`

创建四部分：粗匹配位置编码 `kenc`；4 个 Mamba block 的 `joint_mamba`；粗匹配器；fine 阶段的上采样 FPN、4 个 `MLPMixerEncoderLayer` 和 `FineSubMatching`。

### `coarse_match(data)`

1. `feat_8_0/1` 从 `[B,256,H8,W8]` flatten 成 `[B,256,L]`。
2. `grid_8` `[B,L,2]` 归一化后转为 `[B,2,L]`。
3. `kenc` 将坐标编码成 `[B,256,L]`，加到视觉描述子上。
4. `JointMamba` 在两张图片之间做联合交互，输出形状仍为 `[B,256,L]`。
5. 转置成 `[B,L,256]` 交给 `CoarseMatching`，得到粗匹配索引和置信度。

### `inter_fpn(feat_8, feat_4)`

将 1/8 特征上采样到 1/4：`up2` 的 `[B,256,H8,W8] -> [B,128,H4,W4]`，与 `feat_4` 拼接成 `[B,256,H4,W4]`，卷积后得到 `[B,128,H4,W4]`。再用 `up3` 到 1/2，最终返回 `[B,64,H2,W2]`。空间尺寸扩大约 2 倍两次，通道为 64。

### `fine_preprocess(data, profiler)`

先把两张 `feat_8` 拼成 `[2B,256,H8,W8]`，把 `feat_4` 拼成 `[2B,128,H4,W4]`，通过 FPN 得到 `[2B,64,H2,W2]`。`F.unfold(kernel=5,stride=4)` 为每个位置取 5x5 窗口，单张窗口形状是 `[25,64]`。然后只根据粗匹配的 `(b_ids,i_ids,j_ids)` 选窗口，得到 `feat_f0/1` `[M,64,25]`，送入 MLP-Mixer，最后返回同样形状。

若没有粗匹配，返回两个空 Tensor `[0,25,64]`。

### `forward(data, mode="test")`

写入输入和 coarse 尺寸信息，调用 `coarse_match`，再调用 `fine_preprocess` 和 `fine_matching`。最终匹配点、置信度等仍写入 `data`。

## 6. 粗匹配与细匹配

文件：`src/jamma/matching_module.py`。

### `mask_border(m,b,v)` / `mask_border_with_padding(...)`

把置信度矩阵边缘位置设为 `v`，避免边缘特征参与匹配。粗置信度矩阵可看成 `[B,H0,W0,H1,W1]`；padding 版本还根据有效图像区域计算每张图的边界。

### `compute_max_candidates(p_m0,p_m1)`

从两个 padding mask 统计每张图有效区域的高宽，返回 batch 中可能的最大候选数量总和，主要用于训练时控制显存。

### `generate_random_mask(n,num_true)`

创建长度为 n 的全 False bool Tensor，随机选 `num_true` 个位置置 True。输出 `[n]`。

### `CoarseMatching.forward(feat_c0,feat_c1,data,...)`

输入两个 `[B,L,256]` 描述子。Linear 后用 einsum 计算相似度：`[B,L0,256]` 与 `[B,L1,256]` -> `[B,L0,L1]`。温度缩放、mask 后，训练时对两个方向 softmax；推理时调用 `get_coarse_match_inference`。

### `get_coarse_match_training(...)`

根据 ground truth 和置信度选择训练匹配，写入 `spv_*` 或匹配相关索引。主要输出是粗匹配的 batch/位置索引。

### `get_coarse_match_inference(sim_matrix,data)`

对相似度矩阵执行双向 mutual nearest matching、阈值筛选和边缘移除。输出 `b_ids,i_ids,j_ids,m_bids,mkpts0_c,mkpts1_c,mconf` 等。

### `FineSubMatching.__init__(config,profiler)`

创建 `fine_proj` 和 sub-pixel MLP。fine window `W=5`，所以每个粗匹配对应 `W^2=25` 个局部 token。

### `FineSubMatching.forward(feat_f0_unfold,feat_f1_unfold,data)`

输入 `[M,64,25]`，转成 token 后投影，einsum 得 `[M,25,25]`，两个方向 softmax 相乘得到细置信度矩阵；再调用 `get_fine_sub_match` 选择最可信位置并做亚像素修正。

### `get_fine_sub_match(...)`

阈值筛选细匹配，找到每个窗口的最高置信度位置，把窗口坐标加回粗网格坐标，再用 MLP 预测小数级偏移。最终写入原图坐标 `mkpts0_f/mkpts1_f` 和 `mconf_f`。

## 7. Joint Mamba

文件：`src/jamma/mamba_module.py`。它把二维特征图变成序列交给 `mamba_ssm.Mamba`，再还原回二维特征图。

### `Block.forward(desc)` / `create_block(...)` / `_init_weights(...)`

`create_block` 用 Mamba、LayerNorm/RMSNorm 组装一个 block；`Block.forward` 对输入做 norm，送入 Mamba，再残差相加，形状保持 `[B,L,C]`。`_init_weights` 初始化 Linear/Embedding，并按层数缩放残差分支权重。

### 扫描和还原函数

这些函数共同遵守：输入两张 `[B,C,H,W]` 特征，沿横向、纵向或反向展开成 Mamba 序列，输出通常 `[B,K,L,C]`；`merge_*` 再将它还原为两张 `[B,C,H,W]`。`K` 是扫描方向数。

| 函数 | K | 当前主路径 |
|---|---:|---|
| `scan_jego` / `merge_jego` | 4 | 是；`step_size=2`，交错采样后合并 |
| `scan_jego_seq` / `merge_jego_seq` | 4 | 否；备用序列排列 |
| `scan_vim` / `merge_vim` | 2 | 否；横向正反扫描 |
| `scan_vmamba` / `merge_vmamba` | 4 | 否；横向和纵向正反扫描 |
| `scan_evmamba` / `merge_evmamba` | 4 | 否；带步长的备用实现 |

以 `JointMamba.forward` 为例：`feat_8` 从 `[B,256,H8,W8]` 进入 `scan_jego`，步长 2 后每条序列长度约为 `(H8/2)*(2W8/2)`；返回的 `x` 是 `[B,4,L,256]`。4 个 Mamba 分别处理 `x[:,0]...x[:,3]`，stack 后交给 `merge_jego`，恢复两张 `[B,256,H8,W8]`。最后 `GLU_3` 对两张图片合并的 `[2B,256,H8,W8]` 做通道混合，再拆成两个 `[B,256,L]`。

## 8. 通用匹配层

文件：`src/jamma/utils/utils.py`。

| 函数/类 | 作用和形状 |
|---|---|
| `KeypointEncoder_wo_score.forward` | `[B,2,L] -> [B,d_model,L]`，把坐标编码后加到视觉特征 |
| `normalize_keypoints` | `[B,L,2]` 坐标按图像中心和最大边归一化，形状不变 |
| `TransLN.forward` | `[B,C,L]` 转为 `[B,L,C]` 做 LayerNorm，再转回 |
| `TransLN_2d.forward` | `[B,C,H,W] -> [B,H*W,C]` 归一化后还原，形状不变 |
| `up_conv4.forward` | 两路上采样相加：`[B,Cin,H,W] -> [B,Cout,2H,2W]` |
| `MLP` | 用 1x1 Conv1d 实现逐位置 MLP，长度 L 不变 |
| `GLU.forward` | Linear 门控：最后维度 `dim -> mid_dim -> dim`，形状不变 |
| `GLU_3.forward` | Conv2d 门控，空间和最终通道不变 |
| `conv_3.forward` | 残差 2D 卷积，形状不变 |
| `poolformer.forward` | 残差平均池化，形状不变 |
| `MLPMixerEncoderLayer.forward` | 输入 `[N,L,C]`；先混合 L，再转置混合 C，输出 `[N,L,C]` |

## 9. 数据读取和数据集

文件：`src/utils/dataset.py`、`src/datasets/megadepth.py`、`src/datasets/scannet.py`。

### 图像和尺寸函数

`load_array_from_s3` 从 S3 字节解码图片或 HDF5 depth；`imread_gray/imread_color` 读取本地或 S3 图片；`get_resized_wh` 按最长边计算新尺寸；`get_divisible_wh` 将尺寸调整为可被 df 整除；`pad_bottom_right/pad_bottom_right_c` 把图片补成正方形并可返回 bool mask。

`read_megadepth_color` 输出彩色 Tensor `[3,H,W]`、缩放比例、mask 和 padding 前尺寸；`read_megadepth_gray` 类似但输出灰度；`read_megadepth_depth` 读取深度。`read_scannet_color/gray/depth` 读取 ScanNet 对应数据，`read_scannet_pose` 读位姿，`read_scannet_intrinsic` 读相机内参。

### Dataset 类

`MegaDepthDataset.__len__` 返回 pair 数，`__getitem__` 读取两张图片、深度、内参、相对位姿并组成一个样本字典。`ScanNetDataset.__len__` 同理；`_read_abs_pose` 读取单帧绝对位姿，`_compute_rel_pose` 计算 0 到 1 的相对位姿，`__getitem__` 组成样本。

`RandomConcatSampler.__iter__` 按场景/子数据集采样索引，`__len__` 返回一个 epoch 的样本数。它用于保证训练时不同场景都能被采样。

## 10. 几何监督、损失和指标

### `src/jamma/utils/geometry.py`

`warp_kpts` 输入 `[N,L,2]` 像素点、深度 `[N,H,W]`、位姿 `[N,3,4]`、内参 `[N,3,3]`。流程是取深度 -> 像素反投影到相机 3D `[N,3,L]` -> 刚体变换 -> 投影回另一张图 `[N,L,2]`；同时检查深度非零、是否在图内、深度是否一致，返回 `[N,L]` valid mask 和 `[N,L,2]` warped 点。`warp_kpts_fine` 只是用 `b_ids` 从 batch 中选择粗匹配对应的样本。

### `src/jamma/utils/supervision.py`

`mask_pts_at_padded_regions` 将 padding 区域坐标设为 0。`compute_supervision_coarse` 根据数据集类型调用 `spvs_coarse`；后者生成 coarse 网格、双向 warp、最近邻索引和 ground-truth 置信矩阵 `[N,hw0,hw1]`。`compute_supervision_fine` 调用 `spvs_fine`；后者在每个 `5x5` 窗口内生成 `[M,25,25]` 的细粒度 ground truth。

### `src/losses/loss.py`

`Loss.compute_coarse_loss` 对 coarse confidence matrix 做 focal 类损失；`compute_fine_matching_loss` 对 fine 矩阵计算损失；`compute_sub_pixel_loss` 约束亚像素偏移；`compute_c_weight` 计算粗匹配权重；`forward` 将各项按配置权重相加，返回总损失和日志信息。损失只在训练路径使用。

### `src/utils/metrics.py`

`symmetric_epipolar_distance` 输入两组 `[N,2]` 点和本质矩阵 `[3,3]`，输出每个匹配的误差 `[N]`。`compute_symmetrical_epipolar_errors` 写入 `data['epi_errs']`；`estimate_pose` 用 OpenCV RANSAC，`estimate_lo_pose` 用 PoseLib LO-RANSAC；`compute_pose_errors` 汇总旋转误差、平移误差和 inlier。`error_auc`、`epidist_prec*`、`prec_rec_max_f1` 以及 `aggregate_metrics*` 分别统计 AUC、精度、召回率、F1 和最终测试指标。

## 11. Lightning 训练入口

`train.py::parse_args` 解析配置路径和 Trainer 参数；`main` 读取 yacs 配置、计算真实 batch/lr、创建 `PL_JamMa` 和 `MultiSceneDataModule`，最后 `trainer.fit`。

`test.py::parse_args` 解析测试参数；主程序创建模型、数据模块和 Trainer，调用 `trainer.test`。

`src/lightning/lightning_jamma.py::PL_JamMa._train_inference/_val_inference` 分别执行训练/验证前向；`_compute_metrics` 和 `_compute_metrics_val` 计算指标；`training_step/validation_step/test_step` 是 Lightning 的 batch 入口；`configure_optimizers` 创建优化器和 scheduler；epoch_end 函数汇总日志。

`src/lightning/data.py::MultiSceneDataModule.setup` 创建训练/验证/测试集；`_setup_dataset` 选择 ScanNet 或 MegaDepth；`_build_concat_dataset` 合并多个场景；三个 `*_dataloader` 创建 DataLoader 和 sampler；`_build_dataset` 只是调用给定 Dataset 类的构造函数。

## 12. 配置和其他工具函数速查

`src/config/default.py::get_cfg_defaults` 返回默认配置的 clone，避免修改全局模板。配置重点是 `JAMMA.RESOLUTION=(8,2)`、`COARSE.D_MODEL=256`、`FINE.D_MODEL=128`、`FINE_WINDOW_SIZE=5`。

`src/optimizers/__init__.py::build_optimizer/build_optimizer_tune` 创建 Adam/AdamW；`build_scheduler` 创建 MultiStep、Cosine 或 Exponential scheduler。

`src/utils/augment.py::DarkAug/MobileAug.__call__` 对图片做亮度/移动端风格增强，`build_augmentor` 按名称选择增强器。

`src/utils/dataloader.py::get_local_split` 按 rank 将列表切分给分布式进程。

`src/utils/misc.py::lower_config/upper_config` 在 yacs 配置和普通字典之间转换；`setup_gpus` 解析 GPU 参数；`flattenList` 展平列表；`tqdm_joblib` 将 joblib 进度接到 tqdm。

`src/utils/profiler.py::build_profiler` 创建空 profiler、推理 profiler 或 PyTorch profiler；profiler 只记录耗时，不改变 Tensor。

`src/utils/comm.py` 的 `get_world_size/get_rank/get_local_rank/get_local_size/is_main_process` 查询分布式信息；`synchronize` 同步进程；`all_gather/gather/reduce_dict` 在进程间收集或规约数据；`shared_random_seed` 生成共享随机种子。

`src/utils/plotting.py` 中 `make_confidence_figure`、`make_evaluation_figure_wheel`、`make_matching_figure*` 把 `data` 中的匹配点和置信度画成图片；`draw_kp/vis_matches` 负责画点和连线；`error_colormap/kp_color` 将误差变成颜色。它们主要服务 demo 和验证可视化。

`src/utils/warppers.py` 的 `TensorWrapper` 是让 Tensor 具有统一 `.to/.cpu/.cuda/.detach` 接口的包装；`Pose` 表示位姿，可 `from_Rt/from_aa/from_4x4mat` 创建，`R/t/inv/compose/transform` 读取和运算；`Camera` 表示相机，`from_calibration_matrix` 创建，`project/image2cam/normalize/denormalize` 在 3D、像素和归一化坐标间转换。`src/utils/warppers_utils.py` 提供齐次坐标、旋转和畸变的底层数学函数。

## 12.1 项目自定义函数全集索引

下面只列仓库自己定义的函数名；PyTorch、OpenCV、einops、Lightning 等第三方库的函数不在此重复解释。`__init__`、`forward` 等类方法已在前文按模块解释。

| 文件 | 函数 |
|---|---|
| `src/jamma/mamba_module.py` | `_init_weights`, `create_block`, `scan_jego`, `merge_jego`, `scan_jego_seq`, `merge_jego_seq`, `scan_vim`, `merge_vim`, `scan_vmamba`, `merge_vmamba`, `scan_evmamba`, `merge_evmamba` |
| `src/jamma/matching_module.py` | `mask_border`, `mask_border_with_padding`, `compute_max_candidates`, `generate_random_mask` |
| `src/jamma/utils/geometry.py` | `warp_kpts`, `warp_kpts_fine` |
| `src/jamma/utils/supervision.py` | `mask_pts_at_padded_regions`, `compute_supervision_coarse`, `spvs_coarse`, `compute_supervision_fine`, `spvs_fine` |
| `src/convnextv2/convnextv2.py` | `convnextv2_atto`, `convnextv2_femto`, `convnext_pico`, `convnextv2_nano`, `convnextv2_tiny`, `convnextv2_base`, `convnextv2_large`, `convnextv2_huge` |
| `src/convnextv2/utils.py` | 无独立函数；`LayerNorm`、`GRN` 是类 |
| `src/utils/dataset.py` | `load_array_from_s3`, `imread_gray`, `imread_color`, `get_resized_wh`, `get_divisible_wh`, `pad_bottom_right`, `pad_bottom_right_c`, `read_megadepth_color`, `read_megadepth_gray`, `read_megadepth_depth`, `read_scannet_color`, `read_scannet_gray`, `read_scannet_depth`, `read_scannet_pose`, `read_scannet_intrinsic` |
| `src/utils/metrics.py` | `relative_pose_error`, `symmetric_epipolar_distance`, `compute_symmetrical_epipolar_errors`, `compute_f1`, `estimate_pose`, `estimate_lo_pose`, `compute_pose_errors`, `error_auc`, `epidist_prec`, `epidist_prec_rec`, `epidist_prec_rec_max_f1`, `epidist_prec_rec_max`, `prec_rec_max_f1`, `aggregate_metrics_train_val`, `aggregate_metrics_test`, `aggregate_metrics_f1`, `aggregate_metrics` |
| `src/utils/plotting.py` | `_compute_conf_thresh`, `make_matching_figure_color`, `make_evaluation_figure_color`, `make_colorwheel`, `flow_uv_to_colors`, `coord_trans`, `kp_color`, `draw_kp`, `vis_matches`, `make_evaluation_figure_wheel`, `make_confidence_figure`, `make_matching_figures`, `dynamic_alpha`, `error_colormap` |
| `src/utils/comm.py` | `get_world_size`, `get_rank`, `get_local_rank`, `get_local_size`, `is_main_process`, `synchronize`, `_get_global_gloo_group`, `_serialize_to_tensor`, `_pad_to_largest_tensor`, `all_gather`, `gather`, `shared_random_seed`, `reduce_dict` |
| `src/utils/misc.py` | `lower_config`, `upper_config`, `log_on`, `get_rank_zero_only_logger`, `setup_gpus`, `flattenList`, `tqdm_joblib` |
| `src/utils/warppers_utils.py` | `to_homogeneous`, `from_homogeneous`, `batched_eye_like`, `skew_symmetric`, `transform_points`, `so3exp_map`, `distort_points`, `J_distort_points`, `get_image_coords`, `is_inside` |
| `src/utils/dataloader.py` | `get_local_split` |
| `src/utils/augment.py` | `build_augmentor` |
| `src/utils/profiler.py` | `build_profiler` |
| `src/optimizers/__init__.py` | `build_optimizer`, `build_optimizer_tune`, `build_scheduler` |
| `src/datasets/megadepth.py` | `skew` |
| `src/datasets/sampler.py` | 无独立函数；`RandomConcatSampler` 是类 |
| `src/lightning/data.py` | `_build_dataset` |
| `src/config/default.py` | `get_cfg_defaults` |
| `demo/demo.py`, `train.py`, `test.py` | `main`, `parse_args` |

## 13. 建议的阅读和验证顺序

1. 先运行 `cd demo && python demo.py --help`，只确认入口和参数。
2. 在 `demo/demo.py` 中逐行打印 `imagec_0.shape`。
3. 在 `backbone.py` 的 `data.update` 前打印 `feat_4_0.shape、feat_8_0.shape、grid_8.shape`。
4. 在 `jamma.py` 的 coarse/fine 两处打印形状，重点观察 `[B,C,H,W]`、`[B,C,L]`、`[M,64,25]` 的转换。
5. 再读 `matching_module.py`，因为它负责“从相似度矩阵中选索引”，不是负责提取特征。
6. 最后读几何监督和损失；它们训练时使用真实深度/位姿，demo 推理不需要完整监督数据。

最值得记住的一句话是：`ConvNeXt` 负责把图片变成描述子，`JointMamba` 负责让两张图片的描述子交互，`CoarseMatching` 先找大致对应位置，`FineSubMatching` 再在每个 5x5 小窗口中精修坐标。
