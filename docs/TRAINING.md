

**训练**

**数据集设置**

通常，训练 JamMa 需要两部分数据：原始数据集（即 ScanNet 和 MegaDepth）以及离线生成的数据集索引。数据集索引存储了每个数据集中用于训练/验证/测试的场景、图像对和其他元数据。对于 MegaDepth 数据集，用于训练的图像对之间的相对位姿直接缓存在索引文件中。然而，由于生成的文件大小过大，ScanNet 图像对的相对位姿并未存储。

**下载数据集**

**MegaDepth**

我们使用原始 MegaDepth 数据集中提供的深度图，以及由 D2-Net 预处理的无畸变图像、对应的相机内参和外参。你可以从以下链接分别下载。

- MegaDepth 无畸变图像和处理后的深度图  
  注意：我们仅使用深度图。  
  下载数据的路径将记为 `/path/to/megadepth`

- D2-Net 预处理图像  
  由于 MegaDepth 提供的无畸变图像不附带对应的内参，D2-Net 中手动对图像进行了无畸变处理。  
  下载数据的路径将记为 `/path/to/megadepth_d2net`

**ScanNet**

请按照官方指南设置 ScanNet 数据集。

注意：我们使用 Python 导出的数据，而非 C++ 导出的数据。

**下载数据集索引**

你可以从以下链接下载所需的数据集索引。下载后，解压所需文件。

```bash
unzip downloaded-file.zip

# 提取数据集索引
tar xf train-data/megadepth_indices.tar
tar xf train-data/scannet_indices.tar

# 提取测试数据（可选）
tar xf testdata/megadepth_test_1500.tar
tar xf testdata/scannet_test_1500.tar
```

**构建数据集符号链接**

我们将数据集符号链接到项目主目录下的 `data` 目录。

```bash
# scannet
# -- # 训练和测试数据集
ln -s /path/to/scannet_train/* /path/to/project/data/scannet/train
ln -s /path/to/scannet_test/* /path/to/project/data/scannet/test
# -- # 数据集索引
ln -s /path/to/scannet_indices/* /path/to/project/data/scannet/index

# megadepth
# -- # 训练和测试数据集（训练和测试共用同一数据集）
ln -sv /path/to/megadepth/phoenix /path/to/megadepth_d2net/Undistorted_SfM /path/to/project/data/megadepth/train
ln -sv /path/to/megadepth/phoenix /path/to/megadepth_d2net/Undistorted_SfM /path/to/project/data/megadepth/test
# -- # 数据集索引
ln -s /path/to/megadepth_indices/* /path/to/project/data/megadepth/index
```