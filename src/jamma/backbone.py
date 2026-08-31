"""
把两张原始图片转换成 JamMa 后续匹配所需要的多尺度特征
"""

import torch
from einops import rearrange  # 调整 Tensor 的形状
from kornia.utils import create_meshgrid  # 创建图片上的坐标网格
from src.convnextv2.convnextv2 import (
    convnextv2_nano,  # 导入 ConvNeXtV2 nano 模型->负责“看图片、提取特征”的网络
)
from torch import nn


class CovNextV2_nano(nn.Module):
    def __init__(self):
        """
        将convnextv2_nano从分类器改造为特征提取器Backbone
        """
        super().__init__()  # 初始化 PyTorch 的基础神经网络功能
        # 创建完整 Nano 后，下面几行会把分类网络裁剪成特征提取器。
        self.cnn = convnextv2_nano()  # ConvNeXt V2 Nano 网络对象
        self.cnn.norm = None  # 移除最后的全局 LayerNorm
        self.cnn.head = None  # 移除 Linear 分类头

        self.cnn.downsample_layers[2] = None  # 移除第2层的下采样层
        self.cnn.downsample_layers[3] = None  # 移除第3层的下采样层
        self.cnn.stages[2] = None  # 移除第2阶段网络
        self.cnn.stages[3] = None  # 移除第3阶段网络
        """
        state_dict = {
        'cnn.downsample_layers.0.0.weight': tensor([...]),  # 卷积层权重
        'cnn.downsample_layers.0.0.bias': tensor([...]),    # 卷积层偏置
        'cnn.stages.0.0.dwconv.weight': tensor([...]),      # 深度卷积权重
        'cnn.stages.0.0.norm.weight': tensor([...]),        # 归一化层权重
        'cnn.stages.0.0.norm.bias': tensor([...]),          # 归一化层偏置
        ...  # 总共几百个键值对，覆盖整个网络的每一层参数
}
        """
        state_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/leoluxxx/JamMa/releases/download/v0.1/convnextv2_nano_pretrain.ckpt",
            file_name="convnextv2_nano_pretrain.ckpt",
        )  # PyTorch 提供的一个工具函数，专门用来从网址下载预训练权重文件并加载

        self.cnn.load_state_dict(state_dict, strict=True)

        self.lin_4 = nn.Conv2d(80, 128, 1)  # 输入80通道 → 输出128通道(阶段0)
        self.lin_8 = nn.Conv2d(160, 256, 1)  # 输入160通道 → 输出256通道(阶段1)

    def forward(self, data):
        """
        feature_pyramid[4]: [8, 80, 56, 56] ← 8张图,80通道,56*56分辨率
                ↓
        self.lin_4(...)    : [8, 128, 56, 56] ← 1*1卷积,80→128通道
                ↓
        .split(B) 其中 B=4  : 沿第0维拆成2份,每份4张
                ↓
        feat_4_0: [4, 128, 56, 56] ← 前4张(原 imagec_0)的特征
        feat_4_1: [4, 128, 56, 56] ← 后4张(原 imagec_1)的特征
        """
        # imagec_0/imagec_1: [B,3,H,W]；两张图片合成 [2B,3,H,W] 一次提取特征。
        B, _, H, W = data["imagec_0"].shape
        x = torch.cat(
            [data["imagec_0"], data["imagec_1"]], 0
        )  # 在第0维（batch维度）拼接

        # 返回 {4: [2B,80,H/4,W/4], 8: [2B,160,H/8,W/8]}字典
        feature_pyramid = self.cnn.forward_features_8(x)
        # 1×1卷积，160→256通道
        feat_8_0, feat_8_1 = self.lin_8(feature_pyramid[8]).split(B)
        # 1×1卷积，80→128通道
        feat_4_0, feat_4_1 = self.lin_4(feature_pyramid[4]).split(B)

        """
        生成坐标网格: 
        常见套路是：网络先输出 grid * scale 作为粗定位，再预测一个 (Δx, Δy) 偏移量做精修。
        粗到细（coarse-to-fine）是标准操作。
        create_meshgrid生成压缩后的网格坐标
        """
        scale = 8
        h_8, w_8 = H // scale, W // scale
        device = data["imagec_0"].device
        grid = [
            rearrange(
                (create_meshgrid(h_8, w_8, False, device) * scale).squeeze(0),
                "h w t->(h w) t",
            )
        ] * B  # kpt_xy
        grid_8 = torch.stack(grid, 0)

        data.update(
            {
                "bs": B,  # 批次大小
                "c": feat_8_0.shape[1],  # 尺度8的通道数
                "h_8": h_8,  # 尺度8特征图高度
                "w_8": w_8,  # 尺度8特征图宽度
                "hw_8": h_8 * w_8,  # 特征图总像素数
                "feat_8_0": feat_8_0,  # 图片0的8倍下采样特征
                "feat_8_1": feat_8_1,  # 图片1的8倍下采样特征
                "feat_4_0": feat_4_0,  # 图片0的4倍下采样特征
                "feat_4_1": feat_4_1,  # 图片1的4倍下采样特征
                "grid_8": grid_8,  # 每个特征点对应的原图坐标
            }
        )  # data.update() 是在往输入字典里添加新的键值对, 把前面计算好的所有特征和辅助信息都存进去，方便后续模块使用。
