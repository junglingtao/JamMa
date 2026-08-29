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
        super().__init__()  # 初始化 PyTorch 的基础神经网络功能
        self.cnn = convnextv2_nano()  # ConvNeXt V2 Nano 网络对象
        self.cnn.norm = None  # 移除归一化层
        self.cnn.head = None  # 移除分类头

        self.cnn.downsample_layers[2] = None  # 移除第2层的下采样层
        self.cnn.downsample_layers[3] = None  # 移除第3层的下采样层
        self.cnn.stages[2] = None  # 移除第2阶段网络
        self.cnn.stages[3] = None  # 移除第3阶段网络

        state_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/leoluxxx/JamMa/releases/download/v0.1/convnextv2_nano_pretrain.ckpt",
            file_name="convnextv2_nano_pretrain.ckpt",
        )
        self.cnn.load_state_dict(state_dict, strict=True)

        self.lin_4 = nn.Conv2d(80, 128, 1)
        self.lin_8 = nn.Conv2d(160, 256, 1)

    def forward(self, data):
        B, _, H, W = data["imagec_0"].shape
        x = torch.cat([data["imagec_0"], data["imagec_1"]], 0)
        feature_pyramid = self.cnn.forward_features_8(x)
        feat_8_0, feat_8_1 = self.lin_8(feature_pyramid[8]).split(B)
        feat_4_0, feat_4_1 = self.lin_4(feature_pyramid[4]).split(B)

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
                "bs": B,
                "c": feat_8_0.shape[1],
                "h_8": h_8,
                "w_8": w_8,
                "hw_8": h_8 * w_8,
                "feat_8_0": feat_8_0,
                "feat_8_1": feat_8_1,
                "feat_4_0": feat_4_0,
                "feat_4_1": feat_4_1,
                "grid_8": grid_8,
            }
        )
