"""
组合模块
"""

import torch
from loguru import logger
from src.jamma.backbone import CovNextV2_nano  # CovNextV2_nano提取图片特征
from src.jamma.jamma import JamMa as JamMa_  # JamMa_比较两张图片的特征
from torch import nn  # PyTorch 的神经网络模块


# JamMa(nn.Module)继承了 PyTorch 的 nn.Module
class JamMa(nn.Module):
    def __init__(self, config, pretrained="official") -> None:
        super().__init__()
        """
        ConvNeXt V2 Nano主要负责从两张图片中提取 1/4 尺度的特征 | 1/8 尺度的特征
        创建 backbone 时会从 GitHub 下载预训练权重 convnextv2_nano_pretrain.ckpt
        """
        self.backbone = CovNextV2_nano()  # 创建特征提取网络, save到self.backbone变量
        self.matcher = JamMa_(config)  # 创建JamMa网络, save到self.matcher变量

        # 加载GitHub开源的JamMa权重
        if pretrained == "official":
            state_dict = torch.hub.load_state_dict_from_url(
                "https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt",
                file_name="jamma.ckpt",
            )[
                "state_dict"
            ]  # 取出权重字典的"state_dict"键对应的值, 这个值是一个字典, 里面包含了模型的参数和缓冲区
            """
            把权重加载到当前这个外层 JamMa 模型中
            JamMa
            ├── backbone: CovNextV2_nano
            └── matcher: JamMa_
            state_dict理应满足这种结构, 否则会报错
            """
            self.load_state_dict(state_dict, strict=True)
            logger.info("Load Official JamMa Weight")
        elif pretrained:
            state_dict = torch.load(pretrained, map_location="cpu")["state_dict"]
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained}' as pretrained checkpoint")

    """
    jamma(data)实现调用， __call__ 里做的事情大致包括：
    检查是否处于训练/评估模式
    执行注册的前向钩子（register_forward_pre_hook / register_forward_hook）
    调用你写的 forward
    处理梯度上下文
    """

    def forward(self, data):  # 网络前向传播
        self.backbone(data)  # backbone修改data->补充成特征数据包
        return self.matcher(data)  # data被补充成匹配结果


cfg = {
    "coarse": {
        "d_model": 256,  # 粗粒度特征维度是多少
    },
    "fine": {
        "d_model": 64,  # 细粒度特征维度是多少
        "dsmax_temperature": 0.1,
        "thr": 0.1,  # 匹配阈值是多少
        "inference": True,
    },
    "match_coarse": {
        "thr": 0.2,
        "use_sm": True,
        "border_rm": 2,
        "dsmax_temperature": 0.1,
        "inference": True,
    },
    "fine_window_size": 5,  # 窗口大小是多少
    "resolution": [8, 2],
}  # 字典定义参数模型
