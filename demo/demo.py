"""
读取两张图片，交给 JamMa 神经网络寻找对应点，然后生成两张可视化结果图。
"""

import argparse  # 用于读取命令行参数
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # PyTorch 的函数工具集合
from loguru import logger  # 用于工程项目打印日志

# __file__->当前目录，.parents[1]向上找两级目录，同理0是当前目录，1是上一级目录，2是上两级目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(PROJECT_ROOT)
)  # 将PROJECT_ROOT插入到搜索路径的【最前面】（索引0）

from src.utils.dataset import read_megadepth_color  # 读取 MegaDepth 数据集的彩色图像
from src.utils.plotting import (  # 导入两个画图函数，生成生成viz1.png和viz2.png
    make_confidence_figure,
    make_evaluation_figure_wheel,
)
from utlis import JamMa, cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(  # 创建一个命令行参数解析器
        description="Image pair matching with JamMa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(  # 图片1参数
        "--image1",
        type=str,
        default=str(PROJECT_ROOT / "assets/figs/345822933_b5fb7b6feb_o.jpg"),
        help="Path to the source image",
    )
    parser.add_argument(  # 图片2参数
        "--image2",
        type=str,
        default=str(PROJECT_ROOT / "assets/figs/479605349_8aa68e066d_o.jpg"),
        help="Path to the target image",
    )
    parser.add_argument(  # 输出目录参数
        "--output_dir", type=str, default="output/", help="Path of the outputs"
    )

    opt = parser.parse_args()  # 读取你在终端输入的参数
    Path(opt.output_dir).mkdir(exist_ok=True, parents=True)  # 创建输出目录

    device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择 CPU 或 GPU

    # 根据配置 cfg 创建 JamMa 神经网络->切换到测试模式 eval()->将模型移动到指定设备（CPU 或 GPU）
    jamma = JamMa(config=cfg).eval().to(device)

    """
    image0、image1 即 PyTorch Tensor->[1, 3, H, W]
    scale0、scale1 即图片缩放比例->从原始尺寸缩放到新尺寸时的比例
    mask0、mask1 即图片掩码, 原始832 x 600->832 x 832, mask会标记新增区域
    prepad_size0、prepad_size1 即填充之前的图片尺寸->后续如果需要把匹配点映射回原图，就可能使用这个信息
    """
    image0, scale0, mask0, prepad_size0 = read_megadepth_color(  # 读取第一张图片
        opt.image1,  # 读入jpg
        832,
        16,
        True,  # 图片路径、调整后的最大尺寸、尺寸必须能被 16 整除、是否进行填充
    )
    image1, scale1, mask1, prepad_size1 = read_megadepth_color(
        opt.image2, 832, 16, True
    )
    # 检查 mask 是否存在，图片尺寸改变必须要有mask，否则无法训练
    if mask0 is None or mask1 is None:
        raise RuntimeError("Image masks are required when padding is enabled.")
    """
    缩小mask:
    mask0[None, None]->假设原来的 mask0 形状是[H, W], 加两个 None 后，形状变成[1, 1, H, W]
    [1, 1, H, W]->[batch, channel, height, width] | batch, 一次处理多少张图片 | channel, 通道数W
    """
    mask0 = F.interpolate(
        mask0[None, None].float(),  # tensor要转成浮点数参与计算
        scale_factor=0.125,  # 缩放比例0.125->缩小到原来的 1/8
        mode="nearest",  # 最近邻插值法, 因为mask 只有两种状态0/1
        recompute_scale_factor=False,  # 按照给定的 0.125 使用缩放比例，不重新计算比例
    )[0].bool()  # [1, 1, 新H, 新W]->[1, 新H, 新W], 把浮点数重新转换成布尔值
    mask1 = F.interpolate(
        mask1[None, None].float(),
        scale_factor=0.125,
        mode="nearest",
        recompute_scale_factor=False,
    )[0].bool()
    data = {  # 待计算的矩阵传入GPU
        "imagec_0": image0.to(device),
        "imagec_1": image1.to(device),
        "mask0": mask0.to(device),
        "mask1": mask1.to(device),
    }

    logger.info(
        f"Matching: {opt.image1} and {opt.image2}"
    )  # 打印日志，显示正在匹配的两张图片
    jamma(data)  # 执行模型匹配, 相当于jamma.forward(data), 跑模型测试
    logger.info("Finish Matching, Visualizing")

    # topk:可视化前100个匹配点, dpi为分辨率
    make_confidence_figure(data, path=opt.output_dir + "viz1.png", dpi=300, topk=50)
    make_evaluation_figure_wheel(data, path=opt.output_dir + "viz2.png", topk=50)
    logger.info("Done")
