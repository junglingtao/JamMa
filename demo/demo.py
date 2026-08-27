"""
读取两张图片，交给 JamMa 神经网络寻找对应点，然后生成两张可视化结果图。
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataset import read_megadepth_color
from src.utils.plotting import make_confidence_figure, make_evaluation_figure_wheel
from utlis import JamMa, cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image pair matching with JamMa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image1",
        type=str,
        default=str(PROJECT_ROOT / "assets/figs/345822933_b5fb7b6feb_o.jpg"),
        help="Path to the source image",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default=str(PROJECT_ROOT / "assets/figs/479605349_8aa68e066d_o.jpg"),
        help="Path to the target image",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/", help="Path of the outputs"
    )

    opt = parser.parse_args()
    Path(opt.output_dir).mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    jamma = JamMa(config=cfg).eval().to(device)

    image0, scale0, mask0, prepad_size0 = read_megadepth_color(
        opt.image1, 832, 16, True
    )
    image1, scale1, mask1, prepad_size1 = read_megadepth_color(
        opt.image2, 832, 16, True
    )
    if mask0 is None or mask1 is None:
        raise RuntimeError("Image masks are required when padding is enabled.")

    mask0 = F.interpolate(
        mask0[None, None].float(),
        scale_factor=0.125,
        mode="nearest",
        recompute_scale_factor=False,
    )[0].bool()
    mask1 = F.interpolate(
        mask1[None, None].float(),
        scale_factor=0.125,
        mode="nearest",
        recompute_scale_factor=False,
    )[0].bool()
    data = {
        "imagec_0": image0.to(device),
        "imagec_1": image1.to(device),
        "mask0": mask0.to(device),
        "mask1": mask1.to(device),
    }

    logger.info(f"Matching: {opt.image1} and {opt.image2}")
    jamma(data)
    logger.info("Finish Matching, Visualizing")

    make_confidence_figure(data, path=opt.output_dir + "viz1.png", dpi=300, topk=4000)
    make_evaluation_figure_wheel(data, path=opt.output_dir + "viz2.png", topk=4000)
    logger.info("Done")
