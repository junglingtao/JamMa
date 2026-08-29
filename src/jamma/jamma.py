"""
主模型/主匹配网络
"""

import torch
import torch.nn.functional as F
from einops.einops import rearrange
from src.jamma.mamba_module import JointMamba
from src.jamma.matching_module import CoarseMatching, FineSubMatching
from src.jamma.utils.utils import (
    KeypointEncoder_wo_score,
    MLPMixerEncoderLayer,
    normalize_keypoints,
    up_conv4,
)
from src.utils.profiler import PassThroughProfiler
from torch import nn

torch.backends.cudnn.deterministic = True
INF = 1e9

# 这个文件是匹配网络的“总导演”：
# 先做 coarse（粗）匹配，再从粗匹配位置裁出局部窗口做 fine（细）匹配。
# 注意：data 是一个共享字典，很多函数通过 data.update() 传递中间结果。


class JamMa(nn.Module):
    def __init__(self, config, profiler=None):
        # config 决定特征通道数、匹配阈值和 fine 窗口大小。
        super().__init__()
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.d_model_c = self.config["coarse"]["d_model"]
        self.d_model_f = self.config["fine"]["d_model"]

        self.kenc = KeypointEncoder_wo_score(
            self.d_model_c, [32, 64, 128, self.d_model_c]
        )
        self.joint_mamba = JointMamba(
            self.d_model_c,
            4,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            profiler=self.profiler,
        )
        self.coarse_matching = CoarseMatching(config["match_coarse"], self.profiler)

        self.act = nn.GELU()
        dim = [256, 128, 64]
        self.up2 = up_conv4(dim[0], dim[1], dim[1])  # 1/8 -> 1/4
        self.conv7a = nn.Conv2d(2 * dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.conv7b = nn.Conv2d(dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.up3 = up_conv4(dim[1], dim[2], dim[2])  # 1/4 -> 1/2
        self.conv8a = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)
        self.conv8b = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)

        W = self.config["fine_window_size"]
        self.fine_enc = nn.ModuleList(
            [MLPMixerEncoderLayer(2 * W**2, 64) for _ in range(4)]
        )
        self.fine_matching = FineSubMatching(config, self.profiler)

    def coarse_match(self, data):
        # 输入 feat_8_*: [B, C, H8, W8]；先把二维网格摊平成 L=H8*W8 个位置。
        desc0, desc1 = data["feat_8_0"].flatten(2, 3), data["feat_8_1"].flatten(2, 3)
        kpts0, kpts1 = data["grid_8"], data["grid_8"]
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data["imagec_0"].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data["imagec_1"].shape[-2:])

        # kenc 需要 [B, 2, L]，所以将 [B, L, 2] 的坐标转置。
        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        data.update(
            {
                "feat_8_0": desc0,
                "feat_8_1": desc1,
            }
        )

        # Mamba 让两张图片的 coarse 描述子互相交换信息，形状保持不变。
        with self.profiler.profile("coarse interaction"):
            self.joint_mamba(data)

        # 3. match coarse-level
        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"].flatten(-2), data["mask1"].flatten(-2)

        with self.profiler.profile("coarse matching"):
            self.coarse_matching(
                data["feat_8_0"].transpose(1, 2),
                data["feat_8_1"].transpose(1, 2),
                data,
                mask_c0=mask_c0,
                mask_c1=mask_c1,
            )

    def inter_fpn(self, feat_8, feat_4):
        # FPN：把低分辨率的强语义特征逐级放大，并和浅层特征融合。
        d2 = self.up2(feat_8)  # 1/4
        # 两个输入都是 1/4 尺度，沿通道维拼接：[B,128,H4,W4]x2 -> [B,256,H4,W4]。
        d2 = self.act(self.conv7a(torch.cat([feat_4, d2], 1)))
        feat_4 = self.act(self.conv7b(d2))

        d1 = self.up3(feat_4)  # 1/2
        d1 = self.act(self.conv8a(d1))
        feat_2 = self.conv8b(d1)
        return feat_2

    def fine_preprocess(self, data, profiler):
        # fine 阶段只处理 coarse 已经选出的 M 对位置，而不是处理整张图。
        data["resolution1"] = 8
        stride = data["resolution1"] // self.config["resolution"][1]
        W = self.config["fine_window_size"]
        # 两张图沿 batch 拼接：[B,C,H8,W8]x2 -> [2B,C,H8,W8]。
        feat_8 = torch.cat([data["feat_8_0"], data["feat_8_1"]], 0).view(
            2 * data["bs"], data["c"], data["h_8"], -1
        )
        feat_4 = torch.cat([data["feat_4_0"], data["feat_4_1"]], 0)

        if data["b_ids"].shape[0] == 0:
            feat0 = torch.empty(0, W**2, self.d_model_f, device=feat_4.device)
            feat1 = torch.empty(0, W**2, self.d_model_f, device=feat_4.device)
            return feat0, feat1

        # feat_f = self.inter_fpn(feat_8, feat_4, feat_2)
        feat_f = self.inter_fpn(feat_8, feat_4)
        feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        data.update({"hw0_f": feat_f0.shape[2:], "hw1_f": feat_f1.shape[2:]})

        # 1. unfold：从整张 fine 特征图提取所有 W*W 局部窗口。
        pad = 0 if W % 2 == 0 else W // 2
        feat_f0_unfold = F.unfold(
            feat_f0, kernel_size=(W, W), stride=stride, padding=pad
        )
        feat_f0_unfold = rearrange(feat_f0_unfold, "n (c ww) l -> n l ww c", ww=W**2)
        feat_f1_unfold = F.unfold(
            feat_f1, kernel_size=(W, W), stride=stride, padding=pad
        )
        feat_f1_unfold = rearrange(
            feat_f1_unfold, "n (c ww) l -> n l ww c", ww=W**2
        )  # [b, h_f/stride * w_f/stride, w*w, c]

        # 2. 只保留 coarse 索引对应的窗口，避免无谓的显存开销。
        feat_f0_unfold = feat_f0_unfold[data["b_ids"], data["i_ids"]]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data["b_ids"], data["j_ids"]]  # [n, ww, cf]

        # 每对窗口先拼成 [M,2*W^2,C]，转成 Mixer 使用的 [M,C,2*W^2]。
        feat_f = torch.cat([feat_f0_unfold, feat_f1_unfold], 1).transpose(1, 2)
        for layer in self.fine_enc:
            feat_f = layer(feat_f)
        feat_f0_unfold, feat_f1_unfold = feat_f[:, :, : W**2], feat_f[:, :, W**2 :]
        return feat_f0_unfold, feat_f1_unfold

    def forward(self, data, mode="test"):
        # 整个推理顺序：记录尺寸 -> coarse -> fine -> 将结果写回 data。
        self.mode = mode
        data.update(
            {
                "hw0_i": data["imagec_0"].shape[2:],
                "hw1_i": data["imagec_1"].shape[2:],
                "hw0_c": [data["h_8"], data["w_8"]],
                "hw1_c": [data["h_8"], data["w_8"]],
            }
        )

        self.coarse_match(data)

        with self.profiler.profile("fine matching"):
            # 4. fine-level matching module
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(data, self.profiler)

            # 5. match fine-level and sub-pixel refinement
            self.fine_matching(
                feat_f0_unfold.transpose(1, 2), feat_f1_unfold.transpose(1, 2), data
            )
