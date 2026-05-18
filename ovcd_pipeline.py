"""
Open-Vocabulary Change Detection Pipeline (OVCD)

统一后的 Pipeline：
无论是 SCD(多类语义变化) 还是 BCD(二分类变化)，都遵循统一路径：
1. 提取全局语义变化 (Dual-Head Semantic Change)
2. Presence-Guided 加权滤波 (PGTF)
3. 阈值化生成 Base Change Mask
4. 传递给实例过滤器 (BDWCD)，由它根据配置决定是全局过滤(BCD)还是逐类过滤(SCD)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChangeDetectionResult:
    """变更检测结果的统一数据结构"""
    # 二值变化掩码 [H, W], bool
    change_mask: torch.Tensor
    # 变化强度热图 [H, W], float in [0, 1]
    change_score_map: torch.Tensor
    # From-To 语义变化图: pred_t1[H, W], pred_t2[H, W]
    pred_t1: torch.Tensor
    pred_t2: torch.Tensor
    # 各类别的变化分数 [num_classes]
    presence_delta: torch.Tensor
    class_change_scores: torch.Tensor
    # For SCD / per_class_bcd mode: dict of class_id -> change_mask[H,W]
    per_class_change: Optional[Dict[int, np.ndarray]] = None


class OpenVocabularyChangeDetector:
    def __init__(
        self,
        class_names: List[str],
        device: str = 'cuda',
        use_semantic_cd: bool = True,
        use_presence_filtering: bool = True,
        change_threshold: float = 0.3,
        presence_alpha: float = 1.0,
        bg_idx: int = 0,
        instance_filter = None, # 统一的后处理模块 (BDWCD)
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device
        self.use_semantic_cd = use_semantic_cd
        self.use_presence_filtering = use_presence_filtering
        self.change_threshold = change_threshold
        self.presence_alpha = presence_alpha
        self.bg_idx = bg_idx
        self.instance_filter = instance_filter

    def detect_change(
        self,
        result_t1: Dict[str, torch.Tensor],
        result_t2: Dict[str, torch.Tensor],
        image_t1: Optional[Image.Image] = None,
        image_t2: Optional[Image.Image] = None,
    ) -> ChangeDetectionResult:
        """
        统一入口：基于语义分布和边界后处理检测变化
        """
        seg_logits_t1 = result_t1['seg_logits_cls']  # [C, H, W]
        seg_logits_t2 = result_t2['seg_logits_cls']
        presence_t1 = result_t1['presence_scores_cls']  # [C]
        presence_t2 = result_t2['presence_scores_cls']

        C, H, W = seg_logits_t1.shape

        # ========== Step 1: Semantic Change Detection ==========
        # 利用 JS 散度和 Logit 差异生成高质量的全局变化热图
        semantic_change_map = self._semantic_change_detection(
            seg_logits_t1, seg_logits_t2
        )  # [H, W]

        # ========== Step 2: Presence-Guided Temporal Filtering ==========
        presence_delta = torch.abs(presence_t2 - presence_t1)  # [C]
        
        # if self.use_presence_filtering:
        #     semantic_change_map = self._apply_presence_filtering(
        #         semantic_change_map, seg_logits_t1, seg_logits_t2,
        #         presence_t1, presence_t2, presence_delta
        #     )

        # ========== Step 3: Generation Base Map ==========
        # pred_t1 = seg_logits_t1.argmax(dim=0)  # [H, W]
        # pred_t2 = seg_logits_t2.argmax(dim=0)
        pred_t1 = result_t1['pred_label']  # 已经经过概率阈值过滤的最终预测标签
        pred_t2 = result_t2['pred_label']  # 已经经过概率阈值过滤的最终预测标签

        change_score_map = self._normalize_map(semantic_change_map)
        change_mask = change_score_map > self.change_threshold
        per_class_change = None

        # ========== Step 4: Spatial Filtering (BDWCD) ==========
        # 这里实例过滤器会根据它被配置成的 mode ('global' or 'per_class') 自动处理
        if self.instance_filter is not None:
            _filter_out = self.instance_filter(
                change_mask=change_mask.cpu().numpy(),
                change_score=change_score_map.cpu().numpy(),
                pred_t1=pred_t1.cpu().numpy(),
                pred_t2=pred_t2.cpu().numpy(),
                logits_t1=seg_logits_t1.cpu().numpy(),
                logits_t2=seg_logits_t2.cpu().numpy(),
            )
            change_mask = torch.from_numpy(_filter_out['change_mask']).to(self.device)
            change_score_map = torch.from_numpy(_filter_out['change_score']).to(self.device).float()
            per_class_change = _filter_out.get('per_class_change', None)

        return ChangeDetectionResult(
            change_mask=change_mask,
            change_score_map=change_score_map,
            pred_t1=pred_t1,
            pred_t2=pred_t2,
            presence_delta=presence_delta,
            class_change_scores=presence_delta.clone(),
            per_class_change=per_class_change,
        )

    def _semantic_change_detection(self, logits_t1, logits_t2) -> torch.Tensor:
        pred_t1 = logits_t1.argmax(dim=0)
        pred_t2 = logits_t2.argmax(dim=0)
        hard_change = (pred_t1 != pred_t2).float()

        prob_t1 = F.softmax(logits_t1, dim=0)
        prob_t2 = F.softmax(logits_t2, dim=0)
        m = 0.5 * (prob_t1 + prob_t2)

        kl_pm = (prob_t1 * (prob_t1.log() - m.log())).sum(dim=0)
        kl_qm = (prob_t2 * (prob_t2.log() - m.log())).sum(dim=0)
        js_div = (0.5 * (kl_pm + kl_qm)).clamp(min=0)

        logit_diff = (logits_t2 - logits_t1).abs().max(dim=0)[0]
        change_map = hard_change * (0.5 * js_div + 0.5 * self._normalize_map(logit_diff))
        return change_map

    def _apply_presence_filtering(self, change_map, logits_t1, logits_t2, pres_t1, pres_t2, pres_delta):
        pred_t1 = logits_t1.argmax(dim=0)
        pred_t2 = logits_t2.argmax(dim=0)
        
        pres_t1_pixel = pres_t1[pred_t1]
        pres_t2_pixel = pres_t2[pred_t2]
        noise_mask = (pres_t1_pixel < 0.3) & (pres_t2_pixel < 0.3)

        delta_t1_pixel = pres_delta[pred_t1]
        delta_t2_pixel = pres_delta[pred_t2]
        presence_weight = torch.max(delta_t1_pixel, delta_t2_pixel)

        # enhanced_change = change_map * (1 + self.presence_alpha * presence_weight)
        enhanced_change = change_map.clone()
        enhanced_change[noise_mask] = 0
        return enhanced_change

    def _normalize_map(self, x: torch.Tensor) -> torch.Tensor:
        x_min, x_max = x.min(), x.max()
        return (x - x_min) / (x_max - x_min) if (x_max - x_min > 1e-8) else torch.zeros_like(x)