"""
Boundary-Distance Weighted Change Detection (BDWCD) Module

【终极统一架构】
基于二进制边界等价性理论，彻底统一了 BCD 与 SCD：
1. BCD 是仅有 背景(0) 和 前景(1) 的特殊情况。
2. SCD 是具有 背景(0) 和 多类前景(1..C) 的一般情况。
后处理只需遍历所有前景类别 c ∈ [1, num_classes)，将预测二值化后，统一使用边界距离与 Logit 过滤，
即可在无分支的情况下完美覆盖所有任务。
"""

import numpy as np
from scipy.ndimage import (
    distance_transform_edt, binary_opening,
    label as cc_label, generate_binary_structure,
)
from typing import Dict, Union


class BoundaryDistanceChangeFilter:
    def __init__(
        self,
        # ---- Boundary Distance ----
        boundary_distance_tau: Union[float, dict] = 5.0,   
        boundary_sigmoid_scale: float = 2.0,  
        # ---- Component-level decision ----
        component_level: bool = True,         
        interior_ratio_threshold: float = 0.2, 
        # ---- Morphological opening ----
        use_morphological_opening: bool = True,
        opening_kernel_size: int = 3,
        opening_iterations: int = 1,
        # ---- Small region removal ----
        use_small_region_removal: bool = True,
        min_change_area: Union[int, dict] = 100,
        # ---- Soft XOR (Logit Margin) ----
        logit_margin: float = 0.0,
        # ---- Debug ----
        verbose: bool = False,
        **kwargs,
    ):
        self.boundary_distance_tau = boundary_distance_tau
        self.boundary_sigmoid_scale = boundary_sigmoid_scale
        self.component_level = component_level
        self.interior_ratio_threshold = interior_ratio_threshold
        self.use_morphological_opening = use_morphological_opening
        self.opening_kernel_size = opening_kernel_size
        self.opening_iterations = opening_iterations
        self.use_small_region_removal = use_small_region_removal
        self.min_change_area = min_change_area
        self.logit_margin = logit_margin
        self.verbose = verbose

    def __call__(
        self,
        change_mask: np.ndarray,          # [H, W] bool (由全局语义指标生成的候选区域)
        change_score: np.ndarray,         # [H, W] float 
        pred_t1: np.ndarray,              # [H, W] int 
        pred_t2: np.ndarray,              # [H, W] int 
        logits_t1: np.ndarray = None,     # [C, H, W] float
        logits_t2: np.ndarray = None,
    ) -> Dict[str, np.ndarray]:
        
        H, W = pred_t1.shape
        # 自动推断类别数。BCD时 max_pred 通常为1，num_classes=2；SCD时会更大。
        num_classes = logits_t1.shape[0] if logits_t1 is not None else int(max(pred_t1.max(), pred_t2.max()) + 1)
        
        per_class_change = {}
        all_change = np.zeros((H, W), dtype=bool)

        # 核心：无论是 BCD (循环 1 次) 还是 SCD (循环 C-1 次)，全部用统一的二值逻辑处理
        for c in range(1, num_classes):
            tau_c = self.boundary_distance_tau.get(c, 3.0) if isinstance(self.boundary_distance_tau, dict) else self.boundary_distance_tau
            ma_c = self.min_change_area.get(c, 30) if isinstance(self.min_change_area, dict) else self.min_change_area

            # 1. 为类别 c 生成二值掩码 (语义隔离)
            m_t1 = (pred_t1 == c)
            m_t2 = (pred_t2 == c)

            # 2. 候选变化：该类别的状态发生翻转，并且全局 JS 散度允许它变化
            change_c = change_mask & (m_t1 != m_t2)

            # 3. Soft XOR：抑制模型不自信的微小概率反转
            if self.logit_margin > 0 and logits_t1 is not None and logits_t2 is not None:
                logit_diff = np.abs(logits_t1[c] - logits_t2[c])
                change_c = change_c & (logit_diff > self.logit_margin)

            if not change_c.any():
                per_class_change[c] = np.zeros((H, W), dtype=bool)
                continue

            # 4. 基于二值掩码提取无方向边界 (数学上等价于有方向的距离验证)
            b_t1 = self._find_label_boundaries(m_t1)
            b_t2 = self._find_label_boundaries(m_t2)
            
            # 计算无方向距离场
            d_t1 = distance_transform_edt(~b_t1)
            d_t2 = distance_transform_edt(~b_t2)
            reliability = np.minimum(d_t1, d_t2)

            # 5. 空间距离加权与连通域决策
            if tau_c > 0:
                weight_c = self._sigmoid((reliability - tau_c) / self.boundary_sigmoid_scale)
                change_c = self._apply_component_or_pixel_decision(change_c, weight_c)

            # 6. 形态学与小噪点去除
            change_c = self._apply_morphology(change_c)
            change_c = self._remove_small_regions(change_c, ma_c)

            per_class_change[c] = change_c
            all_change |= change_c

        return {
            'change_mask': all_change,
            'change_score': change_score * all_change.astype(np.float32),
            'per_class_change': per_class_change
        }

    # ================================================================
    # Helper Functions
    # ================================================================
    def _apply_component_or_pixel_decision(self, mask: np.ndarray, weight: np.ndarray) -> np.ndarray:
        if not self.component_level:
            return mask & (weight > 0.5)
            
        core_mask = mask & (weight > 0.5)
        struct_8 = generate_binary_structure(2, 2)
        labeled, num_cc = cc_label(mask, structure=struct_8)
        
        mask_out = np.zeros_like(mask)
        for k in range(1, num_cc + 1):
            comp = (labeled == k)
            if (comp & core_mask).sum() / comp.sum() >= self.interior_ratio_threshold:
                mask_out[comp] = True
        return mask_out

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        if self.use_morphological_opening and mask.any():
            k = self.opening_kernel_size
            struct = np.ones((k, k), dtype=bool)
            return binary_opening(mask, structure=struct, iterations=self.opening_iterations)
        return mask

    def _remove_small_regions(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        if min_area <= 0 or not mask.any():
            return mask
        labeled, num_features = cc_label(mask)
        if num_features == 0: return mask
        
        sizes = np.bincount(labeled.ravel())
        small = np.where(sizes < min_area)[0]
        small = small[small > 0]
        if len(small) > 0:
            mask = mask & ~np.isin(labeled, small)
        return mask

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    @staticmethod
    def _find_label_boundaries(pred: np.ndarray) -> np.ndarray:
        # 这个操作对于 bool 矩阵同样完美适用
        boundary = np.zeros_like(pred, dtype=bool)
        boundary[:-1, :] |= (pred[:-1, :] != pred[1:, :])
        boundary[1:, :]  |= (pred[:-1, :] != pred[1:, :])
        boundary[:, :-1] |= (pred[:, :-1] != pred[:, 1:])
        boundary[:, 1:]  |= (pred[:, :-1] != pred[:, 1:])
        return boundary