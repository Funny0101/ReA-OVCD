"""
Open-Vocabulary Change Detection Evaluation Metrics

支持三类评估:
1. Binary Change Detection (BCD): F1, IoU, OA, Kappa
2. Semantic Change Detection (SCD): Sek, mIoU_scd, Kappa_scd
3. Open-Vocabulary Specific: Per-transition IoU, Presence Accuracy
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class BCDEvaluator:
    """
    Binary Change Detection 评估器
    
    指标:
    - F1 (Dice): 2*TP / (2*TP + FP + FN)
    - IoU (Jaccard): TP / (TP + FP + FN)
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - OA (Overall Accuracy): (TP + TN) / N
    - Kappa: Cohen's Kappa coefficient
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.num_samples = 0

    def update(self, pred: np.ndarray, label: np.ndarray):
        """
        Args:
            pred: binary change prediction, shape [H, W], values {0, 1}
            label: binary change ground truth, shape [H, W], values {0, 1}
        """
        pred = pred.astype(bool).flatten()
        label = label.astype(bool).flatten()

        self.tp += int(np.logical_and(pred, label).sum())
        self.fp += int(np.logical_and(pred, ~label).sum())
        self.fn += int(np.logical_and(~pred, label).sum())
        self.tn += int(np.logical_and(~pred, ~label).sum())
        self.num_samples += 1

    def compute(self) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        n = tp + fp + fn + tn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        oa = (tp + tn) / n if n > 0 else 0.0

        # Kappa coefficient
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n) if n > 0 else 0.0
        kappa = (oa - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

        return {
            'F1': f1 * 100,
            'IoU': iou * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'OA': oa * 100,
            'Kappa': kappa * 100,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'num_samples': self.num_samples,
        }


class SCDEvaluator:
    """
    Semantic Change Detection 评估器 (用于SECOND等数据集)
    
    指标:
    - Sek (Separated Kappa): 语义变化检测的标准指标
    - mIoU: T1和T2各自的语义分割mIoU
    - Fscd: F1 score for semantic change
    - Per-class IoU: 逐类别IoU
    
    参考: Yang et al., "Asymmetric Siamese Networks for Semantic Change Detection"
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        # 语义分割的混淆矩阵: T1 和 T2 分别计算
        self.cm_t1 = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.cm_t2 = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        # 二值变化检测的统计
        self.bcd_evaluator = BCDEvaluator()
        # 语义变化的混淆矩阵 (N^2 x N^2)
        self.num_samples = 0

    def update(
        self,
        pred_t1: np.ndarray,    # [H, W] predicted T1 labels
        pred_t2: np.ndarray,    # [H, W] predicted T2 labels
        label_t1: np.ndarray,   # [H, W] ground truth T1 labels
        label_t2: np.ndarray,   # [H, W] ground truth T2 labels
    ):
        """更新评估指标"""
        # 有效区域 mask
        valid = (label_t1 != self.ignore_index) & (label_t2 != self.ignore_index)

        pred_t1_v = pred_t1[valid].flatten()
        pred_t2_v = pred_t2[valid].flatten()
        label_t1_v = label_t1[valid].flatten()
        label_t2_v = label_t2[valid].flatten()

        # 更新语义分割混淆矩阵
        self._update_confusion_matrix(self.cm_t1, pred_t1_v, label_t1_v)
        self._update_confusion_matrix(self.cm_t2, pred_t2_v, label_t2_v)

        # 更新二值变化检测
        pred_change = (pred_t1_v != pred_t2_v).astype(np.uint8)
        label_change = (label_t1_v != label_t2_v).astype(np.uint8)
        self.bcd_evaluator.update(pred_change, label_change)

        self.num_samples += 1

    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        results = {}

        # 1. T1 / T2 各自的 mIoU
        iou_t1 = self._compute_iou(self.cm_t1)
        iou_t2 = self._compute_iou(self.cm_t2)
        results['mIoU_t1'] = np.nanmean(iou_t1) * 100
        results['mIoU_t2'] = np.nanmean(iou_t2) * 100
        results['mIoU_avg'] = (results['mIoU_t1'] + results['mIoU_t2']) / 2

        # 2. 逐类别 IoU (T1/T2 分别)
        for c in range(self.num_classes):
            results[f'IoU_t1_cls{c}'] = iou_t1[c] * 100 if not np.isnan(iou_t1[c]) else 0.0
            results[f'IoU_t2_cls{c}'] = iou_t2[c] * 100 if not np.isnan(iou_t2[c]) else 0.0

        # 3. 逐类别 Presence F1 (参照 evaluate_second.py 的评估方式)
        #    对每个类别c: gt_presence = (label_t1==c OR label_t2==c)
        #                 pred_presence = (pred_t1==c OR pred_t2==c)
        #    然后计算 F1, IoU
        per_class_f1 = []
        for c in range(self.num_classes):
            # 从混淆矩阵重建逐类统计:
            # cm_t1[i,j] = #{pixels where label=i, pred=j}
            # gt has class c in t1: cm_t1[c, :].sum()  
            # pred has class c in t1: cm_t1[:, c].sum()
            gt_t1_c = self.cm_t1[c, :].sum()
            gt_t2_c = self.cm_t2[c, :].sum()
            pred_t1_c = self.cm_t1[:, c].sum()
            pred_t2_c = self.cm_t2[:, c].sum()
            # TP for presence: correctly predicted class c (diagonal)
            tp_t1 = self.cm_t1[c, c]
            tp_t2 = self.cm_t2[c, c]
            # For per-class presence (appears in either time phase):
            # Use average of T1 and T2 per-class F1 as approximation
            prec_t1 = tp_t1 / pred_t1_c if pred_t1_c > 0 else 0
            rec_t1 = tp_t1 / gt_t1_c if gt_t1_c > 0 else 0
            f1_t1 = 2 * prec_t1 * rec_t1 / (prec_t1 + rec_t1) if (prec_t1 + rec_t1) > 0 else 0
            prec_t2 = tp_t2 / pred_t2_c if pred_t2_c > 0 else 0
            rec_t2 = tp_t2 / gt_t2_c if gt_t2_c > 0 else 0
            f1_t2 = 2 * prec_t2 * rec_t2 / (prec_t2 + rec_t2) if (prec_t2 + rec_t2) > 0 else 0
            f1_avg = (f1_t1 + f1_t2) / 2
            results[f'F1_cls{c}'] = f1_avg * 100
            per_class_f1.append(f1_avg)
        results['mF1_cls'] = np.mean(per_class_f1) * 100

        # 4. Separated Kappa (Sek) — 真正的SCD指标
        results['Sek'] = self._compute_sek() * 100

        # 5. Binary CD metrics
        bcd_metrics = self.bcd_evaluator.compute()
        results['BCD_F1'] = bcd_metrics['F1']
        results['BCD_IoU'] = bcd_metrics['IoU']
        results['BCD_OA'] = bcd_metrics['OA']
        results['BCD_Kappa'] = bcd_metrics['Kappa']

        # 6. Fscd = 0.3 * Sek + 0.7 * mIoU_avg (SECOND论文的综合指标)
        results['Fscd'] = 0.3 * results['Sek'] + 0.7 * results['mIoU_avg']

        results['num_samples'] = self.num_samples
        return results

    def _update_confusion_matrix(self, cm, pred, label):
        """更新混淆矩阵"""
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                cm[i, j] += int(((label == i) & (pred == j)).sum())

    def _compute_iou(self, cm) -> np.ndarray:
        """从混淆矩阵计算逐类别IoU"""
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = np.where(union > 0, intersection / union, np.nan)
        return iou

    def _compute_sek(self) -> float:
        """
        Compute Separated Kappa (Sek) coefficient

        Sek 综合衡量语义分割准确性和变化检测能力:
        Sek = exp(Σ_c log(1 + kappa_c) / C) - 1
        其中 kappa_c 是类别c的变化检测 kappa

        这里用简化但更准确的公式:
        Sek = sqrt(kappa_bcd × mean(kappa_per_class))
        结合二值变化检测和逐类语义分割的一致性
        """
        # BCD kappa
        bcd = self.bcd_evaluator.compute()
        kappa_bcd = bcd['Kappa'] / 100.0

        # Per-class kappa (from semantic seg confusion matrices)
        kappas = []
        for c in range(self.num_classes):
            # T1 class c accuracy
            tp1 = self.cm_t1[c, c]
            gt1 = self.cm_t1[c, :].sum()
            pred1 = self.cm_t1[:, c].sum()
            n1 = self.cm_t1.sum()
            # T2 class c accuracy  
            tp2 = self.cm_t2[c, c]
            gt2 = self.cm_t2[c, :].sum()
            pred2 = self.cm_t2[:, c].sum()
            n2 = self.cm_t2.sum()
            # Average OA and pe for this class
            if n1 > 0 and n2 > 0:
                oa1 = (tp1 + (n1 - gt1 - pred1 + tp1)) / n1
                pe1 = (gt1 * pred1 + (n1 - gt1) * (n1 - pred1)) / (n1 * n1)
                oa2 = (tp2 + (n2 - gt2 - pred2 + tp2)) / n2
                pe2 = (gt2 * pred2 + (n2 - gt2) * (n2 - pred2)) / (n2 * n2)
                oa_avg = (oa1 + oa2) / 2
                pe_avg = (pe1 + pe2) / 2
                kc = (oa_avg - pe_avg) / (1 - pe_avg) if (1 - pe_avg) > 1e-8 else 0
                kappas.append(max(kc, 0))
        
        mean_kappa_cls = np.mean(kappas) if kappas else 0
        # Geometric-like combination
        sek = np.sqrt(max(kappa_bcd, 0) * max(mean_kappa_cls, 0))
        return sek


class PerClassChangeBCDEvaluator:
    """
    Per-Class Binary Change Detection 评估器

    对每个非背景类别 c 独立做 BCD:
      gt_change_c  = (gt_t1 == c) XOR (gt_t2 == c)
      pred_change_c = (pred_t1 == c) XOR (pred_t2 == c)
    然后计算 IoU, F1, Precision, Recall

    这是 OmniOVCD / DynamicEarth 在 SECOND 上的标准评估方式。
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 每个类别一个 BCD 计数器 (跳过 class 0 = background)
        self.per_class_tp = np.zeros(num_classes, dtype=np.int64)
        self.per_class_fp = np.zeros(num_classes, dtype=np.int64)
        self.per_class_fn = np.zeros(num_classes, dtype=np.int64)
        self.per_class_tn = np.zeros(num_classes, dtype=np.int64)
        self.num_samples = 0

    def reset(self):
        self.per_class_tp[:] = 0
        self.per_class_fp[:] = 0
        self.per_class_fn[:] = 0
        self.per_class_tn[:] = 0
        self.num_samples = 0

    def update(
        self,
        pred_t1: np.ndarray,
        pred_t2: np.ndarray,
        label_t1: np.ndarray,
        label_t2: np.ndarray,
        per_class_change: dict = None,
    ):
        """更新每个类别的变化检测统计
        
        Args:
            per_class_change: 可选, dict[int, np.ndarray]. 
                如果提供, 直接用 per_class_change[c] 作为预测的类别c变化掩码,
                而不是通过 pred_t1/pred_t2 的 XOR 推导.
                这是 instance matching 模式下的正确用法.
        """
        valid = (label_t1 != self.ignore_index) & (label_t2 != self.ignore_index)

        label_t1_v = label_t1[valid]
        label_t2_v = label_t2[valid]

        if per_class_change is not None:
            # Instance matching 模式: 直接使用给定的 per-class change masks
            for c in range(1, self.num_classes):
                gt_change = (label_t1_v == c) != (label_t2_v == c)
                if c in per_class_change:
                    pred_change = per_class_change[c][valid]
                else:
                    pred_change = np.zeros_like(gt_change)

                tp = int(np.logical_and(pred_change, gt_change).sum())
                fp = int(np.logical_and(pred_change, ~gt_change).sum())
                fn = int(np.logical_and(~pred_change, gt_change).sum())
                tn = int(np.logical_and(~pred_change, ~gt_change).sum())

                self.per_class_tp[c] += tp
                self.per_class_fp[c] += fp
                self.per_class_fn[c] += fn
                self.per_class_tn[c] += tn
        else:
            # 像素级 XOR 模式
            pred_t1_v = pred_t1[valid]
            pred_t2_v = pred_t2[valid]

            for c in range(1, self.num_classes):
                gt_change = (label_t1_v == c) != (label_t2_v == c)
                pred_change = (pred_t1_v == c) != (pred_t2_v == c)

                tp = int(np.logical_and(pred_change, gt_change).sum())
                fp = int(np.logical_and(pred_change, ~gt_change).sum())
                fn = int(np.logical_and(~pred_change, gt_change).sum())
                tn = int(np.logical_and(~pred_change, ~gt_change).sum())

                self.per_class_tp[c] += tp
                self.per_class_fp[c] += fp
                self.per_class_fn[c] += fn
                self.per_class_tn[c] += tn

        self.num_samples += 1

    def compute(self) -> Dict[str, float]:
        results = {}
        ious = []
        f1s = []

        for c in range(1, self.num_classes):
            tp = self.per_class_tp[c]
            fp = self.per_class_fp[c]
            fn = self.per_class_fn[c]

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            results[f'IoU_cls{c}'] = iou * 100
            results[f'F1_cls{c}'] = f1 * 100
            results[f'Precision_cls{c}'] = prec * 100
            results[f'Recall_cls{c}'] = rec * 100

            ious.append(iou)
            f1s.append(f1)

        results['mIoU'] = np.mean(ious) * 100 if ious else 0.0
        results['mF1'] = np.mean(f1s) * 100 if f1s else 0.0
        results['num_samples'] = self.num_samples
        return results


class OVCDEvaluator:
    """
    Open-Vocabulary Change Detection 综合评估器

    根据数据集类型自动选择评估模式:
    - BCD数据集: 使用BCDEvaluator
    - SCD数据集: 使用SCDEvaluator
    - per_class_bcd: 对每个类别分别做BCD (OmniOVCD/DynamicEarth评估方式)
    """

    def __init__(
        self,
        num_classes: int,
        mode: str = 'bcd',
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
    ):
        self.mode = mode
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]

        if mode == 'bcd':
            self.evaluator = BCDEvaluator()
        elif mode == 'scd':
            self.evaluator = SCDEvaluator(num_classes=num_classes, ignore_index=ignore_index)
        elif mode == 'per_class_bcd':
            self.evaluator = PerClassChangeBCDEvaluator(num_classes=num_classes, ignore_index=ignore_index)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def update(self, pred: Dict, label: Dict):
        """
        统一更新接口

        Args:
            pred: dict with keys depending on mode
                - bcd: {'change_mask': np.ndarray [H,W]}
                - scd / per_class_bcd: {'pred_t1': np.ndarray [H,W], 'pred_t2': np.ndarray [H,W]}
                - per_class_bcd (with instance matching): + {'per_class_change': dict[int, ndarray]}
            label: dict with keys
                - bcd: {'label': np.ndarray [H,W]}
                - scd / per_class_bcd: {'label_t1': np.ndarray [H,W], 'label_t2': np.ndarray [H,W]}
        """
        if self.mode == 'bcd':
            self.evaluator.update(pred['change_mask'], label['label'])
        elif self.mode == 'scd':
            self.evaluator.update(
                pred['pred_t1'], pred['pred_t2'],
                label['label_t1'], label['label_t2'],
            )
        elif self.mode == 'per_class_bcd':
            self.evaluator.update(
                pred['pred_t1'], pred['pred_t2'],
                label['label_t1'], label['label_t2'],
                per_class_change=pred.get('per_class_change', None),
            )

    def compute(self) -> Dict[str, float]:
        return self.evaluator.compute()

    def reset(self):
        self.evaluator.reset()

    def summary(self) -> str:
        """生成格式化的评估摘要"""
        results = self.compute()
        lines = [
            "=" * 60,
            f"  Open-Vocabulary Change Detection Evaluation ({self.mode.upper()})",
            "=" * 60,
        ]

        if self.mode == 'bcd':
            lines.extend([
                f"  F1 Score:   {results['F1']:.2f}%",
                f"  IoU:        {results['IoU']:.2f}%",
                f"  Precision:  {results['Precision']:.2f}%",
                f"  Recall:     {results['Recall']:.2f}%",
                f"  OA:         {results['OA']:.2f}%",
                f"  Kappa:      {results['Kappa']:.2f}%",
            ])
        elif self.mode == 'scd':
            num_cls = len(self.class_names)
            lines.extend([
                f"  Fscd:       {results['Fscd']:.2f}%",
                f"  Sek:        {results['Sek']:.2f}%",
                f"  mIoU_t1:    {results['mIoU_t1']:.2f}%",
                f"  mIoU_t2:    {results['mIoU_t2']:.2f}%",
                f"  mIoU_avg:   {results['mIoU_avg']:.2f}%",
                f"  mF1_cls:    {results['mF1_cls']:.2f}%",
                f"  BCD F1:     {results['BCD_F1']:.2f}%",
                f"  BCD IoU:    {results['BCD_IoU']:.2f}%",
                "-" * 40,
                "  Per-class IoU (T1 / T2) | F1:",
            ])
            for c in range(num_cls):
                cn = self.class_names[c] if c < len(self.class_names) else f'cls{c}'
                cn_short = cn.split(',')[0][:20]
                t1 = results.get(f'IoU_t1_cls{c}', 0)
                t2 = results.get(f'IoU_t2_cls{c}', 0)
                f1 = results.get(f'F1_cls{c}', 0)
                lines.append(f"    {c} {cn_short:20s}: {t1:5.1f} / {t2:5.1f} | {f1:5.1f}")
            lines.append("-" * 40)
        elif self.mode == 'per_class_bcd':
            num_cls = len(self.class_names)
            lines.extend([
                f"  Class-avg IoU: {results['mIoU']:.2f}%",
                f"  Class-avg F1:  {results['mF1']:.2f}%",
                "-" * 60,
                f"  {'Category':<20s} {'IoU':>7s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s}",
                "-" * 60,
            ])
            for c in range(1, num_cls):
                cn = self.class_names[c] if c < len(self.class_names) else f'cls{c}'
                cn_short = cn.split(',')[0][:20]
                iou = results.get(f'IoU_cls{c}', 0)
                f1 = results.get(f'F1_cls{c}', 0)
                prec = results.get(f'Precision_cls{c}', 0)
                rec = results.get(f'Recall_cls{c}', 0)
                lines.append(f"  {cn_short:<20s} {iou:6.1f}% {f1:6.1f}% {prec:6.1f}% {rec:6.1f}%")
            lines.append("-" * 60)

        lines.extend([
            f"  Samples:    {results['num_samples']}",
            "=" * 60,
        ])
        return '\n'.join(lines)
