"""
Open-Vocabulary Change Detection Segmentor
基于 SegEarth-OV3 的双时相语义分割模块

核心能力：
1. 对双时相影像分别做开放词汇语义分割
2. 提取SAM3的条件特征 F_cond 用于特征级对比
3. 提取 presence score 用于类别级变化先验
"""

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Dict, Optional

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class OVCDSegmentor(nn.Module):
    """
    Open-Vocabulary Change Detection Segmentor
    
    对单张图像进行开放词汇语义分割，同时返回：
    - seg_logits: 逐类别的分割logits [num_queries, H, W]
    - presence_scores: 逐类别的presence score [num_queries]
    - fused_features: SAM3的条件特征（用于特征级变化检测）
    """

    def __init__(
        self,
        class_names: List[str],
        device: str = 'cuda',
        confidence_threshold: float = 0.5,
        use_sem_seg: bool = True,
        use_presence_score: bool = True,
        prob_thd: float = 0.0,
        use_transformer_decoder: bool = True,
        sam3_checkpoint: str = '/archive/hot2/chj/data/model/sam3.pt',
        bpe_path: str = './sam3/assets/bpe_simple_vocab_16e6.txt.gz',
    ):
        super().__init__()
        self.device = device
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder

        # Build SAM3 model
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=sam3_checkpoint,
            device=device,
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)

        # Parse class names (support synonym format: "building,roof,house")
        self.query_words = []
        self.query_to_cls = []  # query_idx -> class_idx mapping
        for cls_idx, name_group in enumerate(class_names):
            synonyms = [n.strip() for n in name_group.split(',')]
            for syn in synonyms:
                self.query_words.append(syn)
                self.query_to_cls.append(cls_idx)

        self.num_classes = len(class_names)
        self.num_queries = len(self.query_words)
        self.query_to_cls_tensor = torch.tensor(self.query_to_cls, dtype=torch.long, device=device)
        self.prob_thd = prob_thd

    @torch.no_grad()
    def forward_single(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        对单张图像进行推理，返回详细的中间结果

        Args:
            image: PIL Image

        Returns:
            dict with keys:
            - 'seg_logits': [num_queries, H, W] 逐query的分割logits
            - 'seg_logits_cls': [num_classes, H, W] 聚合到class级别的logits
            - 'presence_scores': [num_queries] 逐query的presence score
            - 'presence_scores_cls': [num_classes] 聚合到class级别的presence score
            - 'pred_label': [H, W] 预测的类别标签
        """
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)
        presence_scores = torch.zeros(self.num_queries, device=self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)

            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(
                    state=inference_state, prompt=query_word
                )

                # Instance head (Transformer decoder)
                if self.use_transformer_decoder:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = inference_state['object_score'][inst_id]

                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape),
                                    size=(h, w), mode='bilinear', align_corners=False,
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(
                                seg_logits[query_idx], instance_logits * instance_score
                            )

                # Semantic head
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape[-2:] != (h, w):
                        semantic_logits = F.interpolate(
                            semantic_logits if semantic_logits.ndim == 4
                            else semantic_logits.unsqueeze(0).unsqueeze(0),
                            size=(h, w), mode='bilinear', align_corners=False,
                        ).squeeze()
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)

                # Presence score
                presence_scores[query_idx] = inference_state['presence_score']

                # Apply presence-guided filtering to logits
                if self.use_presence_score:
                    seg_logits[query_idx] = seg_logits[query_idx] * presence_scores[query_idx]

        # Aggregate queries to class-level (handle synonyms)
        seg_logits_cls = self._aggregate_to_class(seg_logits)
        presence_scores_cls = self._aggregate_presence_to_class(presence_scores)
        # pred_label = seg_logits_cls.argmax(dim=0)
        max_vals, pred_label = seg_logits_cls.max(dim=0)
        pred_label[max_vals < self.prob_thd] = 0 # 结果是 bg_idx -> 正确过滤！
        # print("prob_thd:", self.prob_thd, "max_vals range:", max_vals.min().item(), max_vals.max().item(), "pred_label unique:", torch.unique(pred_label))
        return {
            'seg_logits': seg_logits,
            'seg_logits_cls': seg_logits_cls,
            'presence_scores': presence_scores,
            'presence_scores_cls': presence_scores_cls,
            'pred_label': pred_label,
        }

    def forward_bitemporal(
        self, image_t1: Image.Image, image_t2: Image.Image
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        对双时相图像分别进行推理

        Returns:
            dict with 't1' and 't2' keys, each containing forward_single的输出
        """
        result_t1 = self.forward_single(image_t1)
        result_t2 = self.forward_single(image_t2)
        return {'t1': result_t1, 't2': result_t2}

    def _aggregate_to_class(self, seg_logits: torch.Tensor) -> torch.Tensor:
        """将query级别的logits聚合到class级别 (max over synonyms)"""
        if self.num_queries == self.num_classes:
            return seg_logits

        cls_logits = torch.zeros(
            (self.num_classes, *seg_logits.shape[1:]), device=self.device
        )
        for q_idx in range(self.num_queries):
            c_idx = self.query_to_cls[q_idx]
            cls_logits[c_idx] = torch.max(cls_logits[c_idx], seg_logits[q_idx])
        return cls_logits

    def _aggregate_presence_to_class(self, presence_scores: torch.Tensor) -> torch.Tensor:
        """将query级别的presence score聚合到class级别"""
        if self.num_queries == self.num_classes:
            return presence_scores

        cls_scores = torch.zeros(self.num_classes, device=self.device)
        for q_idx in range(self.num_queries):
            c_idx = self.query_to_cls[q_idx]
            cls_scores[c_idx] = max(cls_scores[c_idx], presence_scores[q_idx])
        return cls_scores

    def slide_inference_single(
        self, image: Image.Image, stride: int = 512, crop_size: int = 512
    ) -> Dict[str, torch.Tensor]:
        """滑动窗口推理（用于大图）"""
        w_img, h_img = image.size
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)
        presence_acc = torch.zeros(self.num_queries, device=self.device)
        num_patches = 0

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = image.crop((x1, y1, x2, y2))
                result = self.forward_single(crop_img)

                preds[:, y1:y2, x1:x2] += result['seg_logits']
                count_mat[:, y1:y2, x1:x2] += 1
                presence_acc += result['presence_scores']
                num_patches += 1

        preds = preds / count_mat
        presence_avg = presence_acc / num_patches
        seg_logits_cls = self._aggregate_to_class(preds)
        presence_scores_cls = self._aggregate_presence_to_class(presence_avg)

        return {
            'seg_logits': preds,
            'seg_logits_cls': seg_logits_cls,
            'presence_scores': presence_avg,
            'presence_scores_cls': presence_scores_cls,
            'pred_label': seg_logits_cls.argmax(dim=0),
        }
