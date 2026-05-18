"""
Open-Vocabulary Change Detection 评估主脚本
统一了 SCD 与 BCD 的评估流程。
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from ovcd_segmentor import OVCDSegmentor
from ovcd_pipeline import OpenVocabularyChangeDetector
from ovcd_datasets import build_cd_dataset
from ovcd_metrics import OVCDEvaluator


# ============ Logging Utility ============

class TeeOutput:
    """Redirect stdout to both terminal and a log file."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


# ============ Visualization Helpers ============

# Default color palette for semantic masks (up to 20 classes)
PALETTE = [
    (255, 255, 255),  # 0: white (background)
    (0, 0, 255),      # 1: blue
    (128, 128, 128),  # 2: gray
    (0, 128, 0),      # 3: dark green
    (0, 255, 0),      # 4: green
    (255, 0, 0),      # 5: red
    (255, 255, 0),    # 6: yellow
    (255, 0, 255),    # 7: magenta
    (0, 255, 255),    # 8: cyan
    (128, 0, 0),      # 9: dark red
    (0, 128, 128),    # 10: teal
    (128, 128, 0),    # 11: olive
    (128, 0, 128),    # 12: purple
    (0, 0, 128),      # 13: navy
    (64, 224, 208),   # 14: turquoise
    (255, 165, 0),    # 15: orange
    (255, 192, 203),  # 16: pink
    (165, 42, 42),    # 17: brown
    (144, 238, 144),  # 18: light green
    (173, 216, 230),  # 19: light blue
]

CHANGE_COLOR = (255, 255, 255)   # White for change
NO_CHANGE_COLOR = (0, 0, 0)      # Black for no change


def colorize_mask(mask: np.ndarray, num_classes: int) -> Image.Image:
    """Convert a class index mask [H,W] to a colored RGB image."""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(PALETTE))):
        color_img[mask == c] = PALETTE[c]
    return Image.fromarray(color_img)


def colorize_binary(mask: np.ndarray) -> Image.Image:
    """Convert a binary mask [H,W] to image (white=change, black=no change)."""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    color_img[mask > 0] = CHANGE_COLOR
    return Image.fromarray(color_img)


def save_legend(class_names: list, save_path: str):
    """Save a legend image showing class-color mapping."""
    num_classes = len(class_names)
    row_h, col_w = 30, 300
    img = Image.new('RGB', (col_w, row_h * num_classes), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for c in range(num_classes):
        color = PALETTE[c] if c < len(PALETTE) else (200, 200, 200)
        y = c * row_h
        draw.rectangle([0, y, row_h, y + row_h], fill=color)
        cn = class_names[c].split(',')[0]
        draw.text((row_h + 5, y + 5), f'{c}: {cn}', fill=(255, 255, 255))
    img.save(save_path)


def save_visualization(
    sample: dict,
    pred_dict: dict,
    label_dict: dict,
    mode: str,
    class_names: list,
    vis_dir: str,
    idx: int,
):
    """Save visualization images for one sample."""
    name = sample.get('name', f'{idx:06d}')
    num_classes = len(class_names)
    sample_dir = os.path.join(vis_dir, name)
    os.makedirs(sample_dir, exist_ok=True)

    # Save original images
    sample['img_t1'].save(os.path.join(sample_dir, 'img_t1.png'))
    sample['img_t2'].save(os.path.join(sample_dir, 'img_t2.png'))

    # Save predicted binary change mask
    if pred_dict.get('change_mask') is not None:
        colorize_binary(pred_dict['change_mask']).save(
            os.path.join(sample_dir, 'pred_change.png'))

    if mode == 'bcd':
        # GT binary label
        if 'label' in label_dict:
            colorize_binary(label_dict['label']).save(
                os.path.join(sample_dir, 'gt_change.png'))

    elif mode in ('scd', 'per_class_bcd'):
        # ---- Semantic predictions (colored) ----
        if pred_dict.get('pred_t1') is not None:
            colorize_mask(pred_dict['pred_t1'], num_classes).save(
                os.path.join(sample_dir, 'pred_t1.png'))
        if pred_dict.get('pred_t2') is not None:
            colorize_mask(pred_dict['pred_t2'], num_classes).save(
                os.path.join(sample_dir, 'pred_t2.png'))

        # ---- GT semantic labels (colored) ----
        if 'label_t1' in label_dict:
            colorize_mask(label_dict['label_t1'], num_classes).save(
                os.path.join(sample_dir, 'gt_t1.png'))
        if 'label_t2' in label_dict:
            colorize_mask(label_dict['label_t2'], num_classes).save(
                os.path.join(sample_dir, 'gt_t2.png'))

        # ---- GT binary change (derived from semantic labels) ----
        if 'label_t1' in label_dict and 'label_t2' in label_dict:
            gt_change = (label_dict['label_t1'] != label_dict['label_t2']).astype(np.uint8)
            colorize_binary(gt_change).save(
                os.path.join(sample_dir, 'gt_change.png'))

        # ---- Per-class predicted change masks ----
        per_class_change = pred_dict.get('per_class_change')
        if per_class_change is not None:
            for c, mask in per_class_change.items():
                cn = class_names[c].split(',')[0] if c < len(class_names) else f'cls{c}'
                colorize_binary(mask.astype(np.uint8)).save(
                    os.path.join(sample_dir, f'pred_change_cls{c}_{cn}.png'))
        else:
            # Derive per-class change from pred_t1/pred_t2 XOR
            if pred_dict.get('pred_t1') is not None and pred_dict.get('pred_t2') is not None:
                for c in range(1, num_classes):
                    cn = class_names[c].split(',')[0] if c < len(class_names) else f'cls{c}'
                    pred_cls_change = ((pred_dict['pred_t1'] == c) != (pred_dict['pred_t2'] == c)).astype(np.uint8)
                    if pred_cls_change.sum() > 0:
                        colorize_binary(pred_cls_change).save(
                            os.path.join(sample_dir, f'pred_change_cls{c}_{cn}.png'))

        # ---- Per-class GT change masks ----
        if 'label_t1' in label_dict and 'label_t2' in label_dict:
            for c in range(1, num_classes):  # skip background
                cn = class_names[c].split(',')[0] if c < len(class_names) else f'cls{c}'
                gt_cls_change = (
                    (label_dict['label_t1'] == c) != (label_dict['label_t2'] == c)
                ).astype(np.uint8)
                if gt_cls_change.sum() > 0:  # Only save if there's actual change
                    colorize_binary(gt_cls_change).save(
                        os.path.join(sample_dir, f'gt_change_cls{c}_{cn}.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='Open-Vocabulary Change Detection Evaluation')

    # Data settings
    parser.add_argument('--dataset', type=str, default='second',
                        choices=['second', 'second_raw', 'levircd', 'levir-cd',
                                 'whucd', 'whu-cd', 'dsifn', 'hrscd', 's2looking'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override dataset root directory')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--mode', type=str, default=None, choices=['bcd', 'scd', 'per_class_bcd'],
                        help='Evaluation mode (auto-detected if not specified)')

    # Model settings
    parser.add_argument('--sam3_checkpoint', type=str,
                        default='/archive/hot2/chj/data/model/sam3.pt')
    parser.add_argument('--confidence_threshold', type=float, default=None)
    parser.add_argument('--classname_path', type=str, default=None,
                        help='Override class names from file')

    # Pipeline settings
    parser.add_argument('--change_threshold', type=float, default=None)
    parser.add_argument('--presence_alpha', type=float, default=None)
    parser.add_argument('--use_presence', action='store_true', default=None)
    parser.add_argument('--no_presence', dest='use_presence', action='store_false')
    parser.add_argument('--use_pamr', action='store_true', default=False)
    parser.add_argument('--use_tif_mask', action='store_true', default=False,
                        help='DSIFN: use .tif mask instead of .jpg label')

    # IACF / BDWCD Post-processing
    parser.add_argument('--no_postprocess', action='store_true', default=False,
                        help='Disable BDWCD post-processing')
    parser.add_argument('--boundary_tau', type=float, default=None,
                        help='Override BDWCD boundary distance threshold')
    
    # Per-class settings (For SCD / per_class_bcd)
    parser.add_argument('--pc_tau', type=float, default=None,
                        help='Per-class BDWCD boundary tau (overrides config)')

    # Sliding window
    parser.add_argument('--slide_stride', type=int, default=None)
    parser.add_argument('--slide_crop', type=int, default=None)

    # Output settings
    parser.add_argument('--save_dir', type=str, default='./ovcd_results/')
    parser.add_argument('--save_vis', action='store_true', default=False)
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Max samples to evaluate (-1 = all)')
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    parser.add_argument('--debug_playground_only', action='store_true', default=False,
                        help='Debug only playground: skip samples whose GT labels do not contain playground class')

    return parser.parse_args()


def load_config_for_dataset(dataset_name: str) -> dict:
    from configs.ovcd_configs import get_config
    return get_config(dataset_name)


def main():
    args = parse_args()

    # ---- Load default config for this dataset ----
    cfg = load_config_for_dataset(args.dataset)

    # Override with command line args
    if args.data_root is not None: cfg['data_root'] = args.data_root
    if args.mode is not None: cfg['mode'] = args.mode
    if args.confidence_threshold is not None: cfg['confidence_threshold'] = args.confidence_threshold
    if args.change_threshold is not None: cfg['change_threshold'] = args.change_threshold
    if args.presence_alpha is not None: cfg['presence_alpha'] = args.presence_alpha
    if args.use_presence is not None: cfg['use_presence'] = args.use_presence
    if args.slide_stride is not None: cfg['slide_stride'] = args.slide_stride
    if args.slide_crop is not None: cfg['slide_crop'] = args.slide_crop

    mode = cfg['mode']
    class_names = cfg['class_names']

    # Create timestamped run directory for this evaluation
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, args.dataset, 'logs', run_id)
    os.makedirs(save_dir, exist_ok=True)

    # Setup logging: tee stdout to log file
    tee = TeeOutput(os.path.join(save_dir, 'eval.log'))
    sys.stdout = tee

    # Save full config at run start
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump({
            'run_id': run_id,
            'args': vars(args),
            'cfg': cfg,
        }, f, indent=2, default=str)

    if args.classname_path:
        with open(args.classname_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]

    playground_idx = 6
    # if args.debug_playground_only:
    #     for i, cn in enumerate(class_names):
    #         if 'playground' in cn.lower():
    #             playground_idx = i
    #             break
    #     if playground_idx is None:
    #         raise ValueError("--debug_playground_only enabled, but no 'playground' class found in class_names")

    print("=" * 70)
    print("  Open-Vocabulary Change Detection Evaluation")
    print("=" * 70)
    print(f"  Dataset:      {args.dataset}")
    print(f"  Mode:         {mode}")
    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  Change Thresh:{cfg['change_threshold']} | Presence alpha: {cfg['presence_alpha']}")
    if args.debug_playground_only:
        print(f"  Debug:        playground-only mode ON (class_id={playground_idx})")

    # ---- Configure BDWCD Post-processor ----
    postprocess_cfg = cfg.get('postprocess', {})
    instance_filter = None
    if not args.no_postprocess and postprocess_cfg:
        from ovcd_postprocess import BoundaryDistanceChangeFilter
        pp_kwargs = postprocess_cfg.copy()
        
        # Determine filter mode and merge configs
        if mode == 'per_class_bcd':
            pp_kwargs['filter_mode'] = 'per_class'
            # Merge per-class specific configs
            if args.pc_tau is not None:
                pp_kwargs['boundary_distance_tau'] = args.pc_tau
            elif 'pc_boundary_tau' in cfg:
                pp_kwargs['boundary_distance_tau'] = cfg['pc_boundary_tau']
            
            if 'pc_min_change_area' in cfg:
                pp_kwargs['min_change_area'] = cfg['pc_min_change_area']
            
            pp_kwargs['logit_margin'] = cfg.get('pc_logit_margin', 0.0)
            
            print(f"  BDWCD:        PER-CLASS (τ={pp_kwargs.get('boundary_distance_tau')}, "
                  f"min_area={pp_kwargs.get('min_change_area')}, logit_margin={pp_kwargs.get('logit_margin')})")
        else:
            pp_kwargs['filter_mode'] = 'global'
            if args.boundary_tau is not None:
                pp_kwargs['boundary_distance_tau'] = args.boundary_tau
                
            print(f"  BDWCD:        GLOBAL (τ={pp_kwargs.get('boundary_distance_tau', 0)}, "
                  f"min_area={pp_kwargs.get('min_change_area', 100)})")
        
        pp_kwargs['verbose'] = args.verbose
        instance_filter = BoundaryDistanceChangeFilter(**pp_kwargs)
    else:
        print("  BDWCD:        OFF")
    print("=" * 70)

    # ---- Build dataset ----
    dataset_kwargs = {'use_tif_mask': True} if (args.dataset == 'dsifn' and args.use_tif_mask) else {}
    dataset = build_cd_dataset(args.dataset, cfg['data_root'], args.split, **dataset_kwargs)

    if args.max_samples > 0:
        dataset.file_list = dataset.file_list[:args.max_samples]

    # ---- Build segmentor & detector ----
    segmentor = OVCDSegmentor(
        class_names=class_names, device='cuda',
        confidence_threshold=cfg['confidence_threshold'],
        prob_thd=cfg['prob_thd'],
        use_sem_seg=True, use_presence_score=cfg['use_presence'],
        sam3_checkpoint=args.sam3_checkpoint,
    )

    detector = OpenVocabularyChangeDetector(
        class_names=class_names, device='cuda',
        change_threshold=cfg['change_threshold'],
        presence_alpha=cfg['presence_alpha'],
        use_presence_filtering=cfg['use_presence'],
        instance_filter=instance_filter,
    )

    evaluator = OVCDEvaluator(num_classes=len(class_names), mode=mode, class_names=class_names)

    # ---- Setup visualization directory ----
    vis_dir = None
    if args.save_vis:
        vis_dir = os.path.join(args.save_dir, args.dataset, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        save_legend(class_names, os.path.join(vis_dir, 'legend.png'))
        print(f"  📁 Visualization dir: {vis_dir}")

    # ---- Run evaluation ----
    print(f"\n🚀 Starting evaluation on {len(dataset)} samples...")
    t_start = time.time()
    use_slide = cfg['slide_stride'] > 0 and cfg['slide_crop'] > 0
    processed_samples = 0
    skipped_no_playground = 0

    for idx in tqdm(range(len(dataset)), desc='Evaluating'):
        sample = dataset[idx]
        img_t1, img_t2 = sample['img_t1'], sample['img_t2']

        # Debug mode: 在推理前过滤掉不含 playground GT 的样本
        if args.debug_playground_only:
            if mode not in ('scd', 'per_class_bcd'):
                raise ValueError("--debug_playground_only only supports scd/per_class_bcd mode")
            if ('label_t1' not in sample) or ('label_t2' not in sample):
                skipped_no_playground += 1
                continue

            has_playground = bool((sample['label_t1'] == playground_idx).any() or
                                  (sample['label_t2'] == playground_idx).any())
            if not has_playground:
                skipped_no_playground += 1
                continue

        # Forward pass
        if use_slide:
            result_t1 = segmentor.slide_inference_single(img_t1, stride=cfg['slide_stride'], crop_size=cfg['slide_crop'])
            result_t2 = segmentor.slide_inference_single(img_t2, stride=cfg['slide_stride'], crop_size=cfg['slide_crop'])
        else:
            result_t1 = segmentor.forward_single(img_t1)
            result_t2 = segmentor.forward_single(img_t2)

        # 核心改进：无论SCD还是BCD，都统一调用 detect_change，底层由 pipeline 与 postprocess 配合处理。
        cd_result = detector.detect_change(result_t1, result_t2, img_t1, img_t2)

        pred_dict = {
            'change_mask': cd_result.change_mask.cpu().numpy() if cd_result.change_mask is not None else None,
            'pred_t1': cd_result.pred_t1.cpu().numpy(),
            'pred_t2': cd_result.pred_t2.cpu().numpy(),
            'per_class_change': cd_result.per_class_change, # BCD此项为None，SCD有值
        }

        # Handle labels
        label_dict = {}
        if mode == 'bcd' and 'label' in sample:
            label_dict['label'] = sample['label']
        elif mode in ('scd', 'per_class_bcd') and 'label_t1' in sample:
            label_dict['label_t1'] = sample['label_t1']
            label_dict['label_t2'] = sample['label_t2']

        if label_dict:
            evaluator.update(pred_dict, label_dict)
            processed_samples += 1

        # Save visualization if requested
        if args.save_vis:
            save_visualization(sample, pred_dict, label_dict, mode, class_names, vis_dir, idx)

    # ---- Results ----
    elapsed = time.time() - t_start
    avg_time = elapsed / max(processed_samples, 1)
    print(f"\n⏱️  Completed in {elapsed:.1f}s ({avg_time:.2f}s/sample, processed={processed_samples}, skipped={skipped_no_playground})")
    print(evaluator.summary())

    results = evaluator.compute()
    results['config'] = {
        'run_id': run_id,
        'dataset': args.dataset, 'mode': mode,
        'change_threshold': cfg['change_threshold'],
        'presence_alpha': cfg['presence_alpha'],
        'confidence_threshold': cfg['confidence_threshold'],
        'use_presence': cfg['use_presence'],
        'slide_stride': cfg['slide_stride'],
        'slide_crop': cfg['slide_crop'],
        'postprocess': 'BDWCD' if instance_filter else 'none',
        'num_classes': len(class_names),
        'class_names': class_names,
        'elapsed_seconds': elapsed,
        'num_samples': processed_samples,
        'num_samples_total': len(dataset),
        'num_samples_skipped_no_playground': skipped_no_playground,
        'debug_playground_only': args.debug_playground_only,
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📄 Results saved to {save_dir}")
    print(f"   ├── eval.log       (本次运行完整日志)")
    print(f"   ├── config.json    (完整配置)")
    print(f"   ├── results.json   (评估结果)")
    if args.save_vis:
        print(f"   └── vis/           (可视化图片)")

    # Close logging tee
    tee.close()

if __name__ == '__main__':
    main()