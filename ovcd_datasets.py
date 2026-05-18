"""
Bi-temporal Dataset Loaders for Open-Vocabulary Change Detection

根据实际数据集目录结构适配:

1. SECOND (Semantic CD):
   - 原始: /SECOND/test/{im1,im2,label1,label2}/  512x512, label=RGB
   - 处理后: /SECOND_processed/Test/{t1,t2,label1,label2,change}/  256x256, label=grayscale(0-6)

2. LEVIR-CD (Binary CD):
   - /levircd/{A,B,label}/  1024x1024, label={0,255}

3. WHU-CD (Binary CD):
   - /whucd/{A,B,label}/  512x512, label={0,255}

4. DSIFN (Binary Building CD):
   - /DSIFN/{A,B,label}/ 图像和标签都是 .jpg
   - 或 /DSIFN/DSIFN/{t1,t2,mask}/ mask是 .tif (float)

5. HRSCD (Semantic CD):
   - /HRSCD_processed/Test/{t1,t2,label1,label2,change}/  256x256
   - label=I;16格式, class: {1,2,3,...}

6. S2Looking (Binary Building CD):
   - /s2looking-256/{train,val,test}/{A,B,label}/  256x256, label={0,255}
"""

import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Callable


class BitemporalCDDataset(Dataset):
    """通用双时相变化检测数据集基类"""

    def __init__(
        self,
        t1_dir: str,
        t2_dir: str,
        label_dir: Optional[str] = None,          # BCD label
        label_t1_dir: Optional[str] = None,        # SCD T1 label
        label_t2_dir: Optional[str] = None,        # SCD T2 label
        img_suffix: str = '.png',
        label_suffix: str = '.png',
        mode: str = 'bcd',                          # 'bcd' or 'scd'
        file_list: Optional[List[str]] = None,      # 指定文件列表(无后缀)
    ):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.label_dir = label_dir
        self.label_t1_dir = label_t1_dir
        self.label_t2_dir = label_t2_dir
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        self.mode = mode

        # Build file list
        if file_list is not None:
            self.file_list = file_list
        else:
            self.file_list = sorted([
                f.replace(img_suffix, '')
                for f in os.listdir(self.t1_dir)
                if f.endswith(img_suffix)
            ])

        print(f"[{self.__class__.__name__}] Loaded {len(self.file_list)} samples (mode={mode})")
        print(f"  T1: {t1_dir}")
        print(f"  T2: {t2_dir}")
        if label_dir:
            print(f"  Label: {label_dir}")
        if label_t1_dir:
            print(f"  Label T1: {label_t1_dir}")
        if label_t2_dir:
            print(f"  Label T2: {label_t2_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict:
        name = self.file_list[idx]

        t1_path = osp.join(self.t1_dir, name + self.img_suffix)
        t2_path = osp.join(self.t2_dir, name + self.img_suffix)
        img_t1 = Image.open(t1_path).convert('RGB')
        img_t2 = Image.open(t2_path).convert('RGB')

        result = {
            'img_t1': img_t1,
            'img_t2': img_t2,
            'name': name,
            't1_path': t1_path,
            't2_path': t2_path,
        }

        if self.mode == 'bcd' and self.label_dir is not None:
            label_path = osp.join(self.label_dir, name + self.label_suffix)
            if osp.exists(label_path):
                label = self._load_binary_label(label_path)
                result['label'] = label
                result['label_path'] = label_path

        elif self.mode == 'scd':
            if self.label_t1_dir and self.label_t2_dir:
                l1_path = osp.join(self.label_t1_dir, name + self.label_suffix)
                l2_path = osp.join(self.label_t2_dir, name + self.label_suffix)
                if osp.exists(l1_path) and osp.exists(l2_path):
                    label_t1 = self._load_semantic_label(l1_path)
                    label_t2 = self._load_semantic_label(l2_path)
                    result['label_t1'] = label_t1
                    result['label_t2'] = label_t2
                    result['label'] = (label_t1 != label_t2).astype(np.uint8)

        return result

    def _load_binary_label(self, path: str) -> np.ndarray:
        """加载二值标签，自动处理不同格式"""
        lbl = Image.open(path)
        arr = np.array(lbl)
        # 处理RGB格式的二值标签
        if arr.ndim == 3:
            arr = arr.max(axis=-1)
        # 归一化到 {0, 1}
        if arr.max() > 1:
            arr = (arr > 127).astype(np.uint8)
        return arr

    def _load_semantic_label(self, path: str) -> np.ndarray:
        """加载语义标签，子类可重写"""
        lbl = Image.open(path)
        arr = np.array(lbl)
        if arr.ndim == 3:
            return arr  # RGB, 子类应重写处理
        return arr.astype(np.uint8)


# ============ SECOND Dataset ============

class SECONDDataset(BitemporalCDDataset):
    """
    SECOND Dataset - Semantic Change Detection

    两种数据格式:
    1. 原始: /SECOND/test/{im1,im2,label1,label2}/  512x512, label=RGB
    2. 处理后: /SECOND_processed/Test/{t1,t2,label1,label2,change}/  256x256, label=grayscale(0-6)

    RGB color → class mapping (原始格式):
        (255,255,255) → 0  background
        (0,0,255)     → 1  water
        (128,128,128) → 2  ground
        (0,128,0)     → 3  low vegetation
        (0,255,0)     → 4  tree
        (255,0,0)     → 5  building
        (128,0,0)     → 5  building (dark red variant)
        (255,255,0)   → 6  playground
    """

    COLOR_MAP = {
        (255, 255, 255): 0,
        (0, 0, 255): 1,
        (128, 128, 128): 2,
        (0, 128, 0): 3,
        (0, 255, 0): 4,
        (255, 0, 0): 5,
        (128, 0, 0): 5,
        (255, 255, 0): 6,
    }

    CLASS_NAMES = [
        'background',
        'water,river,lake',
        'ground,bareland,barren',
        'low_vegetation',
        'tree,forest',
        'building,roof,house',
        'playground,sports_field',
    ]

    NUM_CLASSES = 7

    def __init__(
        self,
        data_root: str,
        use_processed: bool = True,
        split: str = 'test',
        **kwargs,
    ):
        self.use_processed = use_processed

        if use_processed:
            # SECOND_processed: /SECOND_processed/{Test,Train,Val}/{t1,t2,label1,label2}
            split_map = {'test': 'Test', 'train': 'Train', 'val': 'Val'}
            split_dir = osp.join(data_root, split_map.get(split, split))
            super().__init__(
                t1_dir=osp.join(split_dir, 't1'),
                t2_dir=osp.join(split_dir, 't2'),
                label_t1_dir=osp.join(split_dir, 'label1'),
                label_t2_dir=osp.join(split_dir, 'label2'),
                mode='scd',
                **kwargs,
            )
        else:
            # SECOND原始 (512x512): data_root 直接指向 test 目录
            # 使用预生成的灰度标签 label1_gray/label2_gray (值0-6)
            super().__init__(
                t1_dir=osp.join(data_root, 'im1'),
                t2_dir=osp.join(data_root, 'im2'),
                label_t1_dir=osp.join(data_root, 'label1_gray'),
                label_t2_dir=osp.join(data_root, 'label2_gray'),
                mode='scd',
                **kwargs,
            )

    def _load_semantic_label(self, path: str) -> np.ndarray:
        lbl = Image.open(path)
        arr = np.array(lbl)
        # 两种格式都已经是灰度 class index (0-6)
        return arr.astype(np.uint8)

    def _rgb_to_class(self, rgb_label: np.ndarray) -> np.ndarray:
        h, w, _ = rgb_label.shape
        class_map = np.zeros((h, w), dtype=np.uint8)
        for color, cls_idx in self.COLOR_MAP.items():
            mask = np.all(rgb_label == color, axis=-1)
            class_map[mask] = cls_idx
        return class_map


# ============ LEVIR-CD Dataset ============

class LEVIRCDDataset(BitemporalCDDataset):
    """
    LEVIR-CD - Binary Building Change Detection
    目录: /levircd/{A,B,label}/  1024x1024, label={0,255}
    """

    CLASS_NAMES = [
        'background,ground,road,vegetation,no change',
        'building,roof,house',
    ]
    NUM_CLASSES = 2

    def __init__(self, data_root: str, **kwargs):
        super().__init__(
            t1_dir=osp.join(data_root, 'A'),
            t2_dir=osp.join(data_root, 'B'),
            label_dir=osp.join(data_root, 'label'),
            mode='bcd',
            **kwargs,
        )


# ============ WHU-CD Dataset ============

class WHUCDDataset(BitemporalCDDataset):
    """
    WHU Building Change Detection
    目录: /whucd/{A,B,label}/  512x512, label={0,255}
    """

    CLASS_NAMES = [
        'background,ground,road,vegetation,no change',
        'building,roof,house',
    ]
    NUM_CLASSES = 2

    def __init__(self, data_root: str, **kwargs):
        super().__init__(
            t1_dir=osp.join(data_root, 'A'),
            t2_dir=osp.join(data_root, 'B'),
            label_dir=osp.join(data_root, 'label'),
            mode='bcd',
            **kwargs,
        )


# ============ DSIFN Dataset ============

class DSIFNDataset(BitemporalCDDataset):
    """
    DSIFN-CD - Binary Building Change Detection
    目录: /DSIFN/{A,B,label}/  512x512
    注意: 图像和标签都是 .jpg 格式
    标签是jpg压缩的，需要用阈值二值化

    也支持 /DSIFN/DSIFN/{t1,t2,mask}/ (mask为.tif float格式)
    """

    CLASS_NAMES = [
        'background,ground,road,vegetation,no change',
        'building,roof,house',
    ]
    NUM_CLASSES = 2

    def __init__(self, data_root: str, use_tif_mask: bool = False, **kwargs):
        self.use_tif_mask = use_tif_mask

        if use_tif_mask and osp.exists(osp.join(data_root, 'DSIFN', 'mask')):
            super().__init__(
                t1_dir=osp.join(data_root, 'A'),
                t2_dir=osp.join(data_root, 'B'),
                label_dir=osp.join(data_root, 'DSIFN', 'mask'),
                img_suffix='.jpg',
                label_suffix='.tif',
                mode='bcd',
                **kwargs,
            )
        else:
            super().__init__(
                t1_dir=osp.join(data_root, 'A'),
                t2_dir=osp.join(data_root, 'B'),
                label_dir=osp.join(data_root, 'label'),
                img_suffix='.jpg',
                label_suffix='.jpg',
                mode='bcd',
                **kwargs,
            )

    def _load_binary_label(self, path: str) -> np.ndarray:
        lbl = Image.open(path)
        arr = np.array(lbl, dtype=np.float32)

        if path.endswith('.tif'):
            return (arr > 0.5).astype(np.uint8)
        else:
            if arr.ndim == 3:
                arr = arr.max(axis=-1)
            return (arr > 127).astype(np.uint8)


# ============ HRSCD Dataset ============

class HRSCDDataset(BitemporalCDDataset):
    """
    HRSCD - High Resolution Semantic Change Detection
    目录: /HRSCD_processed/Test/{t1,t2,label1,label2,change}/  256x256
    标签格式: I;16 (16-bit), class values: {1,2,3,...}

    HRSCD 类别:
        1: Artificial surfaces
        2: Agricultural areas
        3: Forests
        4: Wetlands
        5: Water
    """

    CLASS_NAMES = [
        'artificial surfaces,building,road,urban',
        'agricultural areas,farmland,cropland',
        'forests,tree,woodland',
        'wetlands,marsh,swamp',
        'water,river,lake',
    ]
    # 注意: 标签值是从1开始的 (1-5), 无背景0类
    # 在评估时需要将label值减1映射到0-4
    NUM_CLASSES = 5
    LABEL_OFFSET = 1  # 标签值需要减去的偏移

    def __init__(self, data_root: str, split: str = 'test', **kwargs):
        split_map = {'test': 'Test', 'train': 'Train', 'val': 'Val'}
        split_dir = osp.join(data_root, split_map.get(split, split))

        super().__init__(
            t1_dir=osp.join(split_dir, 't1'),
            t2_dir=osp.join(split_dir, 't2'),
            label_t1_dir=osp.join(split_dir, 'label1'),
            label_t2_dir=osp.join(split_dir, 'label2'),
            label_dir=osp.join(split_dir, 'change'),
            mode='scd',
            **kwargs,
        )

    def _load_semantic_label(self, path: str) -> np.ndarray:
        """HRSCD: I;16 格式标签, 值从1开始, 映射到0开始"""
        lbl = Image.open(path)
        arr = np.array(lbl, dtype=np.int32)
        # 将 {1,2,3,4,5} 映射到 {0,1,2,3,4}
        arr = arr - self.LABEL_OFFSET
        arr = np.clip(arr, 0, self.NUM_CLASSES - 1)
        return arr.astype(np.uint8)

    def _load_binary_label(self, path: str) -> np.ndarray:
        """HRSCD change: I;16 格式, {0, 1}"""
        lbl = Image.open(path)
        arr = np.array(lbl, dtype=np.int32)
        return (arr > 0).astype(np.uint8)


# ============ S2Looking Dataset ============

class S2LookingDataset(BitemporalCDDataset):
    """
    S2Looking - Binary Building Change Detection (side-looking satellite)
    目录: /s2looking-256/{train,val,test}/{A,B,label}/  256x256, label={0,255}
    """

    CLASS_NAMES = [
        'background,ground,road,vegetation,no change',
        'building,roof,house',
    ]
    NUM_CLASSES = 2

    def __init__(self, data_root: str, split: str = 'test', **kwargs):
        split_dir = osp.join(data_root, split)
        super().__init__(
            t1_dir=osp.join(split_dir, 'Image1'),
            t2_dir=osp.join(split_dir, 'Image2'),
            label_dir=osp.join(split_dir, 'label'),
            mode='bcd',
            **kwargs,
        )


# ============ Dataset Registry ============

DATASET_REGISTRY = {
    'second': SECONDDataset,
    'second_raw': SECONDDataset,
    'levir-cd': LEVIRCDDataset,
    'levircd': LEVIRCDDataset,
    'whu-cd': WHUCDDataset,
    'whucd': WHUCDDataset,
    'dsifn': DSIFNDataset,
    'hrscd': HRSCDDataset,
    's2looking': S2LookingDataset,
}

DEFAULT_DATA_ROOTS = {
    'second': '/archive/hot2/chj/data/datasets/SECOND_processed',
    'second_raw': '/archive/hot2/chj/data/datasets/SECOND/test',
    'levir-cd': '/archive/hot2/chj/data/datasets/levircd',
    'levircd': '/archive/hot2/chj/data/datasets/levircd',
    'whu-cd': '/archive/hot2/chj/data/datasets/whucd',
    'whucd': '/archive/hot2/chj/data/datasets/whucd',
    'dsifn': '/archive/hot2/chj/data/datasets/DSIFN',
    'hrscd': '/archive/hot2/chj/data/datasets/HRSCD_processed',
    's2looking': '/archive/hot0/dql/CD-data/s2looking-256',
}


def build_cd_dataset(
    dataset_name: str,
    data_root: Optional[str] = None,
    split: str = 'test',
    **kwargs,
) -> BitemporalCDDataset:
    """工厂方法: 根据名称创建数据集"""
    name_lower = dataset_name.lower()
    if name_lower not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    if data_root is None:
        data_root = DEFAULT_DATA_ROOTS.get(name_lower)
        if data_root is None:
            raise ValueError(f"No default data_root for '{dataset_name}', please specify --data_root")

    cls = DATASET_REGISTRY[name_lower]

    if name_lower == 'second':
        return cls(data_root=data_root, use_processed=True, split=split, **kwargs)
    elif name_lower == 'second_raw':
        return cls(data_root=data_root, use_processed=False, **kwargs)
    elif name_lower in ('levir-cd', 'levircd', 'whu-cd', 'whucd'):
        return cls(data_root=data_root, **kwargs)
    elif name_lower == 'dsifn':
        return cls(data_root=data_root, **kwargs)
    elif name_lower == 'hrscd':
        return cls(data_root=data_root, split=split, **kwargs)
    elif name_lower == 's2looking':
        return cls(data_root=data_root, split=split, **kwargs)
    else:
        return cls(data_root=data_root, **kwargs)


def get_class_names(dataset_name: str) -> List[str]:
    """获取数据集的类别名称"""
    name_lower = dataset_name.lower()
    cls = DATASET_REGISTRY.get(name_lower)
    if cls and hasattr(cls, 'CLASS_NAMES'):
        return cls.CLASS_NAMES
    return ['background', 'change']


def get_num_classes(dataset_name: str) -> int:
    """获取数据集的类别数"""
    name_lower = dataset_name.lower()
    cls = DATASET_REGISTRY.get(name_lower)
    if cls and hasattr(cls, 'NUM_CLASSES'):
        return cls.NUM_CLASSES
    return 2
