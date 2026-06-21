"""
Open-Vocabulary Change Detection 各数据集超参配置

数据根目录已写死，可通过命令行 --data_root 覆盖。
"""

# ============================================================
# LEVIR-CD (Binary Building Change Detection, 1024x1024)
# ============================================================
LEVIR_CD_CONFIG = dict(
    dataset='levircd',
    mode='bcd',
    data_root='/archive/hot2/chj/data/datasets/levircd',
    class_names=[
        'ground,road,vegetation',
        'building,roof,house',
    ],
    confidence_threshold=0.3,
    prob_thd=0.0,
    change_threshold=0.5,
    presence_alpha=1.0,
    use_presence=True,
    # 1024x1024 较大，使用滑动窗口
    slide_stride=512,
    slide_crop=512,
)

# ============================================================
# WHU-CD (Binary Building Change Detection, 512x512)
# ============================================================
WHU_CD_CONFIG = dict(
    dataset='whucd',
    mode='bcd',
    data_root='/archive/hot2/chj/data/datasets/whucd',
    class_names=[
        'ground,road,vegetation',
        'building,roof,house',
    ],
    confidence_threshold=0.3,
    prob_thd=0.5,
    change_threshold=0.35,
    presence_alpha=1.5,
    use_presence=True,
    slide_stride=0,
    slide_crop=0,
)

# ============================================================
# DSIFN (Binary Building Change Detection, 512x512, jpg format)
# ============================================================
DSIFN_CONFIG = dict(
    dataset='dsifn',
    mode='bcd',
    data_root='/archive/hot2/chj/data/datasets/DSIFN',
    class_names=[
        'ground,road,vegetation,farmland',
        'building,house,roof,urban structure',
    ],
    confidence_threshold=0.3,
    prob_thd=0.1,
    change_threshold=0.3,
    presence_alpha=1.2,
    use_presence=True,
    slide_stride=0,
    slide_crop=0,
)

# ============================================================
# SECOND original (512x512, RGB labels)
# ============================================================
SECOND_RAW_CONFIG = dict(
    dataset='second_raw',
    mode='scd',
    data_root='/archive/hot2/chj/data/datasets/SECOND/test',
    class_names=[
        'background',
        'water,river,lake,pond,reservoir,stream',
        'bare ground,bare soil,barren,dirt,sand',
        'low vegetation,grass,lawn,shrub,grassland,meadow',
        'tree,trees,forest,woodland,canopy,grove',
        'building,roof,house,structure',
        'sports field,running track,athletic track,stadium',
    ],
    confidence_threshold=0.4,
    prob_thd=0.3,
    change_threshold=0.5,
    presence_alpha=1.0,
    use_presence=True,
    slide_stride=0,
    slide_crop=0,
    # class_id: 1=water, 2=ground, 3=low_veg, 4=tree, 5=building, 6=playground
    pc_boundary_tau={1: 5, 2: 6, 3: 6, 4: 3, 5: 6, 6: 6},
    pc_min_change_area={1: 200, 2: 200, 3: 200, 4: 100, 5: 200, 6: 200},
    pc_sigmoid_scale=2.0,
)

# ============================================================
# IACF Post-processing Configs (per dataset)
# ============================================================
# BDWCD 后处理: 每个数据集只需 3 个核心参数
# boundary_distance_tau: 边界抑制距离 (越大→抑制范围越广→FP越少但可能丢TP)
# boundary_sigmoid_scale: sigmoid平滑度 (越大→过渡越柔和)
# min_change_area: 最小变化区域面积 (去噪)
POSTPROCESS_CONFIGS = {
    'levircd': dict(
        boundary_distance_tau=5.0,
        boundary_sigmoid_scale=2.0,
        use_morphological_opening=True,
        opening_kernel_size=3,
        opening_iterations=1,
        use_small_region_removal=True,
        min_change_area=150,
    ),
    'whucd': dict(
        boundary_distance_tau=6.0,     # WHU建筑大, 边缘偏移粗 → 大τ (τ=6最优F1=83.94%)
        boundary_sigmoid_scale=2.5,
        use_morphological_opening=True,
        opening_kernel_size=3,
        opening_iterations=1,
        use_small_region_removal=True,
        min_change_area=100,
    ),
    'dsifn': dict(
        boundary_distance_tau=5.0,
        boundary_sigmoid_scale=2.0,
        use_morphological_opening=True,
        opening_kernel_size=3,
        opening_iterations=1,
        use_small_region_removal=True,
        min_change_area=150,
    ),
    # SECOND: placeholder (per-class τ/area are in SECOND_RAW_CONFIG's pc_boundary_tau/pc_min_change_area)
    'second': dict(
        boundary_distance_tau=5.0,
        boundary_sigmoid_scale=2.0,
        use_morphological_opening=True,
        opening_kernel_size=3,
        opening_iterations=1,
        use_small_region_removal=True,
        min_change_area=200,
    ),
}

# ============================================================
# Registry
# ============================================================
CONFIG_REGISTRY = {
    'second_raw': SECOND_RAW_CONFIG,
    'levircd': LEVIR_CD_CONFIG,
    'levir-cd': LEVIR_CD_CONFIG,
    'whucd': WHU_CD_CONFIG,
    'whu-cd': WHU_CD_CONFIG,
    'dsifn': DSIFN_CONFIG,
}


def get_config(dataset_name: str) -> dict:
    """获取数据集配置（含 BDWCD 后处理参数）"""
    name = dataset_name.lower()
    if name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIG_REGISTRY.keys())}")
    cfg = CONFIG_REGISTRY[name].copy()
    # 自动注入 BDWCD post-processing 配置
    if 'postprocess' not in cfg:
        pp_key = name.replace('-', '')
        if pp_key == 'second_raw':
            pp_key = 'second'
        cfg['postprocess'] = POSTPROCESS_CONFIGS.get(pp_key, {}).copy()
    return cfg
