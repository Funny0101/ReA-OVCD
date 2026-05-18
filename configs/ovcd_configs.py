"""
Open-Vocabulary Change Detection 各数据集超参配置

数据根目录已写死为你服务器上的实际路径，可通过命令行 --data_root 覆盖。
"""

# ============================================================
# SECOND_processed (Semantic Change Detection, 256x256 patches)
# ============================================================
SECOND_CONFIG = dict(
    dataset='second',
    mode='scd',
    data_root='/archive/hot2/chj/data/datasets/SECOND_processed',
    class_names=[
        'road,pavement,street,parking lot,asphalt,sidewalk,highway',
        'water,river,lake,pond,reservoir,stream',
        'bare ground,bare soil,barren,dirt,sand',
        'low vegetation,grass,lawn,shrub,grassland,meadow',
        'tree,trees,forest,woodland,canopy,grove',
        'building,roof,house,structure',
        'playground,sports field,running track,athletic track,stadium,court,recreation ground',
    ],
    confidence_threshold=0.1,
    prob_thd=0.1,
    change_threshold=0.3,
    presence_alpha=1.0,
    use_presence=True,
    slide_stride=0,   # 256x256, 不需要滑窗
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
        # 'road,pavement,street,parking lot,asphalt,sidewalk,highway',
        # 'background,clutter,unknown,pavement,parking lot,asphalt',
        # 'background,clutter,unknown',
        'background',
        # 'background,parking lot',
        'water,river,lake,pond,reservoir,stream',
        'bare ground,bare soil,barren,dirt,sand',
        'low vegetation,grass,lawn,shrub,grassland,meadow',
        'tree,trees,forest,woodland,canopy,grove',
        'building,roof,house,structure',
        # 'bare ground,bare soil,dirt,sand',
        # 'low vegetation,grass,lawn,grassland,meadow',
        # 'tree,trees,forest,canopy,grove',        
        # 'building,roof,house',
        # 'playground,sports field,running track,athletic track,stadium,court,recreation ground',
        # 'playground,sports field,soccer field,basketball court,running track,athletic track,stadium',
        'sports field,running track,athletic track,stadium',
        # 'football court,basketball court,baseball court,running track'
        # 'playground,football court,basketball court,baseball court,running track' 
        # 'sports stadium,track and field,football court,basketball court,baseball court,running track,athletic track,stadium',
        # 'sports stadium,football court,basketball court,baseball court,running track,athletic track,stadium',
        # 'track and field'
        # 
        # 'sports stadium,basketball court,running track,stadium'
        # 'sports stadium,football court,basketball court,baseball court,running track,athletic track,stadium'
        # 'sports stadium,track and field,basketball court,baseball court,running track,athletic track,stadium'
    ],
    # confidence_threshold=0.3,
    confidence_threshold=0.4,
    prob_thd=0.3,
    # change_threshold=0.4,
    change_threshold=0.5,
    presence_alpha=1.0,
    use_presence=True,
    slide_stride=0,
    slide_crop=0,
    # Per-class BDWCD: τ 和 min_area 按类别调优
    # class_id: 1=water, 2=ground, 3=low_veg, 4=tree, 5=building, 6=playground
    pc_boundary_tau={1: 5, 2: 6, 3: 6, 4: 3, 5: 6, 6: 6},
    pc_min_change_area={1: 200, 2: 200, 3: 200, 4: 100, 5: 200, 6: 200},
    pc_sigmoid_scale=2.0,
    pc_logit_margin=0.0,  # Soft XOR: 要求 |logit_t1[c]-logit_t2[c]| > margin
)

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
        # 'road,street,pavement,highway,vegetation,tree,grass,forest,farmland,cropland,water,river,lake,bare soil,dirt,ground',
        # 'road,street,pavement,vegetation,tree,grass,forest,farmland,water,bare soil,ground',
        # 'building,house,roof',
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
        # 'background,ground,road,vegetation,tree,forest,farmland,water',
        # 'building,house,roof',
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
        # 'road,street,pavement,highway,vegetation,tree,grass,forest,farmland,cropland,water,river,lake,bare soil,dirt,ground',
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
# HRSCD (Semantic Change Detection, 256x256 patches)
# ============================================================
HRSCD_CONFIG = dict(
    dataset='hrscd',
    mode='scd',
    data_root='/archive/hot2/chj/data/datasets/HRSCD_processed',
    class_names=[
        'artificial surfaces,building,road,urban',
        'agricultural areas,farmland,cropland',
        'forests,tree,woodland',
        'wetlands,marsh,swamp',
        'water,river,lake',
    ],
    confidence_threshold=0.1,
    prob_thd=0.1,
    change_threshold=0.3,
    presence_alpha=1.0,
    use_presence=True,
    slide_stride=0,
    slide_crop=0,
)

# ============================================================
# S2Looking (Binary Building Change Detection, 256x256 patches)
# ============================================================
S2LOOKING_CONFIG = dict(
    dataset='s2looking',
    mode='bcd',
    data_root='/archive/hot2/chj/data/datasets/s2looking/S2Looking',
    class_names=[
        'background,ground,road,vegetation,no change',
        'building,roof,house',
    ],
    confidence_threshold=0.7,
    prob_thd=0.5,
    change_threshold=0.5,
    presence_alpha=1.0,
    use_presence=True,
    slide_stride=0,   
    slide_crop=0,
)

# ============================================================
# IACF Post-processing Configs (per dataset)
# ============================================================
# BDWCD 后处理: 每个数据集只需 3 个核心参数
# boundary_distance_tau: 边界抑制距离 (越大→抑制范围越广→FP越少但可能丢TP)
# boundary_sigmoid_scale: sigmoid平滑度 (越大→过渡越柔和)
# min_change_area: 最小变化区域面积 (去噪)
POSTPROCESS_CONFIGS = {
    'second': dict(
        boundary_distance_tau=3.0,
        boundary_sigmoid_scale=1.5,
        use_morphological_opening=True,
        opening_kernel_size=3,
        opening_iterations=1,
        use_small_region_removal=True,
        min_change_area=20,
    ),
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
    'hrscd': dict(
        boundary_distance_tau=0,       # HRSCD: recall本身低, BDWCD反效果, 设为0表示禁用
        boundary_sigmoid_scale=1.5,
        use_morphological_opening=True,
        opening_kernel_size=3,
        opening_iterations=1,
        use_small_region_removal=True,
        min_change_area=20,
    ),
    's2looking': dict(
        boundary_distance_tau=8.0,
        boundary_sigmoid_scale=2.0,
        use_morphological_opening=True,
        opening_kernel_size=3,
        opening_iterations=1,
        use_small_region_removal=True,
        min_change_area=150,
    ),
}

# ============================================================
# Registry
# ============================================================
CONFIG_REGISTRY = {
    'second': SECOND_CONFIG,
    'second_raw': SECOND_RAW_CONFIG,
    'levircd': LEVIR_CD_CONFIG,
    'levir-cd': LEVIR_CD_CONFIG,
    'whucd': WHU_CD_CONFIG,
    'whu-cd': WHU_CD_CONFIG,
    'dsifn': DSIFN_CONFIG,
    'hrscd': HRSCD_CONFIG,
    's2looking': S2LOOKING_CONFIG,
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
