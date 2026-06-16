<div align="center">

# ReA-OVCD: Reliability-Aware Open-Vocabulary Change Detection via Semantic and Spatial Refinement

</div>

> **IMPORTANT NOTE:** This repository contains code under active peer review. It is made available for review purposes only. Any use, reproduction, or dissemination of the results prior to the official publication of the associated paper is strictly prohibited without the authors' explicit permission.

## Installation

The environment setup is identical to [SAM 3](https://github.com/facebookresearch/sam3). Simply follow the SAM 3 installation instructions.

## Download Checkpoints

Download SAM 3 checkpoints from [HuggingFace](https://huggingface.co/facebook/sam3) or [ModelScope](https://modelscope.cn/models/facebook/sam3).

## Datasets

We use **test sets only** for zero-shot evaluation. Download and organize as follows:

| Dataset | Type | Test Size | Resolution | Download |
|---------|------|-----------|------------|----------|
| LEVIR-CD | Building | 128 pairs | 0.5m | [Link](https://justchenhao.github.io/LEVIR/) |
| WHU-CD | Building | 690 pairs | 0.075m | [Link](http://gpcv.whu.edu.cn/data/building_dataset.html) |
| DSIFN | Building | 48 pairs | 2m | [Link](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset) |
| SECOND | Semantic (6 classes) | 1000+ pairs | - | [Link](https://captain-whu.github.io/SCD/) |

## Model Evaluation

```bash
python eval_ovcd.py --dataset levircd --mode bcd
python eval_ovcd.py --dataset whucd --mode bcd
python eval_ovcd.py --dataset dsifn --mode bcd
python eval_ovcd.py --dataset second_raw --mode per_class_bcd
```

## Acknowledgement

This implementation is based on [SAM 3](https://github.com/facebookresearch/sam3) and [SegEarth-OV3](https://github.com/earth-insights/SegEarth-OV-3). We thank the authors for their great work.
