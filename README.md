# MDA-DD: Model Distribution-Aware Multimodal Dataset Distillation

**NeurIPS 2026** · [Conference page](https://neurips.cc/virtual/2026/loc/sydney/poster/148787)

**Jiajun Shen<sup>†</sup>, Yonglin Wu<sup>†</sup>, Ruonan Yu, Xinchao Wang<sup>*</sup>**

National University of Singapore · <sup>†</sup> Equal contribution · <sup>*</sup> Corresponding author.

> **In one sentence:** MDA-DD distills image–text data using a queue of models at different training stages, asymmetric dual projection, and second-order cross-modal moment matching.

## ⚙️ Environment

The paper's experiments used one NVIDIA RTX 5090 (32 GB). The provided scripts use Bash and CUDA. The repository includes `requirements.txt`; activate a suitable Conda environment before installing it:

```bash
conda activate MDA-DD
pip install -r requirements.txt
```

The current `requirements.txt` contains conflicting duplicate version pins, so installation needs to be checked before a public release. Pretrained NFNet and BERT weights are downloaded on first use.

## 🗂️ Data preparation

Download the [Flickr30K images](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and [MS-COCO 2014 images](https://cocodataset.org/#download). The code expects these paths relative to the repository root:

```text
distill_utils/data/
├── Flickr30k/
│   └── flickr30k-images/
│       └── <image files>.jpg
└── COCO/
    ├── train2014/
    └── val2014/
```

The dataset loaders download the Flickr30K and COCO Karpathy train/val/test JSON annotations on first use. By default they are stored under `data/Flickr30k_ann/`; you can change that location with `--ann_root`. The COCO Karpathy retrieval test split uses `val2014/` images. Dataset images are not included in this repository.

## 🚀 Distillation and evaluation

The repository provides scripts for both datasets and three synthetic-pair budgets:

| Dataset   | 100 pairs                          | 200 pairs                          | 500 pairs                          |
| :-------- | :--------------------------------- | :--------------------------------- | :--------------------------------- |
| Flickr30K | `scripts/distill_flickr_N100.sh` | `scripts/distill_flickr_N200.sh` | `scripts/distill_flickr_N500.sh` |
| MS-COCO   | `scripts/distill_coco_N100.sh`   | `scripts/distill_coco_N200.sh`   | `scripts/distill_coco_N500.sh`   |

The intended command, run from the repository root, is:

```bash
bash scripts/distill_flickr_N100.sh
```

The code evaluates synthetic data during distillation and prints image/text Recall@1/5/10 every 200 iterations by default. Optional Weights & Biases logging is enabled in the supplied scripts.

## 📖 Citation

```bibtex
@inproceedings{shen2026mda,
  title     = {Model Distribution-Aware Multimodal Dataset Distillation},
  author    = {Shen, Jiajun and Wu, Yonglin and Yu, Ruonan and Wang, Xinchao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
