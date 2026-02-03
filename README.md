# ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization

### [Project Page](https://yyang181.github.io/colormnet/) | [Paper (ArXiv)](https://arxiv.org/abs/2404.06251) | [Supplemental Material](https://arxiv.org/abs/2404.06251) | [Code (Github)](https://github.com/yyang181/colormnet) 

[![google colab logo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1naXws0elPMunfcvKSryLW1lFnPOF6Nb-?usp=sharing) [![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/yyang181/ColorMNet) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=yyang181/colormnet)[![GitHub Stars](https://img.shields.io/github/stars/yyang181/colormnet?style=social)](https://github.com/yyang181/colormnet)


**This repository is the official pytorch implementation of our paper, *ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization*.**

[Yixin Yang](https://imag-njust.net/),
[Jiangxin Dong](https://imag-njust.net/),
[Jinhui Tang](https://imag-njust.net/jinhui-tang/),
[Jinshan Pan](https://jspan.github.io/) <br>

Nanjing University of Science and Technology

## 🔥 News
- [2025-10-05] Integrated with 🤗 [**Hugging Face**](https://huggingface.co/spaces)!  
  Try out the **online demo** here → [![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/yyang181/ColorMNet).  
  *Note:* Due to the **HF Pro Zero-GPU quota**, this space currently has **only 25 minutes of Zero-GPU runtime per day**. Please consider running the demo locally [app.py](https://github.com/yyang181/colormnet/blob/main/app.py) or on [Colab](https://colab.research.google.com/drive/1naXws0elPMunfcvKSryLW1lFnPOF6Nb-?usp=sharing) if you need more time.
- [2025-10-05] Add Gradio demo, see [app.py](https://github.com/yyang181/colormnet/blob/main/app.py)
- [2024-11-14] Add matrics evaluation code, see [evaluation.py](https://github.com/yyang181/colormnet/blob/main/shared/evaluation_matrics/evaluation.py). Demo command ```pip install lpips && python shared/evaluation_matrics/evaluation.py```.
- [2024-09-09] Add training code, see [train.py](https://github.com/yyang181/colormnet/blob/main/train.py).
- [2024-09-09] Colab demo for ColorMNet is available at [![google colab logo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1naXws0elPMunfcvKSryLW1lFnPOF6Nb-?usp=sharing).
- [2024-09-07] Add inference code and pretrained weights, see [test.py](https://github.com/yyang181/colormnet/blob/main/test.py).
- [2024-04-13] Project page released at [ColorMNet Project](https://yyang181.github.io/ColorMNet). Please be patient and stay updated.

## Requirements

* Python 3.8+
* PyTorch 1.11+ (See [PyTorch](https://pytorch.org/) for installation instructions)
* `torchvision` corresponding to the PyTorch version
* OpenCV (try `pip install opencv-python`)
* Others: `pip install -r requirements.txt`

## :briefcase: Dependencies and Installation

```
# git clone this repository

conda create -n colormnet python=3.8 -y
conda activate colormnet 

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# install py-thin-plate-spline
git clone https://github.com/cheind/py-thin-plate-spline.git
cd py-thin-plate-spline && pip install -e . && cd ..

# install Pytorch-Correlation-extension
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git 
cd Pytorch-Correlation-extension && python setup.py install && cd ..

pip install -r requirements.txt
```

#### 

## :gift: Checkpoints

Download the pretrained models manually and put them in `./saves` (create the folder if it doesn't exist).

|   Name    |                             URL                              |
| :-------: | :----------------------------------------------------------: |
| ColorMNet | [model](https://github.com/yyang181/colormnet/releases/download/v0.1/DINOv2FeatureV6_LocalAtten_s2_154000.pth) |

## :zap: Quick Inference

- **Test on Images**:  

  For Windows users, please follow [RuntimeError](https://github.com/yyang181/colormnet/issues/5#issuecomment-2339263103) to avoid multiprocessor Runtime error in data loader. Thanks to [@UPstud](https://github.com/UPstud). 

```
CUDA_VISIBLE_DEVICES=0 python test.py 
# Add --FirstFrameIsNotExemplar if the reference frame is not exactly the first input image. Please make sure the ref frame and the input frames are of the same size. 
```

## Gradio Demo:
```bash
CUDA_VISIBLE_DEVICES=0 python app.py
``` 

## Pipeline (W&B Artifacts)
This repo now supports a lightweight pipeline wrapper under `src/` that keeps the core model logic intact while adding:
- YAML configs (`config/`)
- Stage runners (`src/loader`, `src/augmenter`, `src/splitter`, `src/trainer`, `src/evaluator`)
- W&B artifact lineage between stages
- Shared core code under `shared/` (model, dataset, inference, util, evaluation)

### Stages
1) **Loader**: registers datasets as a W&B artifact (train/val + eval input/ref).
2) **Augmenter (optional)**: pass-through artifact (augmentation stays online in the dataset loader).
3) **Splitter**: creates train/val split lists (no data copying).
4) **Trainer**: trains with the split lists and logs samples/metrics.
5) **Evaluator**: runs inference + paper metrics on the eval set and logs results.

### Example commands
```bash
python -m src.loader.runner --config config/loader/base.yaml
# Optional (pass-through)
python -m src.augmenter.runner --config config/augmenter/base.yaml
python -m src.splitter.runner --config config/splitter/base.yaml
python -m src.trainer.runner --config config/trainer/base.yaml
python -m src.evaluator.runner --config config/evaluator/base.yaml
```

### Notes
- Edit `config/loader/base.yaml` to point at your train/val/eval roots.
- The evaluator expects paired eval input/ref folders (same video folder names).
- LPIPS/FID/CDC require extra deps or downloads (see `shared/evaluation_matrics`).
- S3 uploads are required for all pipeline stages; configure `s3` in each config (bucket/prefix/region).

### DINO Backbone (v2/v3)
You can switch the DINO backbone used by the key encoder via the `dino` block in configs.
Example (v3 via HF, gated model):
```
dino:
  backbone: dinov3_vits16
  source: hf
  hf_model: facebook/dinov3-vits16-pretrain-lvd1689m
  align: interpolate   # or learned
  target_stride: 16
  layer_indices: [8, 9, 10, 11]
  num_prefix_tokens: 5
  freeze: true
```
For gated HF models, authenticate with `huggingface-cli login` or set `HF_TOKEN` in your environment.
If you use `source: hf`, install `transformers` (and its deps).

## Train
### Dataset structure for both the training set and the validation set
```
# Specify --davis_root and --validation_root
data_root/
├── 001/
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
├── 002/
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── ...
```
### Training script
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --master_port 25205 \
    --nproc_per_node=1 \
    train.py \
    --exp_id DINOv2FeatureV6_LocalAtten_DAVISVidevo \
    --davis_root /path/to/your/training/data/\
    --validation_root /path/to/your/validation/data\
    --savepath ./wandb_save_dir
```

### To Do
- [x] Release training code
- [x] Release testing code
- [x] Release pre-trained models
- [x] Release demo

### Citation
If our work is useful for your research, please consider citing:

    @inproceedings{yang2024colormnet,
        author = {Yang, Yixin and Dong, Jiangxin and Tang, Jinhui and Pan Jinshan},
        title = {ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization},
        booktitle = {ECCV},
        year = {2024}
    }

### License

This project is licensed under <a rel="license" href="https://github.com/yyang181/colormnet/blob/main/LICENSE">BY-NC-SA 4.0</a>, while some methods adopted in this project are with other licenses. Please refer to [LICENSES.md](https://github.com/yyang181/colormnet/blob/main/LICENSES.md) for the careful check. Redistribution and use should follow this license.

### Acknowledgement

This project is based on [XMem](https://github.com/hkchengrex/XMem). Some codes are brought from [DINOv2](https://github.com/facebookresearch/dinov2). Thanks for their awesome works.

### Contact

This repo is currently maintained by Yixin Yang ([@yyang181](https://github.com/yyang181)) and is for academic research use only. 
