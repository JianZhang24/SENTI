# SENTI

Official code repository for:

**SENTI: Semantic Enhancement and Relation Propagation Network for Multimodal Emotion Recognition in Conversations**

Jian Zhang, Peizheng Zhao, Qiufeng Wang, Han Liu, Dongming Lu, Fangyu Wu

## Overview

Multimodal Emotion Recognition in Conversations (MERC) aims to infer speaker emotions from textual, acoustic, and visual signals. SENTI addresses two core challenges in MERC:

- **cross-modal semantic discrepancy**, caused by heterogeneous expressive characteristics across modalities;
- **insufficient multimodal interaction**, caused by static or weakly adaptive relation modeling.

To tackle these issues, SENTI introduces two key components:

- **Modality-aware Semantic Enhancement (MSE)**: improves cross-modal semantic coherence by selecting a data-driven primary modality and performing semantics-guided distribution alignment.
- **Distribution-aware Relation Propagation (DRP)**: promotes more coherent cross-modal interaction by enforcing distributional consistency during relation propagation.

According to the paper, SENTI achieves competitive performance on three public datasets: **IEMOCAP**, **MELD**, and **MC-EIU**.

## Highlights

- A unified framework for multimodal semantic enhancement and relation propagation in conversation emotion recognition.
- A modality-aware semantic enhancement strategy for weak or misaligned modalities.
- A distribution-aware relation propagation mechanism for more coherent multimodal interaction.
- Competitive empirical results on public MERC benchmarks.

## Repository Status

This repository currently provides the runnable training and inference pipeline for:

- **IEMOCAP**
- **MELD**

The paper also reports results on **MC-EIU**. If you plan to extend the current codebase to MC-EIU, the main entry points are [reader.py](./reader.py), [preprocess.py](./preprocess.py), and [runner.py](./runner.py).

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

The code is intended to be run with `python3`.

## Backbone Preparation

Download the **T5-base** pretrained model and place the required files under:

```bash
./pretrained_model
```

Typical files include:

- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- tokenizer files such as `spiece.model` and `tokenizer.json`

## Data Preparation

This implementation uses **pre-extracted multimodal features** and expects a pre-built `encoded_data.pkl`.

The code looks for:

- `./IEMOCAP/encoded_data.pkl`, or
- `./MELD/encoded_data.pkl`, or
- `./encoded_data.pkl`

Current dataset routing in [main.py](./main.py) is:

- `iemocap -> ./IEMOCAP`
- `meld -> ./MELD`

If you need to regenerate the original json-style dialogue files, see [preprocess.py](./preprocess.py). The current training path, however, directly loads the encoded pickle file.

Following the paper, the feature setting is:

- **IEMOCAP**: text from RoBERTa, audio from OpenSmile, vision from 3D-CNN
- **MELD**: text from RoBERTa, audio from OpenSmile, vision from DenseNet

## Training

### IEMOCAP

```bash
python3 main.py \
  -backbone ./pretrained_model \
  -run_type train \
  -dataset iemocap \
  -use_gat \
  -window_size 8 \
  -gat 1 \
  -emotion_first \
  -use_video_mode \
  -use_audio_mode \
  -primary_modality auto \
  -mse_weight 0.001 \
  -drp_weight 0.001 \
  -swd_proj_k 4
```

### MELD

```bash
python3 main.py \
  -backbone ./pretrained_model \
  -run_type train \
  -dataset meld \
  -use_gat \
  -emotion_first \
  -use_video_mode \
  -use_audio_mode \
  -primary_modality auto \
  -mse_weight 0.001 \
  -drp_weight 0.001 \
  -swd_proj_k 4
```

## Inference

### IEMOCAP

```bash
python3 main.py \
  -run_type predict \
  -ckpt ./iemocap-best-model/ckpt \
  -output predict_real.json \
  -dataset iemocap \
  -test_batch_size 64
```

### MELD

```bash
python3 main.py \
  -run_type predict \
  -ckpt ./meld-best-model/ckpt \
  -output predict_real.json \
  -dataset meld \
  -test_batch_size 64
```

## Ablation Options

The current codebase exposes the two main SENTI modules as configurable options:

- Disable **MSE**:

```bash
-no_mse
```

- Disable **DRP**:

```bash
-no_drp
```

- Manually set the primary modality:

```bash
-primary_modality text
```

Available choices:

- `auto`
- `text`
- `audio`
- `video`

## Main Files

- [main.py](./main.py): training and prediction entry
- [runner.py](./runner.py): training loop, loss construction, and module integration
- [model.py](./model.py): model definition
- [reader.py](./reader.py): dataset reader and encoded-data loader
- [preprocess.py](./preprocess.py): dataset preprocessing utilities

## Notes

- Large local artifacts such as datasets, checkpoints, encoded pickles, and pretrained weights are not intended to be versioned with the repository.
- If training logs show newly initialized parameters, this is expected for task-specific modules added on top of the pretrained backbone.
- If you use HTTPS for GitHub pushes, use a GitHub Personal Access Token instead of your account password.

## Citation

If you find this repository useful, please cite the paper as:

```bibtex
@misc{zhang2025senti,
  title={SENTI: Semantic Enhancement and Relation Propagation Network for Multimodal Emotion Recognition in Conversations},
  author={Jian Zhang and Peizheng Zhao and Qiufeng Wang and Han Liu and Dongming Lu and Fangyu Wu},
  year={2025}
}
```
