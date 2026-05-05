# boundary_seg

6.S058 Project: Scene Segmentation

Use ADE20K data set with the most common 150 object annotations to evaluate a scene segmentation model that particularly optimizes accuracy on object boundaries by leveraging a hybrid metric that penalizes mislabeling of boundary pixels.

## Quick Start Workflow

This repository wraps the CSAIL ADE20K implementation under `csail_seg/` and adds:

- boundary-aware training loss in [utils/losses.py](utils/losses.py)
- boundary-focused evaluation metrics in [utils/metrics.py](utils/metrics.py)
- simple experiment entrypoints in [train.py](train.py) and [evaluate.py](evaluate.py)

## 1) Environment Setup

From the repo root:

```bash
source venv/bin/activate
pip install -r csail_seg/requirements.txt
```

## 2) ADE20K Data + Metadata

Download ADE20K and metadata files used by CSAIL code:

```bash
cd csail_seg
chmod +x download_ADE20K.sh
./download_ADE20K.sh
cd ..
```

Expected key files:

- `csail_seg/data/training.odgt`
- `csail_seg/data/validation.odgt`
- `csail_seg/data/color150.mat`
- `csail_seg/data/object150_info.csv`

## 3) Baseline Evaluation (Pretrained)

Evaluate the pretrained checkpoint and report both standard and boundary metrics:

```bash
python evaluate.py \
	--cfg csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml \
	--gpu 0 \
	--boundary-width 2
```

Notes:

- By default, [evaluate.py](evaluate.py) will use `csail_seg/ckpt/encoder_epoch_30.pth` and `csail_seg/ckpt/decoder_epoch_30.pth` when present.
- You can pass explicit checkpoints with `--weights-encoder` and `--weights-decoder`.

## 4) Baseline Finetuning (No Boundary Loss)

Use `lambda_boundary=0.0` as the control experiment:

```bash
python train.py \
	--cfg csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml \
	--gpu 0 \
	--output-dir ckpt/exp_baseline \
	--num-epoch 2 \
	--epoch-iters 200 \
	--lambda-boundary 0.0 \
	--boundary-width 2
```

## 5) Boundary-Aware Finetuning

Enable the boundary term in the hybrid loss:

```bash
python train.py \
	--cfg csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml \
	--gpu 0 \
	--output-dir ckpt/exp_boundary \
	--num-epoch 2 \
	--epoch-iters 200 \
	--lambda-boundary 0.5 \
	--boundary-width 2
```

Hybrid objective:

$$
L = L_{seg} + \lambda_{boundary} L_{boundary}
$$

where boundary pixels are upweighted by `(1 + lambda_boundary)`.

## 6) Compare Results

Evaluate both checkpoints with [evaluate.py](evaluate.py) and compare:

- Mean IoU
- Pixel Accuracy
- Boundary IoU
- Boundary Precision / Recall / F1

Start with a small sweep:

- `lambda_boundary`: `0.1, 0.3, 0.5, 1.0`
- `boundary_width`: `1, 2, 3`

## 7) Suggested Reporting Template

Track each run with:

- config name
- checkpoint path
- lambda_boundary
- boundary_width
- mIoU
- pixel accuracy
- boundary IoU
- boundary F1

This gives a clean baseline-vs-boundary comparison for your exploratory study.

## SegNet Baseline

For a closer ADE20K baseline, use the new SegNet path:

```bash
python train_segnet.py \
	--cfg csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml \
	--gpu -1 \
	--output-dir ckpt/segnet_baseline \
	--run-name segnet_baseline \
	--num-epoch 2 \
	--epoch-iters 200 \
	--lambda-boundary 0.0
```

Then evaluate a saved checkpoint with the paper metrics:

```bash
python evaluate_segnet.py \
	--cfg csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml \
	--gpu -1 \
	--checkpoint ckpt/segnet_baseline/segnet_epoch_2.pth \
	--run-name segnet_baseline \
	--split-name val \
	--metrics-dir metrics
```

ADE20K paper metrics reported by the new evaluation script:

- Pixel accuracy
- Mean accuracy
- Mean IoU
- Weighted IoU
