import argparse
import csv
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSAIL_ROOT = os.path.join(PROJECT_ROOT, "csail_seg")
if CSAIL_ROOT not in sys.path:
	sys.path.insert(0, CSAIL_ROOT)

from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.lib.nn import user_scattered_collate
from mit_semseg.lib.utils import as_numpy
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion

from utils.metrics import boundary_iou, boundary_precision_recall_f1


def move_to_device(obj, device):
	if torch.is_tensor(obj):
		return obj.to(device, non_blocking=(device.type == "cuda"))
	if isinstance(obj, dict):
		return {k: move_to_device(v, device) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [move_to_device(v, device) for v in obj]
	return obj


def resolve_weights(args) -> Dict[str, str]:
	if args.weights_encoder and args.weights_decoder:
		return {
			"encoder": args.weights_encoder,
			"decoder": args.weights_decoder,
		}

	fallback_encoder = os.path.join(CSAIL_ROOT, "ckpt", "encoder_epoch_30.pth")
	fallback_decoder = os.path.join(CSAIL_ROOT, "ckpt", "decoder_epoch_30.pth")
	if os.path.exists(fallback_encoder) and os.path.exists(fallback_decoder):
		return {
			"encoder": fallback_encoder,
			"decoder": fallback_decoder,
		}

	checkpoint = args.checkpoint if args.checkpoint else cfg.VAL.checkpoint
	return {
		"encoder": os.path.join(cfg.DIR, "encoder_" + checkpoint),
		"decoder": os.path.join(cfg.DIR, "decoder_" + checkpoint),
	}


def prepare_dataset_paths() -> None:
	"""Fallback to CSAIL local data paths if config defaults do not exist."""
	def has_ade_content(root: str) -> bool:
		return os.path.exists(os.path.join(root, "ADEChallengeData2016"))

	if (not os.path.exists(cfg.DATASET.root_dataset)) or (not has_ade_content(cfg.DATASET.root_dataset)):
		candidate = os.path.join(CSAIL_ROOT, "data")
		if os.path.exists(candidate):
			cfg.DATASET.root_dataset = candidate

	if not os.path.exists(cfg.DATASET.list_train):
		cfg.DATASET.list_train = os.path.join(CSAIL_ROOT, "data", "training.odgt")

	if not os.path.exists(cfg.DATASET.list_val):
		cfg.DATASET.list_val = os.path.join(CSAIL_ROOT, "data", "validation.odgt")


def ensure_eval_metric_files(metrics_dir: str):
	os.makedirs(metrics_dir, exist_ok=True)
	csv_path = os.path.join(metrics_dir, "eval_metrics.csv")
	jsonl_path = os.path.join(metrics_dir, "eval_metrics.jsonl")

	header = [
		"run_name",
		"split_name",
		"mean_iou",
		"pixel_accuracy",
		"inference_time",
		"boundary_iou",
		"boundary_precision",
		"boundary_recall",
		"boundary_f1",
		"boundary_width",
		"weights_encoder",
		"weights_decoder",
	]
	if not os.path.exists(csv_path):
		with open(csv_path, "w", newline="", encoding="utf-8") as f:
			csv.writer(f).writerow(header)

	return csv_path, jsonl_path


def save_eval_metrics(args, weights: Dict[str, str], metrics: Dict[str, float], metrics_dir: str) -> None:
	csv_path, jsonl_path = ensure_eval_metric_files(metrics_dir)

	row = {
		"run_name": args.run_name,
		"split_name": args.split_name,
		"mean_iou": metrics["mean_iou"],
		"pixel_accuracy": metrics["pixel_accuracy"],
		"inference_time": metrics["inference_time"],
		"boundary_iou": metrics["boundary_iou"],
		"boundary_precision": metrics["boundary_precision"],
		"boundary_recall": metrics["boundary_recall"],
		"boundary_f1": metrics["boundary_f1"],
		"boundary_width": args.boundary_width,
		"weights_encoder": weights["encoder"],
		"weights_decoder": weights["decoder"],
	}

	with open(csv_path, "a", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow([
			row["run_name"],
			row["split_name"],
			row["mean_iou"],
			row["pixel_accuracy"],
			row["inference_time"],
			row["boundary_iou"],
			row["boundary_precision"],
			row["boundary_recall"],
			row["boundary_f1"],
			row["boundary_width"],
			row["weights_encoder"],
			row["weights_decoder"],
		])

	with open(jsonl_path, "a", encoding="utf-8") as f:
		f.write(json.dumps(row) + "\n")

	print("Saved eval metrics to {} and {}".format(csv_path, jsonl_path))


def validate_dataset_content(list_path: str) -> None:
	"""Fail early with a clear message if ADE images/annotations are missing."""
	if not os.path.exists(list_path):
		raise FileNotFoundError("Evaluation list not found: {}".format(list_path))

	with open(list_path, "r", encoding="utf-8") as f:
		first = json.loads(f.readline().rstrip())

	img_path = os.path.join(cfg.DATASET.root_dataset, first["fpath_img"])
	seg_path = os.path.join(cfg.DATASET.root_dataset, first["fpath_segm"])
	if (not os.path.exists(img_path)) or (not os.path.exists(seg_path)):
		raise RuntimeError(
			"ADE20K files are missing under root_dataset='{}'.\n"
			"Expected sample image: {}\n"
			"Expected sample label: {}\n"
			"Run: cd csail_seg && chmod +x download_ADE20K.sh && ./download_ADE20K.sh".format(
				cfg.DATASET.root_dataset,
				img_path,
				seg_path,
			)
		)


def evaluate(segmentation_module, loader, device, boundary_width: int):
	acc_meter = AverageMeter()
	intersection_meter = AverageMeter()
	union_meter = AverageMeter()
	time_meter = AverageMeter()

	boundary_iou_meter = AverageMeter()
	boundary_p_meter = AverageMeter()
	boundary_r_meter = AverageMeter()
	boundary_f1_meter = AverageMeter()

	segmentation_module.eval()

	for batch_data in loader:
		batch_data = batch_data[0]
		seg_label = as_numpy(batch_data["seg_label"][0])
		img_resized_list = batch_data["img_data"]

		if device.type == "cuda":
			torch.cuda.synchronize()
		tic = time.perf_counter()

		with torch.no_grad():
			seg_size = (seg_label.shape[0], seg_label.shape[1])
			scores = torch.zeros(1, cfg.DATASET.num_class, seg_size[0], seg_size[1], device=device)

			for img in img_resized_list:
				feed_dict = batch_data.copy()
				feed_dict["img_data"] = img
				del feed_dict["img_ori"]
				del feed_dict["info"]
				feed_dict = move_to_device(feed_dict, device)

				pred_tmp = segmentation_module(feed_dict, segSize=seg_size)
				scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

			_, pred = torch.max(scores, dim=1)
			pred = as_numpy(pred.squeeze(0).cpu())

		if device.type == "cuda":
			torch.cuda.synchronize()
		time_meter.update(time.perf_counter() - tic)

		# Standard segmentation metrics
		acc, pix = accuracy(pred, seg_label)
		intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
		acc_meter.update(acc, pix)
		intersection_meter.update(intersection)
		union_meter.update(union)

		# Boundary metrics
		pred_t = torch.from_numpy(pred)
		seg_t = torch.from_numpy(seg_label)
		b_iou = boundary_iou(pred_t, seg_t, boundary_width=boundary_width, ignore_index=-1)
		b_prf = boundary_precision_recall_f1(pred_t, seg_t, boundary_width=boundary_width, ignore_index=-1)
		boundary_iou_meter.update(b_iou)
		boundary_p_meter.update(b_prf["precision"])
		boundary_r_meter.update(b_prf["recall"])
		boundary_f1_meter.update(b_prf["f1"])

	iou = intersection_meter.sum / (union_meter.sum + 1e-10)
	print("[Eval Summary]")
	print("Mean IoU: {:.4f}".format(iou.mean()))
	print("Pixel Accuracy: {:.2f}%".format(acc_meter.average() * 100.0))
	print("Inference Time/Image: {:.4f}s".format(time_meter.average()))
	print("Boundary IoU: {:.4f}".format(boundary_iou_meter.average()))
	print("Boundary Precision: {:.4f}".format(boundary_p_meter.average()))
	print("Boundary Recall: {:.4f}".format(boundary_r_meter.average()))
	print("Boundary F1: {:.4f}".format(boundary_f1_meter.average()))

	return {
		"mean_iou": float(iou.mean()),
		"pixel_accuracy": float(acc_meter.average() * 100.0),
		"inference_time": float(time_meter.average()),
		"boundary_iou": float(boundary_iou_meter.average()),
		"boundary_precision": float(boundary_p_meter.average()),
		"boundary_recall": float(boundary_r_meter.average()),
		"boundary_f1": float(boundary_f1_meter.average()),
	}


def main(args):
	cfg.merge_from_file(args.cfg)
	cfg.merge_from_list(args.opts)
	prepare_dataset_paths()

	if args.list_path:
		cfg.DATASET.list_val = args.list_path

	validate_dataset_content(cfg.DATASET.list_val)

	if args.gpu >= 0 and torch.cuda.is_available():
		device = torch.device("cuda", args.gpu)
		torch.cuda.set_device(args.gpu)
	else:
		device = torch.device("cpu")

	weights = resolve_weights(args)
	if not os.path.exists(weights["encoder"]) or not os.path.exists(weights["decoder"]):
		raise FileNotFoundError(
			"Could not resolve checkpoint weights. "
			"Pass --weights-encoder and --weights-decoder explicitly."
		)

	net_encoder = ModelBuilder.build_encoder(
		arch=cfg.MODEL.arch_encoder.lower(),
		fc_dim=cfg.MODEL.fc_dim,
		weights=weights["encoder"],
	)
	net_decoder = ModelBuilder.build_decoder(
		arch=cfg.MODEL.arch_decoder.lower(),
		fc_dim=cfg.MODEL.fc_dim,
		num_class=cfg.DATASET.num_class,
		weights=weights["decoder"],
		use_softmax=True,
	)

	crit = nn.NLLLoss(ignore_index=-1)
	segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
	segmentation_module.to(device)

	dataset_val = ValDataset(
		cfg.DATASET.root_dataset,
		cfg.DATASET.list_val,
		cfg.DATASET,
	)
	loader_val = torch.utils.data.DataLoader(
		dataset_val,
		batch_size=cfg.VAL.batch_size,
		shuffle=False,
		collate_fn=user_scattered_collate,
		num_workers=min(4, os.cpu_count() or 1),
		drop_last=False,
	)

	metrics = evaluate(segmentation_module, loader_val, device, boundary_width=args.boundary_width)
	save_eval_metrics(args, weights, metrics, args.metrics_dir)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate ADE20K model with boundary metrics")
	parser.add_argument(
		"--cfg",
		default="csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml",
		type=str,
		help="Path to CSAIL config file",
	)
	parser.add_argument("--gpu", default=0, type=int, help="CUDA device id, use -1 for CPU")
	parser.add_argument("--checkpoint", default="", type=str, help="Checkpoint suffix like epoch_30.pth")
	parser.add_argument("--weights-encoder", default="", type=str, help="Explicit path to encoder checkpoint")
	parser.add_argument("--weights-decoder", default="", type=str, help="Explicit path to decoder checkpoint")
	parser.add_argument("--split-name", default="val", type=str, help="Split label written to metric files")
	parser.add_argument("--run-name", default="baseline", type=str, help="Run label written to metric files")
	parser.add_argument("--list-path", default="", type=str, help="Optional .odgt file to evaluate instead of cfg DATASET.list_val")
	parser.add_argument("--metrics-dir", default="metrics", type=str, help="Directory to store eval_metrics.csv/jsonl")
	parser.add_argument("--boundary-width", default=2, type=int, help="Boundary dilation width in pixels")
	parser.add_argument(
		"opts",
		default=None,
		nargs=argparse.REMAINDER,
		help="Modify config options using the command-line",
	)
	args = parser.parse_args()
	main(args)
