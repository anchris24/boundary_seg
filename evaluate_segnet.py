import argparse
import csv
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.segnet import SegNet
from utils.dataset import resolve_ade20k_paths
from utils.metrics import boundary_precision_recall_f1, confusion_matrix, segmentation_metrics_from_confusion

from csail_seg.mit_semseg.dataset import ValDataset, TestDataset
from csail_seg.mit_semseg.lib.nn import user_scattered_collate
from csail_seg.mit_semseg.config import cfg


def resolve_cfg_path(cfg_arg: str) -> str:
    if os.path.exists(cfg_arg):
        return cfg_arg
    candidate = os.path.join(PROJECT_ROOT, "csail_seg", "config", cfg_arg)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError("Could not find config file '{}'.".format(cfg_arg))


def prepare_dataset_paths() -> None:
    paths = resolve_ade20k_paths(PROJECT_ROOT)
    if not os.path.exists(cfg.DATASET.root_dataset) or not os.path.exists(os.path.join(cfg.DATASET.root_dataset, "ADEChallengeData2016")):
        cfg.DATASET.root_dataset = paths["data_root"]
    cfg.DATASET.list_train = paths["train_odgt"]
    cfg.DATASET.list_val = paths["val_odgt"]


def configure_segnet_defaults() -> None:
    cfg.DATASET.segm_downsampling_rate = 1
    cfg.DATASET.padding_constant = max(int(cfg.DATASET.padding_constant), 32)
    if tuple(cfg.DATASET.imgSizes) == (300, 375, 450, 525, 600):
        cfg.DATASET.imgSizes = (256,)
        cfg.DATASET.imgMaxSize = 512


def validate_dataset_content(list_path: str) -> None:
    with open(list_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline().rstrip())
    img_path = os.path.join(cfg.DATASET.root_dataset, first["fpath_img"])
    seg_path = os.path.join(cfg.DATASET.root_dataset, first["fpath_segm"])
    if not os.path.exists(img_path) or not os.path.exists(seg_path):
        raise RuntimeError(
            "ADE20K files are missing under root_dataset='{}'.\n"
            "Expected sample image: {}\n"
            "Expected sample label: {}".format(cfg.DATASET.root_dataset, img_path, seg_path)
        )


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=(device.type == "cuda"))
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [move_to_device(v, device) for v in obj]
    return obj


def build_metrics_files(metrics_dir: str):
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, "eval_metrics.csv")
    jsonl_path = os.path.join(metrics_dir, "eval_metrics.jsonl")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "run_name",
                "split_name",
                "pixel_accuracy",
                "mean_accuracy",
                "mean_iou",
                "weighted_iou",
                "boundary_iou",
                "boundary_precision",
                "boundary_recall",
                "boundary_f1",
                "boundary_width",
                "checkpoint",
            ])
    return csv_path, jsonl_path


def save_metrics(args, metrics: dict, csv_path: str, jsonl_path: str) -> None:
    row = {
        "run_name": args.run_name,
        "split_name": args.split_name,
        "pixel_accuracy": metrics["pixel_accuracy"],
        "mean_accuracy": metrics["mean_accuracy"],
        "mean_iou": metrics["mean_iou"],
        "weighted_iou": metrics["weighted_iou"],
        "boundary_iou": metrics["boundary_iou"],
        "boundary_precision": metrics["boundary_precision"],
        "boundary_recall": metrics["boundary_recall"],
        "boundary_f1": metrics["boundary_f1"],
        "boundary_width": args.boundary_width,
        "checkpoint": args.checkpoint,
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            row["run_name"], row["split_name"], row["pixel_accuracy"], row["mean_accuracy"], row["mean_iou"], row["weighted_iou"],
            row["boundary_iou"], row["boundary_precision"], row["boundary_recall"], row["boundary_f1"], row["boundary_width"], row["checkpoint"],
        ])
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_model(args, device):
    model = SegNet(num_class=cfg.DATASET.num_class, pretrained_encoder=False)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def evaluate(model, loader, device, boundary_width: int):
    model.eval()
    hist = None
    boundary_iou_values = []
    boundary_precision_values = []
    boundary_recall_values = []
    boundary_f1_values = []
    inference_times = []

    for batch_data in loader:
        batch_data = batch_data[0]
        seg_label = batch_data["seg_label"][0]
        img_list = batch_data["img_data"]
        seg_size = tuple(seg_label.shape[-2:])

        if device.type == "cuda":
            torch.cuda.synchronize()
        tic = time.perf_counter()

        with torch.no_grad():
            scores = None
            for img in img_list:
                feed = batch_data.copy()
                feed["img_data"] = img
                del feed["img_ori"]
                del feed["info"]
                feed = move_to_device(feed, device)
                logits = model(feed["img_data"], seg_size=seg_size)
                scores = logits if scores is None else scores + logits / len(img_list)
            pred = torch.argmax(scores, dim=1).squeeze(0).cpu()

        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - tic)

        batch_hist = confusion_matrix(pred, seg_label, num_class=cfg.DATASET.num_class, ignore_index=-1)
        hist = batch_hist if hist is None else hist + batch_hist

        b = boundary_precision_recall_f1(pred, seg_label, boundary_width=boundary_width, ignore_index=-1)
        boundary_precision_values.append(b["precision"])
        boundary_recall_values.append(b["recall"])
        boundary_f1_values.append(b["f1"])

        # Boundary IoU for the current sample.
        tp_fp_fn_iou = (b["precision"], b["recall"], b["f1"])
        # Recompute boundary IoU directly from the current sample for logging consistency.
        # boundary IoU = TP / (TP + FP + FN)
        # We derive it from precision/recall via the usual boundary F1 relation is not stable enough,
        # so compute it from the confusion on-the-fly below.
        # Keep a direct scalar by reusing the boundary helper.
        # (This call is cheap compared with the model forward.)
        from utils.metrics import boundary_iou as _boundary_iou
        boundary_iou_values.append(_boundary_iou(pred, seg_label, boundary_width=boundary_width, ignore_index=-1))

    summary = segmentation_metrics_from_confusion(hist) if hist is not None else {
        "pixel_accuracy": 0.0,
        "mean_accuracy": 0.0,
        "mean_iou": 0.0,
        "weighted_iou": 0.0,
    }
    pixel_accuracy = summary["pixel_accuracy"]
    mean_accuracy = summary["mean_accuracy"]
    mean_iou = summary["mean_iou"]
    weighted_iou = summary["weighted_iou"]
    inference_time = float(sum(inference_times) / max(len(inference_times), 1))
    boundary_iou_mean = float(sum(boundary_iou_values) / max(len(boundary_iou_values), 1))
    boundary_precision_mean = float(sum(boundary_precision_values) / max(len(boundary_precision_values), 1))
    boundary_recall_mean = float(sum(boundary_recall_values) / max(len(boundary_recall_values), 1))
    boundary_f1_mean = float(sum(boundary_f1_values) / max(len(boundary_f1_values), 1))

    print("[Eval Summary]")
    print("Pixel Accuracy: {:.4f}".format(pixel_accuracy))
    print("Mean Accuracy: {:.4f}".format(mean_accuracy))
    print("Mean IoU: {:.4f}".format(mean_iou))
    print("Weighted IoU: {:.4f}".format(weighted_iou))
    print("Inference Time/Image: {:.4f}s".format(inference_time))
    print("Boundary IoU: {:.4f}".format(boundary_iou_mean))
    print("Boundary Precision: {:.4f}".format(boundary_precision_mean))
    print("Boundary Recall: {:.4f}".format(boundary_recall_mean))
    print("Boundary F1: {:.4f}".format(boundary_f1_mean))

    boundary_summary = {
        "pixel_accuracy": float(pixel_accuracy),
        "mean_accuracy": float(mean_accuracy),
        "mean_iou": float(mean_iou),
        "weighted_iou": float(weighted_iou),
        "boundary_iou": boundary_iou_mean,
        "boundary_precision": boundary_precision_mean,
        "boundary_recall": boundary_recall_mean,
        "boundary_f1": boundary_f1_mean,
        "inference_time": inference_time,
    }
    return boundary_summary


def main(args):
    cfg_path = resolve_cfg_path(args.cfg)
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(args.opts)

    prepare_dataset_paths()
    configure_segnet_defaults()
    if args.list_path:
        cfg.DATASET.list_val = args.list_path
    validate_dataset_content(cfg.DATASET.list_val)

    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
        torch.cuda.set_device(args.gpu)

    model = load_model(args, device)

    dataset = ValDataset(cfg.DATASET.root_dataset, cfg.DATASET.list_val, cfg.DATASET)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=args.workers,
        drop_last=False,
    )

    metrics = evaluate(model, loader, device, boundary_width=args.boundary_width)
    csv_path, jsonl_path = build_metrics_files(args.metrics_dir)
    save_metrics(args, metrics, csv_path, jsonl_path)
    print("Saved eval metrics to {} and {}".format(csv_path, jsonl_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegNet evaluation for ADE20K")
    parser.add_argument("--cfg", default="csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml", type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to a saved segnet checkpoint")
    parser.add_argument("--split-name", default="val", type=str)
    parser.add_argument("--run-name", default="segnet_baseline", type=str)
    parser.add_argument("--list-path", default="", type=str)
    parser.add_argument("--metrics-dir", default="metrics", type=str)
    parser.add_argument("--boundary-width", default=2, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
