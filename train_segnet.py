import argparse
import csv
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.segnet import SegNet
from utils.dataset import resolve_ade20k_paths
from utils.losses import boundary_aware_ce_loss

from csail_seg.mit_semseg.dataset import TrainDataset
from csail_seg.mit_semseg.lib.nn import user_scattered_collate
from csail_seg.mit_semseg.config import cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_metrics_files(output_dir: str, run_name: str):
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    iter_csv = os.path.join(metrics_dir, "train_iter_metrics.csv")
    epoch_csv = os.path.join(metrics_dir, "train_epoch_metrics.csv")
    meta_json = os.path.join(metrics_dir, "run_metadata.json")
    if not os.path.exists(iter_csv):
        with open(iter_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["run_name", "epoch", "iter", "global_iter", "loss", "acc", "lr", "time_avg", "data_time_avg"])
    if not os.path.exists(epoch_csv):
        with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["run_name", "epoch", "loss_avg", "acc_avg", "time_avg", "data_time_avg"])
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump({"run_name": run_name, "created_at_unix": time.time(), "model": "segnet"}, f, indent=2)
    return iter_csv, epoch_csv


def adjust_poly_lr(optimizer, base_lr: float, cur_iter: int, max_iter: int, power: float):
    lr = base_lr * ((1.0 - float(cur_iter) / max_iter) ** power)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def set_bn_eval(module) -> None:
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_one_epoch(model, loader, optimizer, device, args, epoch, max_iter, iter_csv_path, epoch_csv_path):
    model.train()
    if args.freeze_bn or args.batch_size_per_gpu < 2:
        set_bn_eval(model)

    batch_time = 0.0
    data_time = 0.0
    running_loss = 0.0
    running_acc = 0.0
    rows = []

    tic = time.time()
    iterator = iter(loader)
    for i in range(args.epoch_iters):
        batch_data = next(iterator)[0]
        data_time += time.time() - tic
        batch_data = move_to_device(batch_data, device)
        img = batch_data["img_data"]
        seg_label = batch_data["seg_label"]

        cur_iter = i + (epoch - 1) * args.epoch_iters
        lr = adjust_poly_lr(optimizer, args.lr, cur_iter, max_iter, args.lr_pow)

        optimizer.zero_grad(set_to_none=True)
        logits = model(img, seg_size=tuple(seg_label.shape[-2:]))
        loss = boundary_aware_ce_loss(
            logits,
            seg_label,
            lambda_boundary=args.lambda_boundary,
            boundary_width=args.boundary_width,
            ignore_index=-1,
        )

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        else:
            print("Warning: non-finite loss encountered at epoch {}, iter {}; skipping update.".format(epoch, i))
            continue

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            valid = seg_label >= 0
            correct = ((pred == seg_label) & valid).float().sum()
            denom = valid.float().sum().clamp_min(1.0)
            acc = (correct / denom).item() * 100.0

        running_loss += loss.item()
        running_acc += acc
        batch_time += time.time() - tic
        tic = time.time()
        rows.append([args.run_name, epoch, i, cur_iter, loss.item(), acc, lr, batch_time / (i + 1), data_time / (i + 1)])

        if i % args.disp_iter == 0:
            print(
                "Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr: {:.6f}, Acc: {:.2f}, Loss: {:.6f}".format(
                    epoch, i, args.epoch_iters, batch_time / (i + 1), data_time / (i + 1), lr, acc, loss.item()
                )
            )

    with open(iter_csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    with open(epoch_csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            args.run_name,
            epoch,
            running_loss / max(args.epoch_iters, 1),
            running_acc / max(args.epoch_iters, 1),
            batch_time / max(args.epoch_iters, 1),
            data_time / max(args.epoch_iters, 1),
        ])


def save_checkpoint(model, out_dir: str, epoch: int):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "segnet_epoch_{}.pth".format(epoch)))


def main(args):
    cfg_path = resolve_cfg_path(args.cfg)
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(args.opts)

    prepare_dataset_paths()
    configure_segnet_defaults()
    validate_dataset_content(cfg.DATASET.list_train)

    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
        torch.cuda.set_device(args.gpu)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    iter_csv_path, epoch_csv_path = build_metrics_files(args.output_dir, args.run_name)

    model = SegNet(num_class=cfg.DATASET.num_class, pretrained_encoder=True).to(device)
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=args.batch_size_per_gpu,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    max_iter = args.num_epoch * args.epoch_iters

    for epoch in range(1, args.num_epoch + 1):
        train_one_epoch(model, loader_train, optimizer, device, args, epoch, max_iter, iter_csv_path, epoch_csv_path)
        if epoch % args.save_every == 0:
            save_checkpoint(model, args.output_dir, epoch)

    print("Training complete. Checkpoints saved under {}".format(args.output_dir))
    print("Train metrics written to {} and {}".format(iter_csv_path, epoch_csv_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegNet baseline training for ADE20K")
    parser.add_argument("--cfg", default="csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml", type=str)
    parser.add_argument("--gpu", default=-1, type=int, help="CUDA device id, use -1 for CPU")
    parser.add_argument("--output-dir", default="ckpt/segnet_baseline", type=str)
    parser.add_argument("--run-name", default="segnet_baseline", type=str)
    parser.add_argument("--num-epoch", default=2, type=int)
    parser.add_argument("--epoch-iters", default=200, type=int)
    parser.add_argument("--batch-size-per-gpu", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--save-every", default=1, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--lr-pow", default=0.9, type=float)
    parser.add_argument("--lambda-boundary", default=0.0, type=float)
    parser.add_argument("--boundary-width", default=2, type=int)
    parser.add_argument("--seed", default=304, type=int)
    parser.add_argument("--disp-iter", default=10, type=int)
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
