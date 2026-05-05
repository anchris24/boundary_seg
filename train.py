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
CSAIL_ROOT = os.path.join(PROJECT_ROOT, "csail_seg")
if CSAIL_ROOT not in sys.path:
	sys.path.insert(0, CSAIL_ROOT)

from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset
from mit_semseg.lib.nn import user_scattered_collate
from mit_semseg.models import ModelBuilder

from utils.losses import combined_segmentation_loss


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def adjust_poly_lr(optimizer, base_lr: float, cur_iter: int, max_iter: int, power: float):
	lr = base_lr * ((1.0 - float(cur_iter) / max_iter) ** power)
	for group in optimizer.param_groups:
		group["lr"] = lr
	return lr


def resolve_init_weights(args):
	if args.init_encoder and args.init_decoder:
		return args.init_encoder, args.init_decoder

	fallback_encoder = os.path.join(CSAIL_ROOT, "ckpt", "encoder_epoch_30.pth")
	fallback_decoder = os.path.join(CSAIL_ROOT, "ckpt", "decoder_epoch_30.pth")
	if os.path.exists(fallback_encoder) and os.path.exists(fallback_decoder):
		return fallback_encoder, fallback_decoder

	# Empty string means random init from torchvision pretrain inside ModelBuilder.
	return "", ""


def build_models_with_fallback(cfg, init_encoder: str, init_decoder: str):
	"""Build models and recover gracefully from incompatible checkpoint shapes."""
	try:
		net_encoder = ModelBuilder.build_encoder(
			arch=cfg.MODEL.arch_encoder.lower(),
			fc_dim=cfg.MODEL.fc_dim,
			weights=init_encoder,
		)
	except RuntimeError as e:
		print("Warning: encoder checkpoint load failed ({}). Retrying without encoder weights.".format(e))
		net_encoder = ModelBuilder.build_encoder(
			arch=cfg.MODEL.arch_encoder.lower(),
			fc_dim=cfg.MODEL.fc_dim,
			weights="",
		)

	try:
		net_decoder = ModelBuilder.build_decoder(
			arch=cfg.MODEL.arch_decoder.lower(),
			fc_dim=cfg.MODEL.fc_dim,
			num_class=cfg.DATASET.num_class,
			weights=init_decoder,
			use_softmax=False,
		)
	except RuntimeError as e:
		print("Warning: decoder checkpoint load failed ({}). Retrying without decoder weights.".format(e))
		net_decoder = ModelBuilder.build_decoder(
			arch=cfg.MODEL.arch_decoder.lower(),
			fc_dim=cfg.MODEL.fc_dim,
			num_class=cfg.DATASET.num_class,
			weights="",
			use_softmax=False,
		)

	return net_encoder, net_decoder


def build_optimizer(net_encoder, net_decoder, lr_encoder: float, lr_decoder: float, momentum: float, weight_decay: float):
	params = [
		{"params": net_encoder.parameters(), "lr": lr_encoder},
		{"params": net_decoder.parameters(), "lr": lr_decoder},
	]
	return torch.optim.SGD(params, momentum=momentum, weight_decay=weight_decay)


def effective_lr_scale(batch_size_per_gpu: int) -> float:
	"""Reduce learning rate for very small batches to improve stability."""
	if batch_size_per_gpu >= 2:
		return 1.0
	return 0.1


def move_to_device(obj, device):
	if torch.is_tensor(obj):
		return obj.to(device, non_blocking=(device.type == "cuda"))
	if isinstance(obj, dict):
		return {k: move_to_device(v, device) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [move_to_device(v, device) for v in obj]
	return obj


def set_batchnorm_eval(module) -> None:
	for m in module.modules():
		if isinstance(m, nn.modules.batchnorm._BatchNorm):
			m.eval()


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


def resolve_cfg_path(cfg_arg: str) -> str:
	"""Accept absolute, relative, or shorthand cfg names.

	Examples:
	- csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml
	- ade20k-resnet50dilated-ppm_deepsup.yaml
	"""
	if os.path.exists(cfg_arg):
		return cfg_arg

	candidate = os.path.join(CSAIL_ROOT, "config", cfg_arg)
	if os.path.exists(candidate):
		return candidate

	raise FileNotFoundError(
		"Could not find config file '{}'. Try --cfg csail_seg/config/<name>.yaml".format(cfg_arg)
	)


def prepare_metrics_files(output_dir: str, run_name: str):
	metrics_dir = os.path.join(output_dir, "metrics")
	os.makedirs(metrics_dir, exist_ok=True)

	iter_csv = os.path.join(metrics_dir, "train_iter_metrics.csv")
	epoch_csv = os.path.join(metrics_dir, "train_epoch_metrics.csv")
	meta_json = os.path.join(metrics_dir, "run_metadata.json")

	iter_header = [
		"run_name", "epoch", "iter", "global_iter", "loss", "acc", "lr_encoder", "lr_decoder", "time_avg", "data_time_avg"
	]
	epoch_header = ["run_name", "epoch", "loss_avg", "acc_avg", "time_avg", "data_time_avg"]

	if not os.path.exists(iter_csv):
		with open(iter_csv, "w", newline="", encoding="utf-8") as f:
			csv.writer(f).writerow(iter_header)
	if not os.path.exists(epoch_csv):
		with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
			csv.writer(f).writerow(epoch_header)

	with open(meta_json, "w", encoding="utf-8") as f:
		json.dump({"run_name": run_name, "created_at_unix": time.time()}, f, indent=2)

	return iter_csv, epoch_csv


def validate_dataset_content() -> None:
	"""Fail early with a clear message if ADE images/annotations are missing."""
	if not os.path.exists(cfg.DATASET.list_train):
		raise FileNotFoundError("Training list not found: {}".format(cfg.DATASET.list_train))

	with open(cfg.DATASET.list_train, "r", encoding="utf-8") as f:
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


def run_one_epoch(
	net_encoder,
	net_decoder,
	loader,
	optimizer,
	device,
	args,
	epoch: int,
	max_iter: int,
	iter_csv_path: str,
	epoch_csv_path: str,
):
	net_encoder.train()
	net_decoder.train()
	if args.freeze_bn or args.batch_size_per_gpu < 2:
		set_batchnorm_eval(net_encoder)
		set_batchnorm_eval(net_decoder)

	batch_time = 0.0
	data_time = 0.0
	running_loss = 0.0
	running_acc = 0.0

	tic = time.time()
	iterator = iter(loader)
	rows = []
	for i in range(args.epoch_iters):
		batch_data = next(iterator)[0]
		data_time += time.time() - tic

		batch_data = move_to_device(batch_data, device)
		img = batch_data["img_data"]
		seg_label = batch_data["seg_label"]

		cur_iter = i + (epoch - 1) * args.epoch_iters
		lr_scale = effective_lr_scale(args.batch_size_per_gpu)
		lr_enc = adjust_poly_lr(optimizer, args.lr_encoder * lr_scale, cur_iter, max_iter, args.lr_pow)
		lr_dec = args.lr_decoder * lr_scale * ((1.0 - float(cur_iter) / max_iter) ** args.lr_pow)
		optimizer.param_groups[1]["lr"] = lr_dec

		optimizer.zero_grad()

		encoder_feats = net_encoder(img, return_feature_maps=True)
		decoder_out = net_decoder(encoder_feats)

		if isinstance(decoder_out, tuple):
			pred, pred_deepsup = decoder_out
		else:
			pred, pred_deepsup = decoder_out, None

		main_loss = combined_segmentation_loss(
			pred,
			seg_label,
			lambda_boundary=args.lambda_boundary,
			boundary_width=args.boundary_width,
			ignore_index=-1,
		)
		total_loss = main_loss

		if pred_deepsup is not None and args.deep_sup_scale > 0:
			deep_loss = combined_segmentation_loss(
				pred_deepsup,
				seg_label,
				lambda_boundary=args.lambda_boundary,
				boundary_width=args.boundary_width,
				ignore_index=-1,
			)
			total_loss = total_loss + args.deep_sup_scale * deep_loss

		if not torch.isfinite(total_loss):
			print("Warning: non-finite loss encountered at epoch {}, iter {}; skipping update.".format(epoch, i))
			optimizer.zero_grad(set_to_none=True)
			continue

		total_loss.backward()
		torch.nn.utils.clip_grad_norm_(list(net_encoder.parameters()) + list(net_decoder.parameters()), max_norm=5.0)
		optimizer.step()

		with torch.no_grad():
			_, pred_label = torch.max(pred, dim=1)
			valid = (seg_label >= 0)
			correct = ((pred_label == seg_label) & valid).float().sum()
			denom = valid.float().sum().clamp_min(1.0)
			acc = (correct / denom).item() * 100.0

		running_loss += total_loss.item()
		running_acc += acc
		batch_time += time.time() - tic
		tic = time.time()
		rows.append([
			args.run_name,
			epoch,
			i,
			cur_iter,
			total_loss.item(),
			acc,
			lr_enc,
			lr_dec,
			batch_time / (i + 1),
			data_time / (i + 1),
		])

		if i % args.disp_iter == 0:
			avg_loss = running_loss / (i + 1)
			avg_acc = running_acc / (i + 1)
			avg_batch_time = batch_time / (i + 1)
			avg_data_time = data_time / (i + 1)
			print(
				"Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr_enc: {:.6f}, lr_dec: {:.6f}, "
				"Acc: {:.2f}, Loss: {:.6f}".format(
					epoch,
					i,
					args.epoch_iters,
					avg_batch_time,
					avg_data_time,
					lr_enc,
					lr_dec,
					avg_acc,
					avg_loss,
				)
			)

	with open(iter_csv_path, "a", newline="", encoding="utf-8") as f:
		csv.writer(f).writerows(rows)

	epoch_row = [
		args.run_name,
		epoch,
		running_loss / max(args.epoch_iters, 1),
		running_acc / max(args.epoch_iters, 1),
		batch_time / max(args.epoch_iters, 1),
		data_time / max(args.epoch_iters, 1),
	]
	with open(epoch_csv_path, "a", newline="", encoding="utf-8") as f:
		csv.writer(f).writerow(epoch_row)


def save_checkpoint(net_encoder, net_decoder, out_dir: str, epoch: int):
	os.makedirs(out_dir, exist_ok=True)
	torch.save(net_encoder.state_dict(), os.path.join(out_dir, "encoder_epoch_{}.pth".format(epoch)))
	torch.save(net_decoder.state_dict(), os.path.join(out_dir, "decoder_epoch_{}.pth".format(epoch)))


def main(args):
	cfg_path = resolve_cfg_path(args.cfg)
	cfg.merge_from_file(cfg_path)
	cfg.merge_from_list(args.opts)

	if args.gpu >= 0 and torch.cuda.is_available():
		device = torch.device("cuda", args.gpu)
		torch.cuda.set_device(args.gpu)
	else:
		device = torch.device("cpu")

	set_seed(args.seed)
	prepare_dataset_paths()
	validate_dataset_content()
	os.makedirs(args.output_dir, exist_ok=True)
	iter_csv_path, epoch_csv_path = prepare_metrics_files(args.output_dir, args.run_name)

	init_encoder, init_decoder = resolve_init_weights(args)
	net_encoder, net_decoder = build_models_with_fallback(cfg, init_encoder, init_decoder)
	net_encoder.to(device)
	net_decoder.to(device)

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

	optimizer = build_optimizer(
		net_encoder,
		net_decoder,
		lr_encoder=args.lr_encoder,
		lr_decoder=args.lr_decoder,
		momentum=args.beta1,
		weight_decay=args.weight_decay,
	)

	max_iter = args.num_epoch * args.epoch_iters
	for epoch in range(1, args.num_epoch + 1):
		run_one_epoch(
			net_encoder,
			net_decoder,
			loader_train,
			optimizer,
			device,
			args,
			epoch,
			max_iter,
			iter_csv_path,
			epoch_csv_path,
		)
		if (epoch % args.save_every) == 0:
			save_checkpoint(net_encoder, net_decoder, args.output_dir, epoch)

	print("Training complete. Checkpoints saved under {}".format(args.output_dir))
	print("Train metrics written to {} and {}".format(iter_csv_path, epoch_csv_path))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Boundary-aware ADE20K training wrapper")
	parser.add_argument("--cfg", default="csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml", type=str)
	parser.add_argument("--gpu", default=0, type=int, help="CUDA device id, use -1 for CPU")

	parser.add_argument("--output-dir", default="ckpt/boundary_experiments", type=str)
	parser.add_argument("--run-name", default="baseline", type=str, help="Name used in saved metric files")
	parser.add_argument("--init-encoder", default="", type=str)
	parser.add_argument("--init-decoder", default="", type=str)

	parser.add_argument("--num-epoch", default=2, type=int)
	parser.add_argument("--epoch-iters", default=200, type=int)
	parser.add_argument("--batch-size-per-gpu", default=1, type=int)
	parser.add_argument("--workers", default=2, type=int)
	parser.add_argument("--save-every", default=1, type=int)

	parser.add_argument("--lr-encoder", default=0.01, type=float)
	parser.add_argument("--lr-decoder", default=0.01, type=float)
	parser.add_argument("--lr-pow", default=0.9, type=float)
	parser.add_argument("--beta1", default=0.9, type=float)
	parser.add_argument("--weight-decay", default=1e-4, type=float)

	parser.add_argument("--deep-sup-scale", default=0.4, type=float)
	parser.add_argument("--lambda-boundary", default=0.0, type=float)
	parser.add_argument("--boundary-width", default=2, type=int)
	parser.add_argument("--seed", default=304, type=int)
	parser.add_argument("--disp-iter", default=20, type=int)
	parser.add_argument("--freeze-bn", action="store_true", help="Keep BatchNorm layers in eval mode during training")

	parser.add_argument(
		"opts",
		default=None,
		nargs=argparse.REMAINDER,
		help="Modify config options using the command-line",
	)

	args = parser.parse_args()
	main(args)
