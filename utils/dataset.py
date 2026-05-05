import json
import os
from typing import Dict, List, Tuple


def resolve_ade20k_paths(project_root: str) -> Dict[str, str]:
	"""Return canonical ADE20K paths used by this project.

	The repository contains the CSAIL codebase inside `csail_seg/`, so we keep
	path resolution in one place for training/evaluation scripts.
	"""
	csail_root = os.path.join(project_root, "csail_seg")
	data_root = os.path.join(csail_root, "data")
	return {
		"project_root": project_root,
		"csail_root": csail_root,
		"data_root": data_root,
		"train_odgt": os.path.join(data_root, "training.odgt"),
		"val_odgt": os.path.join(data_root, "validation.odgt"),
		"color150": os.path.join(data_root, "color150.mat"),
		"object_info": os.path.join(data_root, "object150_info.csv"),
		"default_encoder_ckpt": os.path.join(csail_root, "ckpt", "encoder_epoch_30.pth"),
		"default_decoder_ckpt": os.path.join(csail_root, "ckpt", "decoder_epoch_30.pth"),
	}


def validate_ade20k_layout(project_root: str) -> Tuple[bool, List[str]]:
	"""Check whether required ADE20K metadata/checkpoint files are present."""
	paths = resolve_ade20k_paths(project_root)
	required = [
		paths["train_odgt"],
		paths["val_odgt"],
		paths["color150"],
		paths["object_info"],
	]
	missing = [p for p in required if not os.path.exists(p)]
	return len(missing) == 0, missing


def preview_odgt(odgt_path: str, limit: int = 3) -> List[dict]:
	"""Load up to `limit` JSONL rows from an .odgt file for quick inspection."""
	rows = []
	if not os.path.exists(odgt_path):
		return rows

	with open(odgt_path, "r", encoding="utf-8") as f:
		for _, line in zip(range(limit), f):
			rows.append(json.loads(line.rstrip()))
	return rows
