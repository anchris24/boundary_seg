from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F


def _to_nhw(labels: torch.Tensor) -> torch.Tensor:
	if labels.ndim == 2:
		return labels.unsqueeze(0)
	if labels.ndim == 3:
		return labels
	raise ValueError("Expected labels with shape (H, W) or (N, H, W)")


def _boundary_mask(labels: torch.Tensor, boundary_width: int = 1, ignore_index: int = -1) -> torch.Tensor:
	labels = _to_nhw(labels)
	n, h, w = labels.shape

	pad = F.pad(labels.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=float(ignore_index)).squeeze(1).long()
	center = labels
	up = pad[:, 0:h, 1:w + 1]
	down = pad[:, 2:h + 2, 1:w + 1]
	left = pad[:, 1:h + 1, 0:w]
	right = pad[:, 1:h + 1, 2:w + 2]

	valid_center = center != ignore_index
	edge = torch.zeros_like(center, dtype=torch.bool)
	for nb in (up, down, left, right):
		valid_nb = nb != ignore_index
		edge |= valid_center & valid_nb & (nb != center)

	if boundary_width <= 1:
		return edge
	dilated = F.max_pool2d(edge.float().unsqueeze(1), kernel_size=2 * boundary_width + 1, stride=1, padding=boundary_width)
	return dilated.squeeze(1) > 0


def boundary_confusion(
	pred_labels: torch.Tensor,
	gt_labels: torch.Tensor,
	boundary_width: int = 1,
	ignore_index: int = -1,
) -> Dict[str, float]:
	"""Compute TP/FP/FN counts for boundary predictions."""
	pred = _to_nhw(pred_labels)
	gt = _to_nhw(gt_labels)
	if pred.shape != gt.shape:
		raise ValueError("Prediction and ground-truth shapes must match")

	pred_b = _boundary_mask(pred, boundary_width=boundary_width, ignore_index=ignore_index)
	gt_b = _boundary_mask(gt, boundary_width=boundary_width, ignore_index=ignore_index)

	valid = gt != ignore_index
	tp = (pred_b & gt_b & valid).sum().item()
	fp = (pred_b & (~gt_b) & valid).sum().item()
	fn = ((~pred_b) & gt_b & valid).sum().item()
	return {"tp": float(tp), "fp": float(fp), "fn": float(fn)}


def boundary_precision_recall_f1(
	pred_labels: torch.Tensor,
	gt_labels: torch.Tensor,
	boundary_width: int = 1,
	ignore_index: int = -1,
) -> Dict[str, float]:
	stats = boundary_confusion(
		pred_labels,
		gt_labels,
		boundary_width=boundary_width,
		ignore_index=ignore_index,
	)
	tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
	precision = tp / (tp + fp + 1e-10)
	recall = tp / (tp + fn + 1e-10)
	f1 = (2.0 * precision * recall) / (precision + recall + 1e-10)
	return {
		"precision": precision,
		"recall": recall,
		"f1": f1,
	}


def boundary_iou(
	pred_labels: torch.Tensor,
	gt_labels: torch.Tensor,
	boundary_width: int = 1,
	ignore_index: int = -1,
) -> float:
	stats = boundary_confusion(
		pred_labels,
		gt_labels,
		boundary_width=boundary_width,
		ignore_index=ignore_index,
	)
	tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
	return tp / (tp + fp + fn + 1e-10)


def confusion_matrix(
	pred_labels: Union[np.ndarray, torch.Tensor],
	gt_labels: Union[np.ndarray, torch.Tensor],
	num_class: int,
	ignore_index: int = -1,
) -> np.ndarray:
	"""Compute a dense confusion matrix for semantic segmentation."""
	pred = pred_labels.detach().cpu().numpy() if torch.is_tensor(pred_labels) else np.asarray(pred_labels)
	gt = gt_labels.detach().cpu().numpy() if torch.is_tensor(gt_labels) else np.asarray(gt_labels)
	valid = gt != ignore_index
	pred = pred[valid].astype(np.int64)
	gt = gt[valid].astype(np.int64)
	mask = (gt >= 0) & (gt < num_class)
	gt = gt[mask]
	pred = pred[mask]
	indices = num_class * gt + pred
	hist = np.bincount(indices, minlength=num_class * num_class)
	return hist.reshape(num_class, num_class)


def segmentation_metrics_from_confusion(hist: np.ndarray) -> Dict[str, float]:
	"""Compute paper metrics from a confusion matrix."""
	hist = hist.astype(np.float64)
	true_pos = np.diag(hist)
	gt_pixels = hist.sum(axis=1)
	pred_pixels = hist.sum(axis=0)
	total_pixels = hist.sum()

	pixel_accuracy = true_pos.sum() / (total_pixels + 1e-10)
	class_acc = np.divide(true_pos, gt_pixels + 1e-10)
	class_iou = np.divide(true_pos, gt_pixels + pred_pixels - true_pos + 1e-10)
	valid_classes = gt_pixels > 0
	mean_accuracy = np.nanmean(class_acc[valid_classes]) if np.any(valid_classes) else 0.0
	mean_iou = np.nanmean(class_iou[valid_classes]) if np.any(valid_classes) else 0.0
	weighted_iou = np.nansum(class_iou * (gt_pixels / (total_pixels + 1e-10)))

	return {
		"pixel_accuracy": float(pixel_accuracy),
		"mean_accuracy": float(mean_accuracy),
		"mean_iou": float(mean_iou),
		"weighted_iou": float(weighted_iou),
		"per_class_accuracy": class_acc,
		"per_class_iou": class_iou,
		"gt_pixels": gt_pixels,
	}


def segmentation_metrics(
	pred_labels: Union[np.ndarray, torch.Tensor],
	gt_labels: Union[np.ndarray, torch.Tensor],
	num_class: int,
	ignore_index: int = -1,
) -> Dict[str, float]:
	"""Convenience wrapper for the four ADE20K metrics from the paper."""
	hist = confusion_matrix(pred_labels, gt_labels, num_class=num_class, ignore_index=ignore_index)
	return segmentation_metrics_from_confusion(hist)
