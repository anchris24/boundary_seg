from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _neighbor_disagreement(labels: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
	"""Return 1 where labels disagree with at least one 4-neighbor.

	Args:
		labels: Tensor of shape (N, H, W).
		ignore_index: Label id to ignore.
	"""
	if labels.ndim != 3:
		raise ValueError("Expected labels with shape (N, H, W)")

	n, h, w = labels.shape
	center = labels

	# Pad with ignore labels so padded comparisons do not create false edges.
	pad = F.pad(center.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=float(ignore_index)).squeeze(1).long()

	up = pad[:, 0:h, 1:w + 1]
	down = pad[:, 2:h + 2, 1:w + 1]
	left = pad[:, 1:h + 1, 0:w]
	right = pad[:, 1:h + 1, 2:w + 2]

	valid_center = center != ignore_index

	disagreement = torch.zeros_like(center, dtype=torch.bool)
	for nb in (up, down, left, right):
		valid_nb = nb != ignore_index
		disagreement |= valid_center & valid_nb & (nb != center)
	return disagreement


def boundary_mask_from_labels(
	labels: torch.Tensor,
	boundary_width: int = 1,
	ignore_index: int = -1,
) -> torch.Tensor:
	"""Create boundary mask from semantic labels.

	Returns:
		Bool tensor of shape (N, H, W).
	"""
	edge = _neighbor_disagreement(labels, ignore_index=ignore_index)
	if boundary_width <= 1:
		return edge

	edge_float = edge.float().unsqueeze(1)
	dilated = F.max_pool2d(edge_float, kernel_size=2 * boundary_width + 1, stride=1, padding=boundary_width)
	return dilated.squeeze(1) > 0


def boundary_aware_nll_loss(
	log_probs: torch.Tensor,
	target: torch.Tensor,
	lambda_boundary: float = 1.0,
	boundary_width: int = 1,
	ignore_index: int = -1,
) -> torch.Tensor:
	"""Per-pixel NLL with higher weight on boundary pixels.

	This is a weighted variant of NLL where non-boundary pixels get weight 1,
	and boundary pixels get weight (1 + lambda_boundary).
	"""
	per_pixel = F.nll_loss(log_probs, target, ignore_index=ignore_index, reduction="none")
	valid = target != ignore_index
	boundary = boundary_mask_from_labels(target, boundary_width=boundary_width, ignore_index=ignore_index)

	weights = torch.ones_like(per_pixel)
	if lambda_boundary > 0:
		weights = weights + lambda_boundary * boundary.float()

	weighted = per_pixel * weights * valid.float()
	denom = (weights * valid.float()).sum().clamp_min(1e-6)
	return weighted.sum() / denom


class BoundaryAwareNLLLoss(nn.Module):
	"""nn.Module wrapper around `boundary_aware_nll_loss`."""

	def __init__(
		self,
		lambda_boundary: float = 1.0,
		boundary_width: int = 1,
		ignore_index: int = -1,
	):
		super().__init__()
		self.lambda_boundary = lambda_boundary
		self.boundary_width = boundary_width
		self.ignore_index = ignore_index

	def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		return boundary_aware_nll_loss(
			log_probs=log_probs,
			target=target,
			lambda_boundary=self.lambda_boundary,
			boundary_width=self.boundary_width,
			ignore_index=self.ignore_index,
		)


def combined_segmentation_loss(
	log_probs: torch.Tensor,
	target: torch.Tensor,
	lambda_boundary: float,
	boundary_width: int,
	ignore_index: int = -1,
	reduction: str = "mean",
) -> torch.Tensor:
	"""Convenience helper to switch between baseline and boundary-aware loss."""
	if lambda_boundary <= 0:
		return F.nll_loss(log_probs, target, ignore_index=ignore_index, reduction=reduction)
	return boundary_aware_nll_loss(
		log_probs=log_probs,
		target=target,
		lambda_boundary=lambda_boundary,
		boundary_width=boundary_width,
		ignore_index=ignore_index,
	)


def boundary_aware_ce_loss(
	logits: torch.Tensor,
	target: torch.Tensor,
	lambda_boundary: float = 1.0,
	boundary_width: int = 1,
	ignore_index: int = -1,
) -> torch.Tensor:
	"""Boundary-aware cross entropy for raw logits."""
	per_pixel = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
	valid = target != ignore_index
	boundary = boundary_mask_from_labels(target, boundary_width=boundary_width, ignore_index=ignore_index)

	weights = torch.ones_like(per_pixel)
	if lambda_boundary > 0:
		weights = weights + lambda_boundary * boundary.float()

	weighted = per_pixel * weights * valid.float()
	denom = (weights * valid.float()).sum().clamp_min(1e-6)
	return weighted.sum() / denom
