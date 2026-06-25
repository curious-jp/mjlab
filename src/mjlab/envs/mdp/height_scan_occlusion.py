"""2.5D line-of-sight occlusion for height-scan sensors.

A ``RayCastSensor`` grid casts rays straight down, so its height scan always reports
the true top surface -- even terrain a real robot could never see. On hardware, terrain
is perceived from a LiDAR mounted above the base, so ground hidden behind steps, walls,
and ledges is *occluded* and unknown.

This module approximates that with a cheap 2.5D line-of-sight (horizon) test over the
scan grid, from a virtual LiDAR viewpoint: a cell is occluded when nearer-taller terrain
rises above the line of sight to it. It is pure torch and operates on the sensor's real
ray geometry and metric hit heights, so no resolution/scale guessing is required.

The companion observation term ``height_scan_occluded`` (in ``observations.py``) wires
this to a sensor in the scene.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_visibility_mask(
  hit_heights: torch.Tensor,
  ray_xy: torch.Tensor,
  lidar_xy: tuple[float, float],
  lidar_z: float,
  *,
  num_samples: int = 24,
  eps_deg: float = 0.5,
) -> torch.Tensor:
  """1.0 where a grid cell is visible from the virtual LiDAR, 0.0 where occluded.

  Args:
    hit_heights: ``[B, N]`` metric height of the sensor frame above each hit point
      (``pos_w.z - hit_pos_w.z``); larger == lower terrain. Misses should already be
      filled with a large "deep" value so holes do not cast shadows.
    ray_xy: ``[N, 2]`` per-ray ``(x, y)`` offsets in the sensor's (yaw-aligned) frame,
      in metres. Must form a regular grid (e.g. ``GridPatternCfg``).
    lidar_xy: Virtual LiDAR ``(x, y)`` in the same frame (metres).
    lidar_z: Virtual LiDAR height above the sensor frame origin (metres).
    num_samples: Ray-march samples from the LiDAR to each target cell.
    eps_deg: Angular tolerance (degrees) for the horizon comparison.

  Returns:
    Float mask of shape ``[B, N]`` (1.0 visible, 0.0 occluded).
  """
  if hit_heights.shape[-1] != ray_xy.shape[0]:
    raise ValueError(
      f"hit_heights last dim {hit_heights.shape[-1]} != n_rays {ray_xy.shape[0]}."
    )
  b, n = hit_heights.shape
  device = hit_heights.device
  ray_xy = ray_xy.to(device=device, dtype=hit_heights.dtype)

  # Infer the regular grid from the ray offsets (order-independent).
  xs = torch.unique(ray_xy[:, 0])
  ys = torch.unique(ray_xy[:, 1])
  nx, ny = int(xs.numel()), int(ys.numel())
  if nx * ny != n:
    raise ValueError(f"ray_xy does not form a regular grid: {nx}x{ny} != {n} rays.")
  x0, x1 = float(xs[0]), float(xs[-1])
  y0, y1 = float(ys[0]), float(ys[-1])
  res_x = (x1 - x0) / max(nx - 1, 1)
  res_y = (y1 - y0) / max(ny - 1, 1)

  # Scatter heights into a [B, 1, Ny, Nx] field for bilinear sampling.
  ix = torch.round((ray_xy[:, 0] - x0) / max(res_x, 1e-9)).long().clamp(0, nx - 1)
  iy = torch.round((ray_xy[:, 1] - y0) / max(res_y, 1e-9)).long().clamp(0, ny - 1)
  flat_idx = iy * nx + ix  # [N]
  field = hit_heights.new_zeros((b, ny * nx))
  field[:, flat_idx] = hit_heights
  field = field.view(b, 1, ny, nx)

  # March samples from the LiDAR (xy) to each target cell: [N, K, 2].
  k = max(2, num_samples)
  lidar = torch.tensor(lidar_xy, device=device, dtype=hit_heights.dtype).view(1, 1, 2)
  frac = torch.linspace(0.0, 1.0, k, device=device, dtype=hit_heights.dtype)
  sample_xy = lidar + frac.view(1, k, 1) * (ray_xy.view(n, 1, 2) - lidar)
  dist = torch.linalg.norm(sample_xy - lidar, dim=-1).clamp_min(1e-6)  # [N, K]

  # Normalize sample xy to grid_sample coords in [-1, 1] (align_corners).
  cx, hx = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
  cy, hy = 0.5 * (y0 + y1), 0.5 * (y1 - y0)
  gx = (sample_xy[..., 0] - cx) / max(hx, 1e-6)
  gy = (sample_xy[..., 1] - cy) / max(hy, 1e-6)
  grid = torch.stack([gx, gy], dim=-1).view(1, n, k, 2).expand(b, n, k, 2)
  h_sample = F.grid_sample(
    field, grid, mode="bilinear", padding_mode="border", align_corners=True
  ).view(b, n, k)

  # Depression angle (looking down) at each sample: larger == steeper down. A cell is
  # occluded iff a strictly-nearer sample has a smaller depression angle (nearer-taller
  # terrain shadows it). The target cell is the last sample (frac == 1).
  depression = torch.atan2(lidar_z + h_sample, dist.view(1, n, k))  # [B, N, K]
  eps = math.radians(eps_deg)
  min_nearer = depression[..., : k - 1].min(dim=-1).values  # [B, N]
  target = depression[..., -1]  # [B, N]
  return (target <= min_nearer + eps).to(hit_heights.dtype)
