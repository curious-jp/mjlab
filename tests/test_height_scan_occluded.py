"""Tests for the 2.5D height-scan occlusion math and observation term."""

from unittest.mock import MagicMock

import torch

from mjlab.envs.mdp.height_scan_occlusion import compute_visibility_mask
from mjlab.envs.mdp.observations import height_scan_occluded

NX, NY = 17, 11
SIZE = (1.6, 1.0)
RES = 0.1


def _grid_ray_xy() -> torch.Tensor:
  """Per-ray (x, y), matching GridPatternCfg flatten order (meshgrid 'xy')."""
  sx, sy = SIZE
  x = torch.arange(-sx / 2, sx / 2 + RES * 0.5, RES)[:NX]
  y = torch.arange(-sy / 2, sy / 2 + RES * 0.5, RES)[:NY]
  gx, gy = torch.meshgrid(x, y, indexing="xy")  # [NY, NX]
  return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # [N, 2]


def test_flat_terrain_all_visible() -> None:
  ray_xy = _grid_ray_xy()
  heights = torch.full((1, NX * NY), 0.3)  # sensor 0.3 m above flat ground
  mask = compute_visibility_mask(heights, ray_xy, (0.0, 0.0), 0.25)
  assert mask.shape == (1, NX * NY)
  assert torch.all(mask > 0.5)


def test_tall_wall_occludes_behind() -> None:
  ray_xy = _grid_ray_xy()
  x = ray_xy[:, 0]
  heights = torch.full((NX * NY,), 0.3)
  # Wall top 0.5 m ABOVE the sensor origin (well above the 0.25 m LiDAR): metric
  # height = sensor_z - hit_z = -0.5.
  heights[(x > 0.25) & (x < 0.35)] = -0.5
  mask = compute_visibility_mask(heights.view(1, -1), ray_xy, (0.0, 0.0), 0.25).view(-1)

  far = x > 0.45
  near = (x > 0.0) & (x < 0.2)
  assert torch.any(far)
  assert mask[far].mean() < 0.5  # mostly shadowed
  assert torch.all(mask[near] > 0.5)  # in front of the wall stays visible


def test_order_independent_grid() -> None:
  # Shuffled ray order must give the same per-ray result (scatter-based field).
  ray_xy = _grid_ray_xy()
  x = ray_xy[:, 0]
  heights = torch.full((NX * NY,), 0.3)
  heights[(x > 0.25) & (x < 0.35)] = -0.5
  perm = torch.randperm(NX * NY)
  m0 = compute_visibility_mask(heights.view(1, -1), ray_xy, (0.0, 0.0), 0.25).view(-1)
  m1 = compute_visibility_mask(
    heights[perm].view(1, -1), ray_xy[perm], (0.0, 0.0), 0.25
  ).view(-1)
  assert torch.equal(m0[perm], m1)


def _mock_sensor(heights: torch.Tensor, ray_xy: torch.Tensor, max_distance=5.0):
  """A RayCastSensor stand-in: pos_w - hit_pos_w.z == heights, no misses."""
  b, n = heights.shape
  sensor = MagicMock()
  sensor.cfg.max_distance = max_distance
  sensor.data.pos_w = torch.zeros(b, 3)
  hit = torch.zeros(b, n, 3)
  hit[..., 2] = -heights  # pos_z(0) - hit_z == heights
  sensor.data.hit_pos_w = hit
  sensor.data.distances = torch.ones(b, n)  # all hits
  offsets = torch.zeros(n, 3)
  offsets[:, :2] = ray_xy
  sensor.local_offsets = offsets
  return sensor


def test_observation_returns_height_and_mask() -> None:
  ray_xy = _grid_ray_xy()
  x = ray_xy[:, 0]
  heights = torch.full((1, NX * NY), 0.3)
  heights[0, (x > 0.25) & (x < 0.35)] = -0.5
  sensor = _mock_sensor(heights, ray_xy)
  env = MagicMock()
  env.scene = {"terrain_scan": sensor}

  out = height_scan_occluded(env, "terrain_scan", lidar_offset=(0.0, 0.0, 0.25))
  assert out.shape == (1, 2 * NX * NY)  # [normalized_height | mask]
  norm, mask = out[:, : NX * NY], out[:, NX * NY :]
  # Occluded cells are filled with max_distance (normalized 1.0) and flagged invalid.
  occluded = mask[0] < 0.5
  assert torch.any(occluded)
  assert torch.allclose(norm[0, occluded], torch.ones(int(occluded.sum())))

  out_no_mask = height_scan_occluded(
    env, "terrain_scan", lidar_offset=(0.0, 0.0, 0.25), return_mask=False
  )
  assert out_no_mask.shape == (1, NX * NY)
