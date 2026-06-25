"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, RayCastSensor

from . import height_scan_occlusion as mdp_occlusion

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def base_lin_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


##
# Joint state.
##


def joint_pos_rel(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


##
# Actions.
##


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command


##
# Sensors.
##


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Get observation from a built-in sensor by name."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  offset: float = 0.0,
  miss_value: float | None = None,
) -> torch.Tensor:
  """Height scan from a raycast sensor.

  Returns the height of the sensor frame above each hit point.

  Args:
    env: The environment.
    sensor_name: Name of a RayCastSensor in the scene.
    offset: Constant offset subtracted from heights.
    miss_value: Value to use for rays that miss (distance < 0).
      Defaults to the sensor's ``max_distance``.

  Returns:
    Tensor of shape [B, N] where B is num_envs and N is num_rays.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  if miss_value is None:
    miss_value = sensor.cfg.max_distance
  heights = (
    sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.hit_pos_w[..., 2] - offset
  )
  miss_mask = sensor.data.distances < 0
  return torch.where(miss_mask, torch.full_like(heights, miss_value), heights)


def height_scan_occluded(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  lidar_offset: tuple[float, float, float] = (0.0, 0.0, 0.25),
  offset: float = 0.0,
  miss_value: float | None = None,
  num_samples: int = 24,
  eps_deg: float = 0.5,
  return_mask: bool = True,
) -> torch.Tensor:
  """Occlusion-aware ("realistic") height scan from a grid raycast sensor.

  Unlike :func:`height_scan`, which always reports the true top surface, this marks
  cells the robot could not actually perceive from a virtual LiDAR (line of sight
  blocked by nearer-taller terrain, or no return) as invalid. See
  :mod:`mjlab.envs.mdp.height_scan_occlusion` for the 2.5D horizon test.

  The returned heights are **normalized** by the sensor's ``max_distance`` (occluded /
  miss cells set to that max), so unlike :func:`height_scan` you do NOT apply a cfg
  ``scale``. With ``return_mask=True`` the per-cell validity mask is appended, giving
  a flat ``[B, 2N]`` tensor (``[normalized_height | mask]``) that a downstream encoder
  can reshape to a 2-channel grid.

  Args:
    env: The environment.
    sensor_name: Name of a grid :class:`RayCastSensor` in the scene.
    lidar_offset: Virtual LiDAR position relative to the sensor frame (metres),
      in the sensor's yaw-aligned frame (x forward, y left, z up).
    offset: Constant offset subtracted from heights before normalization.
    miss_value: Height used for invalid (occluded / miss) cells. Defaults to the
      sensor's ``max_distance``.
    num_samples: Ray-march samples for the line-of-sight test.
    eps_deg: Angular tolerance (degrees) for the horizon comparison.
    return_mask: If True, append the ``[B, N]`` validity mask, returning ``[B, 2N]``.

  Returns:
    Tensor of shape ``[B, 2N]`` (default) or ``[B, N]`` if ``return_mask`` is False.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  max_distance = sensor.cfg.max_distance
  if miss_value is None:
    miss_value = max_distance

  heights = (
    sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.hit_pos_w[..., 2] - offset
  )
  miss_mask = sensor.data.distances < 0
  # Fill misses with the "deep" miss value so holes don't cast line-of-sight shadows.
  heights = torch.where(miss_mask, torch.full_like(heights, miss_value), heights)

  ray_xy = sensor.local_offsets[:, :2]
  lx, ly, lz = lidar_offset
  visible = mdp_occlusion.compute_visibility_mask(
    heights, ray_xy, (lx, ly), lz, num_samples=num_samples, eps_deg=eps_deg
  )
  invalid = (visible < 0.5) | miss_mask

  heights = torch.where(invalid, torch.full_like(heights, miss_value), heights)
  normalized = heights / max_distance
  if not return_mask:
    return normalized
  mask = (~invalid).to(normalized.dtype)
  return torch.cat([normalized, mask], dim=-1)
