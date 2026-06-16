"""Tests for terrain viewer source discovery helpers."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from mjlab.scripts import visualize_terrain
from mjlab.tasks import registry
from mjlab.terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg


def _package(name: str) -> types.ModuleType:
  module = types.ModuleType(name)
  module.__path__ = []
  return module


def _flat_generator(seed: int | None = None) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    seed=seed,
    size=(4.0, 3.0),
    num_rows=10,
    num_cols=1,
    sub_terrains={"flat": BoxFlatTerrainCfg(proportion=1.0)},
  )


def test_discover_evaluation_terrain_options_uses_optional_provider(monkeypatch):
  terrains_module = types.ModuleType("src.tasks.velocity.evaluation.terrains")
  terrains_module.make_rough_curriculum_corridor_cfg = _flat_generator

  for name in (
    "src",
    "src.tasks",
    "src.tasks.velocity",
    "src.tasks.velocity.evaluation",
  ):
    monkeypatch.setitem(sys.modules, name, _package(name))
  monkeypatch.setitem(
    sys.modules, "src.tasks.velocity.evaluation.terrains", terrains_module
  )

  options = visualize_terrain.discover_evaluation_terrain_options()

  assert list(options) == ["rough_curriculum_corridor"]
  option = options["rough_curriculum_corridor"]
  assert option.source == visualize_terrain.TERRAIN_SOURCE_EVALUATION
  cfg = option.build_generator(123)
  assert cfg.seed == 123
  assert cfg.num_rows == 10
  assert cfg.num_cols == 1
  assert cfg.size == (4.0, 3.0)


def test_discover_environment_terrain_options_filters_generator_tasks(monkeypatch):
  generator_task = "Test-Viewer-Generator-Env"
  flat_task = "Test-Viewer-Flat-Env"
  gen_cfg = _flat_generator(seed=5)
  gen_env = SimpleNamespace(
    scene=SimpleNamespace(
      terrain=SimpleNamespace(terrain_type="generator", terrain_generator=gen_cfg)
    )
  )
  flat_env = SimpleNamespace(
    scene=SimpleNamespace(
      terrain=SimpleNamespace(terrain_type="plane", terrain_generator=None)
    )
  )

  monkeypatch.setitem(
    registry._REGISTRY,
    generator_task,
    registry._TaskCfg(gen_env, gen_env, None, None),
  )
  monkeypatch.setitem(
    registry._REGISTRY,
    flat_task,
    registry._TaskCfg(flat_env, flat_env, None, None),
  )

  options = visualize_terrain.discover_environment_terrain_options(
    import_package_names=()
  )

  assert generator_task in options
  assert flat_task not in options
  cfg = options[generator_task].build_generator(77)
  assert cfg.seed == 77
  assert cfg is not gen_cfg
  assert gen_cfg.seed == 5
