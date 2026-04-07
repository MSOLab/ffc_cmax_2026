from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    description: str
    mip_args: tuple[str, ...]


PRESETS: tuple[ExperimentPreset, ...] = (
    ExperimentPreset(
        name="binary_auto",
        description=(
            "All a/b binary, delta set to p_max per instance, full strengthening."
        ),
        mip_args=(),
    ),
    ExperimentPreset(
        name="binary_pmax1",
        description=(
            "All a/b binary, delta set to p_max + 1 per instance, full strengthening."
        ),
        mip_args=("--delta-pmax-plus-one",),
    ),
    ExperimentPreset(
        name="base_binary_pmax1",
        description=(
            "All a/b binary, delta p_max + 1, base model only without extra valid inequalities."
        ),
        mip_args=("--delta-pmax-plus-one", "--base-model-only"),
    ),
)


def list_presets() -> tuple[ExperimentPreset, ...]:
    return PRESETS


def get_preset(name: str) -> ExperimentPreset:
    alias_map = {
        "continuous_auto": "binary_auto",
        "continuous_pmax1": "binary_pmax1",
        "base_continuous_pmax1": "base_binary_pmax1",
        "partial_binary_10_pmax1": "binary_pmax1",
        "partial_binary_20_pmax1": "binary_pmax1",
    }
    normalized_name = alias_map.get(name, name)
    for preset in PRESETS:
        if preset.name == normalized_name:
            return preset
    available = ", ".join(preset.name for preset in PRESETS)
    raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")
