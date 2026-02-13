"""Tests for config_presets.py — static configuration presets."""

from __future__ import annotations

from neural_memory.config_presets import (
    BALANCED,
    MAX_RECALL,
    SAFE_COST,
    apply_preset,
    compute_diff,
    get_preset,
    list_presets,
)
from neural_memory.unified_config import (
    BrainSettings,
    MaintenanceConfig,
    UnifiedConfig,
)


class TestListPresets:
    def test_returns_three_presets(self) -> None:
        presets = list_presets()
        assert len(presets) == 3

    def test_each_has_name_and_description(self) -> None:
        for p in list_presets():
            assert "name" in p
            assert "description" in p
            assert len(p["description"]) > 0

    def test_names_are_expected(self) -> None:
        names = [p["name"] for p in list_presets()]
        assert "safe-cost" in names
        assert "balanced" in names
        assert "max-recall" in names


class TestGetPreset:
    def test_returns_preset_by_name(self) -> None:
        assert get_preset("safe-cost") is SAFE_COST
        assert get_preset("balanced") is BALANCED
        assert get_preset("max-recall") is MAX_RECALL

    def test_returns_none_for_unknown(self) -> None:
        assert get_preset("nonexistent") is None
        assert get_preset("") is None


class TestComputeDiff:
    def test_no_diff_when_matching(self, tmp_path: object) -> None:
        """Balanced preset matches default config — no changes."""
        config = UnifiedConfig()
        changes = compute_diff(config, BALANCED)
        assert changes == []

    def test_detects_brain_changes(self) -> None:
        config = UnifiedConfig()
        changes = compute_diff(config, SAFE_COST)
        brain_changes = [c for c in changes if c["section"] == "brain"]
        assert len(brain_changes) > 0
        keys_changed = {c["key"] for c in brain_changes}
        assert "decay_rate" in keys_changed

    def test_detects_maintenance_changes(self) -> None:
        config = UnifiedConfig()
        changes = compute_diff(config, MAX_RECALL)
        maint_changes = [c for c in changes if c["section"] == "maintenance"]
        assert len(maint_changes) > 0

    def test_includes_current_and_new_values(self) -> None:
        config = UnifiedConfig()
        changes = compute_diff(config, SAFE_COST)
        for change in changes:
            assert "current" in change
            assert "new" in change
            assert "section" in change
            assert "key" in change


class TestApplyPreset:
    def test_applies_safe_cost(self) -> None:
        config = UnifiedConfig()
        changes = apply_preset(config, SAFE_COST)
        assert len(changes) > 0
        assert config.brain.decay_rate == 0.2
        assert config.brain.max_spread_hops == 2
        assert config.brain.max_context_tokens == 500

    def test_applies_max_recall(self) -> None:
        config = UnifiedConfig()
        apply_preset(config, MAX_RECALL)
        assert config.brain.decay_rate == 0.05
        assert config.brain.max_spread_hops == 6
        assert config.brain.max_context_tokens == 3000

    def test_balanced_is_noop(self) -> None:
        config = UnifiedConfig()
        changes = apply_preset(config, BALANCED)
        assert changes == []

    def test_preserves_unrelated_config(self) -> None:
        config = UnifiedConfig()
        config.current_brain = "test-brain"
        apply_preset(config, SAFE_COST)
        assert config.current_brain == "test-brain"

    def test_returns_change_list(self) -> None:
        config = UnifiedConfig()
        changes = apply_preset(config, SAFE_COST)
        assert isinstance(changes, list)
        for change in changes:
            assert "section" in change
            assert "key" in change

    def test_does_not_mutate_preset(self) -> None:
        import copy

        original = copy.deepcopy(SAFE_COST)
        config = UnifiedConfig()
        apply_preset(config, SAFE_COST)
        assert original == SAFE_COST


class TestPresetStructure:
    """Validate preset dicts have correct structure."""

    def test_all_presets_have_brain_section(self) -> None:
        for name in ("safe-cost", "balanced", "max-recall"):
            preset = get_preset(name)
            assert preset is not None
            assert "brain" in preset

    def test_brain_keys_are_valid(self) -> None:
        valid_keys = set(BrainSettings().to_dict().keys())
        for name in ("safe-cost", "balanced", "max-recall"):
            preset = get_preset(name)
            assert preset is not None
            for key in preset.get("brain", {}):
                assert key in valid_keys, f"Invalid brain key '{key}' in preset '{name}'"

    def test_maintenance_keys_are_valid(self) -> None:
        valid_keys = set(MaintenanceConfig().to_dict().keys())
        for name in ("safe-cost", "balanced", "max-recall"):
            preset = get_preset(name)
            assert preset is not None
            for key in preset.get("maintenance", {}):
                assert key in valid_keys, f"Invalid maintenance key '{key}' in preset '{name}'"
