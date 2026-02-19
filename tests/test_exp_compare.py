"""Tests for exp_compare module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from exp_compare.io import load_run_summaries, compute_intersection
from exp_compare.metrics import compute_rpdf, compute_rpdv, compute_rank
from exp_compare.main import (
    CompareConfig,
    build_wide_rpdf,
    build_wide_rpdv,
    build_summary_rpdf,
    build_summary_rpdv,
)


class TestRPDMetrics:
    """Tests for RPDf and RPDv computation."""

    def test_rpdf_basic(self):
        """Test basic RPDF computation."""
        # obj=100, ref=80 -> (100-80)/((100+80)/2) = 20/90 = 0.222...
        result = compute_rpdf(100.0, 80.0)
        expected = 20.0 / 90.0
        assert abs(result - expected) < 1e-10

    def test_rpdf_zero_zero(self):
        """Test RPDF when both obj and ref are zero."""
        result = compute_rpdf(0.0, 0.0)
        assert result == 0.0

    def test_rpdf_denominator_zero(self):
        """Test RPDF when denominator is zero (obj=-ref but not both zero)."""
        result = compute_rpdf(10.0, -10.0)
        assert pd.isna(result)

    def test_rpdf_array(self):
        """Test RPDF with array input."""
        objs = pd.Series([100.0, 80.0, 0.0])
        refs = pd.Series([80.0, 80.0, 0.0])
        results = compute_rpdf(objs, refs)
        assert len(results) == 3
        assert abs(results[0] - (20.0 / 90.0)) < 1e-10
        assert results[1] == 0.0  # obj == ref
        assert results[2] == 0.0  # both zero

    def test_rpdv_basic(self):
        """Test basic RPDV computation."""
        # obj=100, ref=80 -> (100-80)/80 = 0.25
        result = compute_rpdv(100.0, 80.0)
        expected = 0.25
        assert abs(result - expected) < 1e-10

    def test_rpdv_zero_ref(self):
        """Test RPDV when ref is zero."""
        result = compute_rpdv(10.0, 0.0)
        assert pd.isna(result)

    def test_rank_basic(self):
        """Test rank computation."""
        values = pd.Series([100.0, 80.0, 90.0, 80.0, 120.0])
        ranks = compute_rank(values)
        # Dense rank: 80->1, 80->1, 90->2, 100->3, 120->4
        expected = [3.0, 1.0, 2.0, 1.0, 4.0]
        assert list(ranks) == expected

    def test_rank_with_nan(self):
        """Test rank computation with NaN values."""
        values = pd.Series([100.0, float("nan"), 80.0])
        ranks = compute_rank(values)
        assert ranks[0] == 2.0
        assert pd.isna(ranks[1])
        assert ranks[2] == 1.0


class TestWideFormat:
    """Tests for wide format DataFrame generation."""

    def test_build_wide_rpdf(self):
        """Test wide format with RPDf values."""
        df = pd.DataFrame({
            "name": ["inst1", "inst1", "inst2", "inst2"],
            "algoUid": ["run1::s1", "run2::s1", "run1::s1", "run2::s1"],
            "RPDf": [0.1, 0.2, 0.15, 0.25],
        })

        wide = build_wide_rpdf(df)

        assert "name" in wide.columns
        assert "run1::s1" in wide.columns
        assert "run2::s1" in wide.columns
        assert len(wide) == 2  # 2 instances
        assert wide[wide["name"] == "inst1"]["run1::s1"].values[0] == 0.1
        assert wide[wide["name"] == "inst1"]["run2::s1"].values[0] == 0.2

    def test_build_wide_rpdv(self):
        """Test wide format with RPDv values."""
        df = pd.DataFrame({
            "name": ["inst1", "inst1", "inst2", "inst2"],
            "algoUid": ["run1::s1", "run2::s1", "run1::s1", "run2::s1"],
            "RPDv": [0.15, 0.25, 0.18, 0.28],
        })

        wide = build_wide_rpdv(df)

        assert "name" in wide.columns
        assert "run1::s1" in wide.columns
        assert "run2::s1" in wide.columns
        assert len(wide) == 2
        assert wide[wide["name"] == "inst1"]["run1::s1"].values[0] == 0.15

    def test_build_summary_rpdf(self):
        """Test summary statistics computation."""
        wide = pd.DataFrame({
            "name": ["inst1", "inst2"],
            "run1::s1": [0.1, 0.2],
            "run2::s1": [0.15, 0.25],
        })

        summary = build_summary_rpdf(wide)

        assert len(summary) == 2
        assert "algoUid" in summary.columns
        assert "count" in summary.columns
        assert "mean" in summary.columns
        assert "std" in summary.columns


class TestIOIntegration:
    """Integration tests for IO functions."""

    def test_compute_intersection_basic(self):
        """Test basic intersection computation."""
        names = {
            "run1": {"inst1", "inst2", "inst3"},
            "run2": {"inst2", "inst3", "inst4"},
            "run3": {"inst2", "inst3"},
        }
        intersection = compute_intersection(names)
        assert intersection == {"inst2", "inst3"}

    def test_compute_intersection_empty(self):
        """Test empty intersection."""
        names = {
            "run1": {"inst1"},
            "run2": {"inst2"},
        }
        intersection = compute_intersection(names)
        assert intersection == set()

    def test_compute_intersection_single_run(self):
        """Test intersection with single run."""
        names = {
            "run1": {"inst1", "inst2"},
        }
        intersection = compute_intersection(names)
        assert intersection == {"inst1", "inst2"}


def create_test_summary_csv(
    tmp_path: Path, run_id: str, instances: list[str], scenarios: list[str]
) -> tuple[Path, pd.DataFrame]:
    """Create a test summary CSV file."""
    rows = []
    for inst in instances:
        for scenario in scenarios:
            rows.append({
                "name": inst,
                "scenario": scenario,
                "bestObj": 100.0 + hash(inst + scenario) % 50,
            })

    df = pd.DataFrame(rows)
    csv_path = tmp_path / f"{run_id}_summary.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


class TestEndToEnd:
    """End-to-end tests for the comparison pipeline."""

    def test_full_pipeline(self):
        """Test full comparison pipeline with temporary files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test data
            instances = ["inst1", "inst2", "inst3"]
            scenarios = ["scenario_1"]

            # Create two run directories
            run1_path = tmp_path / "run1"
            run2_path = tmp_path / "run2"
            run1_path.mkdir()
            run2_path.mkdir()

            _, df1 = create_test_summary_csv(run1_path, "run1", instances, scenarios)
            _, df2 = create_test_summary_csv(run2_path, "run2", instances, scenarios)

            # Create config
            config = CompareConfig(
                runs=[
                    {"path": str(run1_path), "summary_csv": "run1_summary.csv"},
                    {"path": str(run2_path), "summary_csv": "run2_summary.csv"},
                ],
                reference={
                    "mode": "best_among_compared",
                    "sense": "min",
                },
                output={
                    "out_dir": str(tmp_path / "output"),
                    "basename": "test",
                },
            )

            # Import and run
            from exp_compare.main import run_comparison

            exit_code = run_comparison(config)

            assert exit_code == 0

            # Verify output files
            output_dir = tmp_path / "output"
            assert (output_dir / "test_long.csv").exists()
            assert (output_dir / "test_rpdf_wide.csv").exists()
            assert (output_dir / "test_rpdv_wide.csv").exists()
            assert (output_dir / "test_summary_rpdf.csv").exists()
            assert (output_dir / "test_summary_rpdv.csv").exists()

            # Verify long format
            long_df = pd.read_csv(output_dir / "test_long.csv")
            assert "name" in long_df.columns
            assert "algoUid" in long_df.columns
            assert "RPDf" in long_df.columns
            assert "rank" in long_df.columns

            # Verify wide format
            wide_df = pd.read_csv(output_dir / "test_rpdf_wide.csv")
            assert "name" in wide_df.columns
            assert len(wide_df) == len(instances)
