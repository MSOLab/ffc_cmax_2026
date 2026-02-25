"""Tests for StageDispatcher class."""

import inspect

from hybridflowshop.dispatcher import StageDispatcher


class TestStageDispatcher:
    """Tests for StageDispatcher class."""

    def test_get_schedule_by_ds_cds_has_all_common_params(self) -> None:
        """Test DS-CDS has all common parameters."""
        dispatcher = StageDispatcher.__new__(StageDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_ds_cds)
        params = sig.parameters
        assert "schedule" in params
        assert "from_stage" in params
        assert "job_2_release_t" in params

    def test_get_schedule_by_ds_gupta_has_all_common_params(self) -> None:
        """Test DS-Gupta has all common parameters."""
        dispatcher = StageDispatcher.__new__(StageDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_ds_gupta)
        params = sig.parameters
        assert "schedule" in params
        assert "from_stage" in params
        assert "job_2_release_t" in params

    def test_get_schedule_by_ds_palmer_has_all_common_params(self) -> None:
        """Test DS-Palmer has all common parameters."""
        dispatcher = StageDispatcher.__new__(StageDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_ds_palmer)
        params = sig.parameters
        assert "schedule" in params
        assert "from_stage" in params
        assert "job_2_release_t" in params
