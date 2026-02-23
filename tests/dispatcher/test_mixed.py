"""Tests for MixedDispatcher class."""

import inspect

from hybridflowshop.dispatcher import MixedDispatcher


class TestMixedDispatcher:
    """Tests for MixedDispatcher class."""

    def test_get_schedule_by_cds_has_in_place(self) -> None:
        """Test mixed CDS has in_place parameter."""
        dispatcher = MixedDispatcher.__new__(MixedDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_cds)
        assert "in_place" in sig.parameters

    def test_get_schedule_by_gupta_has_in_place(self) -> None:
        """Test mixed Gupta has in_place parameter."""
        dispatcher = MixedDispatcher.__new__(MixedDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_gupta)
        assert "in_place" in sig.parameters

    def test_get_schedule_by_palmer_has_in_place(self) -> None:
        """Test mixed Palmer has in_place parameter."""
        dispatcher = MixedDispatcher.__new__(MixedDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_palmer)
        assert "in_place" in sig.parameters

    def test_get_schedule_by_cds_has_all_common_params(self) -> None:
        """Test mixed CDS has all common parameters."""
        dispatcher = MixedDispatcher.__new__(MixedDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_cds)
        params = sig.parameters
        assert "schedule" in params
        assert "in_place" in params
        assert "from_stage" in params
        assert "job_2_release_t" in params

    def test_get_schedule_by_gupta_has_all_common_params(self) -> None:
        """Test mixed Gupta has all common parameters."""
        dispatcher = MixedDispatcher.__new__(MixedDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_gupta)
        params = sig.parameters
        assert "schedule" in params
        assert "in_place" in params
        assert "from_stage" in params
        assert "job_2_release_t" in params

    def test_get_schedule_by_palmer_has_all_common_params(self) -> None:
        """Test mixed Palmer has all common parameters."""
        dispatcher = MixedDispatcher.__new__(MixedDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_palmer)
        params = sig.parameters
        assert "schedule" in params
        assert "in_place" in params
        assert "from_stage" in params
        assert "job_2_release_t" in params

    def test_get_np_candidates(self) -> None:
        """Test NP candidate generation."""
        dispatcher = MixedDispatcher.__new__(MixedDispatcher)
        dispatcher.job_count = 8  # Must be set since __init__ isn't called
        np_list = dispatcher._get_np_candidates()
        # job_count=8, so candidates should be [8, 4, 2, 1, 0]
        # (ceiling of 8/2 = 4, ceiling of 4/2 = 2, ceiling of 2/2 = 1, then 0)
        assert np_list == [8, 4, 2, 1, 0]
