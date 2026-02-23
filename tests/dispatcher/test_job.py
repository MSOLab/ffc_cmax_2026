"""Tests for JobDispatcher class."""

import inspect

from hybridflowshop.dispatcher import JobDispatcher


class TestJobDispatcher:
    """Tests for JobDispatcher class."""

    def test_get_schedule_by_dj_cds_has_in_place(self) -> None:
        """Test DJ-CDS has in_place parameter."""
        dispatcher = JobDispatcher.__new__(JobDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_dj_cds)
        assert "in_place" in sig.parameters

    def test_get_schedule_by_dj_gupta_has_in_place(self) -> None:
        """Test DJ-Gupta has in_place parameter."""
        dispatcher = JobDispatcher.__new__(JobDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_dj_gupta)
        assert "in_place" in sig.parameters

    def test_get_schedule_by_dj_palmer_has_in_place(self) -> None:
        """Test DJ-Palmer has in_place parameter."""
        dispatcher = JobDispatcher.__new__(JobDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_dj_palmer)
        assert "in_place" in sig.parameters

    def test_get_schedule_by_dj_cds_has_all_common_params(self) -> None:
        """Test DJ-CDS has all common parameters."""
        dispatcher = JobDispatcher.__new__(JobDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_dj_cds)
        params = sig.parameters
        assert "schedule" in params
        assert "in_place" in params
        assert "from_stage" in params
        assert "job_2_release_t" in params

    def test_get_schedule_by_dj_gupta_has_all_common_params(self) -> None:
        """Test DJ-Gupta has all common parameters."""
        dispatcher = JobDispatcher.__new__(JobDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_dj_gupta)
        params = sig.parameters
        assert "schedule" in params
        assert "in_place" in params
        assert "from_stage" in params
        assert "job_2_release_t" in params

    def test_get_schedule_by_dj_palmer_has_all_common_params(self) -> None:
        """Test DJ-Palmer has all common parameters."""
        dispatcher = JobDispatcher.__new__(JobDispatcher)
        sig = inspect.signature(dispatcher.get_schedule_by_dj_palmer)
        params = sig.parameters
        assert "schedule" in params
        assert "in_place" in params
        assert "from_stage" in params
        assert "job_2_release_t" in params
