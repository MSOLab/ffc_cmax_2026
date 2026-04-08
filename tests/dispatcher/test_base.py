"""Tests for BaseDispatcher class."""

from hybridflowshop.dispatcher import (
    BaseDispatcher,
    BN2DDispatcher,
    JobDispatcher,
    MixedDispatcher,
    StageDispatcher,
)


class TestBaseDispatcher:
    """Tests for BaseDispatcher class."""

    def test_get_johnsons_rule_sequence(self) -> None:
        """Test Johnson's rule sequence generation (static method, no instance needed)."""
        dispatcher = BaseDispatcher.__new__(BaseDispatcher)
        # Simple 2-stage problem for Johnson's rule
        job_name_2_p1_map = {"A": 3, "B": 2, "C": 4}
        job_name_2_p2_map = {"A": 2, "B": 3, "C": 1}
        sequence = dispatcher.get_johnsons_rule_sequence(
            job_name_2_p1_map, job_name_2_p2_map
        )
        # Johnson's rule: jobs with p1 <= p2 go first (sorted by p1), others go last (sorted by p2 desc)
        # A: p1=3, p2=2 -> p1 > p2, goes to L2
        # B: p1=2, p2=3 -> p1 <= p2, goes to L1
        # C: p1=4, p2=1 -> p1 > p2, goes to L2
        # L1 = [B], L2 = [A, C] sorted by p2 desc -> [A, C] (A has p2=2, C has p2=1)
        # Result: [B, A, C]
        assert sequence == ["B", "A", "C"]

    def test_get_johnsons_rule_sequence_uses_tiebreak_rank_when_available(self) -> None:
        """ES/LS rank should break Johnson-sequence ties when provided."""
        dispatcher = BaseDispatcher.__new__(BaseDispatcher)
        dispatcher.job_tiebreak_rank = {"A": 1, "B": 0}
        dispatcher.job_id_2_original_index = {"A": 0, "B": 1}

        sequence = dispatcher.get_johnsons_rule_sequence(
            {"A": 2, "B": 2},
            {"A": 5, "B": 5},
        )

        assert sequence == ["B", "A"]


class TestDispatcherStructure:
    """Tests for dispatcher class hierarchy."""

    def test_stage_dispatcher_extends_base_dispatcher(self) -> None:
        """Test StageDispatcher extends BaseDispatcher."""
        assert issubclass(StageDispatcher, BaseDispatcher)

    def test_job_dispatcher_extends_base_dispatcher(self) -> None:
        """Test JobDispatcher extends BaseDispatcher."""
        assert issubclass(JobDispatcher, BaseDispatcher)

    def test_mixed_dispatcher_extends_base_dispatcher(self) -> None:
        """Test MixedDispatcher extends BaseDispatcher."""
        assert issubclass(MixedDispatcher, BaseDispatcher)

    def test_bn2d_dispatcher_extends_base_dispatcher(self) -> None:
        """Test BN2DDispatcher extends BaseDispatcher."""
        assert issubclass(BN2DDispatcher, BaseDispatcher)
