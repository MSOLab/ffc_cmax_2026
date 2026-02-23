"""Tests for BaseDispatcher class."""

from hybridflowshop.dispatcher import (
    BaseDispatcher,
    Dispatcher,
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


class TestDispatcherStructure:
    """Tests for dispatcher class hierarchy."""

    def test_dispatcher_extends_mixed_dispatcher(self) -> None:
        """Test Dispatcher extends MixedDispatcher."""
        assert issubclass(Dispatcher, MixedDispatcher)

    def test_mixed_dispatcher_extends_base_dispatcher(self) -> None:
        """Test MixedDispatcher extends BaseDispatcher."""
        assert issubclass(MixedDispatcher, BaseDispatcher)

    def test_stage_dispatcher_extends_base_dispatcher(self) -> None:
        """Test StageDispatcher extends BaseDispatcher."""
        assert issubclass(StageDispatcher, BaseDispatcher)

    def test_job_dispatcher_extends_base_dispatcher(self) -> None:
        """Test JobDispatcher extends BaseDispatcher."""
        assert issubclass(JobDispatcher, BaseDispatcher)
