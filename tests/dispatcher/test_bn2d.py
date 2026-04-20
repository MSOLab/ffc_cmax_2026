"""Tests for BN2D dispatcher and utilities."""

import io

import pytest
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hybridflowshop.dispatcher import BN2DDispatcher, BN2DOption
from hybridflowshop.dispatcher.utils import reverse_even_positions


@pytest.fixture
def simple_hfs_instance():
    """Create a simple 3-job, 2-stage hybrid flow shop instance for testing.

    PRA format:
    - Line 1: job count
    - Line 2: stage count
    - Job count lines follow, each with stage_count processing times
    """
    data = """\
3
2
2 3
1 2
3 1
4 2
"""
    stream = io.StringIO(data)
    return HybridFlowshopParameters.from_pra_data("test_simple", stream)


@pytest.fixture
def two_stage_hfs_instance():
    """Create a 3-job, 2-stage instance with different processing times."""
    data = """\
3
2
2 3
1 2
3 1
4 2
"""
    stream = io.StringIO(data)
    return HybridFlowshopParameters.from_pra_data("test_2stage", stream)


@pytest.fixture
def three_stage_hfs_instance():
    """Create a 4-job, 3-stage hybrid flow shop instance for testing."""
    data = """\
4
3
2 3 1
1 2 3
3 1 2
2 2 1
4 1 3
"""
    stream = io.StringIO(data)
    return HybridFlowshopParameters.from_pra_data("test_3stage", stream)


@pytest.fixture
def multi_machine_hfs_instance():
    """Create a 3-job, 2-stage instance with multiple machines per stage."""
    data = """\
3
2
2 3
1 2
3 1
2 1
"""
    stream = io.StringIO(data)
    return HybridFlowshopParameters.from_pra_data("test_multi_mc", stream)


@pytest.fixture
def five_stage_hfs_instance():
    """Create a 4-job, 5-stage single-machine instance for anchor-band tests."""
    data = """\
4
5
1 1 1 1 1
2 3 4 3 2
3 2 1 2 3
1 4 2 4 1
4 1 3 1 4
"""
    stream = io.StringIO(data)
    return HybridFlowshopParameters.from_pra_data("test_5stage", stream)


class TestReverseEvenPositions:
    """Tests for reverse_even_positions utility function."""

    def test_reverse_even_positions_basic(self) -> None:
        """Test basic even position reversal."""
        seq = ["A", "B", "C", "D", "E", "F", "G", "H"]
        result = reverse_even_positions(seq, in_place=False)
        # Even positions (1-based): B, D, F, H -> reversed: H, F, D, B
        assert result == ["A", "H", "C", "F", "E", "D", "G", "B"]

    def test_reverse_even_positions_odd_length(self) -> None:
        """Test reversal with odd length sequence."""
        seq = ["A", "B", "C", "D", "E"]
        result = reverse_even_positions(seq, in_place=False)
        # Even positions: B, D -> reversed: D, B
        assert result == ["A", "D", "C", "B", "E"]

    def test_reverse_even_positions_empty(self) -> None:
        """Test reversal with empty sequence."""
        seq: list[str] = []
        result = reverse_even_positions(seq, in_place=False)
        assert result == []

    def test_reverse_even_positions_single_element(self) -> None:
        """Test reversal with single element."""
        seq = ["A"]
        result = reverse_even_positions(seq, in_place=False)
        assert result == ["A"]

    def test_reverse_even_positions_two_elements(self) -> None:
        """Test reversal with two elements."""
        seq = ["A", "B"]
        result = reverse_even_positions(seq, in_place=False)
        # Even position: B -> reversed: B
        assert result == ["A", "B"]

    def test_reverse_even_positions_in_place(self) -> None:
        """Test that in_place=True modifies original list."""
        seq = ["A", "B", "C", "D", "E", "F", "G", "H"]
        original_id = id(seq)
        result = reverse_even_positions(seq, in_place=True)
        assert id(seq) == original_id
        assert result is seq
        assert result == ["A", "H", "C", "F", "E", "D", "G", "B"]

    def test_reverse_even_positions_not_in_place(self) -> None:
        """Test that in_place=False returns new list."""
        seq = ["A", "B", "C", "D"]
        original_id = id(seq)
        result = reverse_even_positions(seq, in_place=False)
        assert id(result) != original_id
        assert result == ["A", "D", "C", "B"]
        # Original should be unchanged
        assert seq == ["A", "B", "C", "D"]


class TestBN2DOption:
    """Tests for BN2DOption dataclass."""

    def test_bn2d_option_default_values(self) -> None:
        """Test BN2DOption with default values."""
        option = BN2DOption()
        assert option.left_cap_multiplier is None
        assert option.right_cap_multiplier is None
        assert option.left_cap_portion is None
        assert option.right_cap_portion is None
        assert option.normalize_by_stage_cnt is False
        assert option.randomize_mid_all is False
        assert option.reverse_mid_even is False
        assert option.reverse_mid_all is False
        assert option.mixed_schedule_for_former_stages is False
        assert option.mixed_schedule_for_later_stages is False

    def test_bn2d_option_with_left_cap_multiplier(self) -> None:
        """Test BN2DOption with left_cap_multiplier."""
        option = BN2DOption(left_cap_multiplier=2)
        assert option.left_cap_multiplier == 2
        assert option.right_cap_multiplier is None

    def test_bn2d_option_with_right_cap_portion(self) -> None:
        """Test BN2DOption with right_cap_portion."""
        option = BN2DOption(right_cap_portion=0.3)
        assert option.right_cap_portion == pytest.approx(0.3)

    def test_bn2d_option_normalize_by_stage_cnt(self) -> None:
        """Test BN2DOption with normalize_by_stage_cnt."""
        option = BN2DOption(normalize_by_stage_cnt=True)
        assert option.normalize_by_stage_cnt is True

    def test_bn2d_option_reverse_modes_mutually_exclusive(self) -> None:
        """Test that reverse modes can be set independently."""
        option = BN2DOption(
            randomize_mid_all=True,
            reverse_mid_even=True,
            reverse_mid_all=True,
        )
        assert option.randomize_mid_all is True
        assert option.reverse_mid_even is True
        assert option.reverse_mid_all is True

    def test_bn2d_option_mixed_schedules(self) -> None:
        """Test BN2DOption with mixed schedule flags."""
        option = BN2DOption(
            mixed_schedule_for_former_stages=True,
            mixed_schedule_for_later_stages=True,
        )
        assert option.mixed_schedule_for_former_stages is True
        assert option.mixed_schedule_for_later_stages is True


class TestBN2DDispatcher:
    """Tests for BN2DDispatcher class."""

    def test_init(self, simple_hfs_instance) -> None:
        """Test BN2DDispatcher initialization."""
        dispatcher = BN2DDispatcher(simple_hfs_instance)
        assert dispatcher.instance == simple_hfs_instance

    def test_get_bottleneck_stage_basic(self, simple_hfs_instance) -> None:
        """Test bottleneck stage identification."""
        dispatcher = BN2DDispatcher(simple_hfs_instance)
        bottleneck = dispatcher._get_bottleneck_stage()
        # Bottleneck is stage with max total_p / machine_count
        assert bottleneck in simple_hfs_instance.stage_id_list

    def test_get_schedule_by_bn2d_all_stages(self, simple_hfs_instance) -> None:
        """Test BN2D schedule with all stages as bottleneck candidates."""
        dispatcher = BN2DDispatcher(simple_hfs_instance)
        option = BN2DOption()
        schedule = dispatcher.get_schedule_by_bn2d_all_stages(option)

        assert schedule is not None
        assert schedule.makespan > 0
        # Verify all jobs are scheduled at all stages
        for job in simple_hfs_instance.job_id_list:
            for stage in simple_hfs_instance.stage_id_list:
                assert schedule.get_job_end_time(stage, job) is not None

    def test_get_schedule_by_bn2d_single_stage(self, simple_hfs_instance) -> None:
        """Test BN2D schedule with single bottleneck stage."""
        dispatcher = BN2DDispatcher(simple_hfs_instance)
        option = BN2DOption()
        schedule = dispatcher.get_schedule_by_bn2d_single_stage(option)

        assert schedule is not None
        assert schedule.makespan > 0

    def test_get_schedule_by_bn2d_with_capping(self, simple_hfs_instance) -> None:
        """Test BN2D schedule with left/right cap using multipliers."""
        dispatcher = BN2DDispatcher(simple_hfs_instance)
        option = BN2DOption(
            left_cap_multiplier=1,
            right_cap_multiplier=1,
        )
        schedule = dispatcher.get_schedule_by_bn2d_all_stages(option)

        assert schedule is not None
        assert schedule.makespan > 0

    def test_get_schedule_by_bn2d_with_portion_capping(
        self, simple_hfs_instance
    ) -> None:
        """Test BN2D schedule with left/right cap using portions."""
        dispatcher = BN2DDispatcher(simple_hfs_instance)
        option = BN2DOption(
            left_cap_portion=0.3,
            right_cap_portion=0.3,
        )
        schedule = dispatcher.get_schedule_by_bn2d_all_stages(option)

        assert schedule is not None
        assert schedule.makespan > 0

    def test_get_schedule_by_bn2d_with_reverse_mid(
        self, simple_hfs_instance
    ) -> None:
        """Test BN2D schedule with mid-job reversal."""
        dispatcher = BN2DDispatcher(simple_hfs_instance)
        option = BN2DOption(
            reverse_mid_even=True,
        )
        schedule = dispatcher.get_schedule_by_bn2d_all_stages(option)

        assert schedule is not None
        assert schedule.makespan > 0

    def test_get_schedule_by_bn2d_normalize_by_stage_cnt(
        self, simple_hfs_instance
    ) -> None:
        """Test BN2D schedule with stage count normalization."""
        dispatcher = BN2DDispatcher(simple_hfs_instance)
        option = BN2DOption(
            normalize_by_stage_cnt=True,
        )
        schedule = dispatcher.get_schedule_by_bn2d_all_stages(option)

        assert schedule is not None
        assert schedule.makespan > 0

    def test_get_schedule_by_bn2d_with_mixed_later_stages(
        self, three_stage_hfs_instance
    ) -> None:
        """Test BN2D schedule with mixed schedules for later stages only."""
        dispatcher = BN2DDispatcher(three_stage_hfs_instance)
        option = BN2DOption(
            mixed_schedule_for_later_stages=True,
        )
        schedule = dispatcher.get_schedule_by_bn2d_all_stages(option)

        assert schedule is not None
        assert schedule.makespan > 0

    def test_get_schedule_by_bn2d_randomize_mid(
        self, simple_hfs_instance
    ) -> None:
        """Test BN2D schedule with randomized mid jobs."""
        dispatcher = BN2DDispatcher(simple_hfs_instance)
        option = BN2DOption(
            randomize_mid_all=True,
        )
        schedule = dispatcher.get_schedule_by_bn2d_all_stages(option)

        assert schedule is not None
        assert schedule.makespan > 0

    def test_get_schedule_by_bn2d_single_stage_specific_bottleneck(
        self, simple_hfs_instance
    ) -> None:
        """Test BN2D schedule for a specific bottleneck stage."""
        dispatcher = BN2DDispatcher(simple_hfs_instance)
        option = BN2DOption()

        # Schedule using single stage method
        schedule = dispatcher.get_schedule_by_bn2d_single_stage(option)

        assert schedule is not None
        assert schedule.makespan > 0

    def test_get_schedule_by_two_way_stage_band_respects_anchor_sequences(
        self,
        five_stage_hfs_instance,
    ) -> None:
        """Anchor-band two-way dispatch should preserve the given middle-stage order."""
        dispatcher = BN2DDispatcher(five_stage_hfs_instance)
        option = BN2DOption(
            mixed_schedule_for_former_stages=False,
            mixed_schedule_for_later_stages=False,
        )
        jobs = list(five_stage_hfs_instance.job_id_list)
        anchor_stage_ids = five_stage_hfs_instance.stage_id_list[1:4]
        stage_2_job_sequence = {
            anchor_stage_ids[0]: [jobs[0], jobs[1], jobs[2], jobs[3]],
            anchor_stage_ids[1]: [jobs[3], jobs[2], jobs[1], jobs[0]],
            anchor_stage_ids[2]: [jobs[1], jobs[3], jobs[0], jobs[2]],
        }
        stage_2_job_2_release = {
            anchor_stage_ids[0]: {
                jobs[0]: 5,
                jobs[1]: 0,
                jobs[2]: 3,
                jobs[3]: 1,
            },
            anchor_stage_ids[1]: {
                jobs[0]: 9,
                jobs[1]: 7,
                jobs[2]: 6,
                jobs[3]: 4,
            },
            anchor_stage_ids[2]: {
                jobs[0]: 13,
                jobs[1]: 11,
                jobs[2]: 12,
                jobs[3]: 10,
            },
        }

        schedule = dispatcher.get_schedule_by_two_way_stage_band(
            anchor_stage_ids,
            stage_2_job_sequence,
            option,
            stage_2_job_2_release=stage_2_job_2_release,
        )

        assert schedule is not None
        assert schedule.makespan > 0

        for job_id in five_stage_hfs_instance.job_id_list:
            for stage_id in five_stage_hfs_instance.stage_id_list:
                assert schedule.get_job_end_time(stage_id, job_id) is not None

        for stage_id in anchor_stage_ids:
            observed_jobs = [
                job_id
                for _mc_id, _start, _end, job_id in sorted(
                    schedule.iter_operations_on_stage(stage_id),
                    key=lambda row: (row[1], row[2], row[3]),
                )
            ]
            assert observed_jobs == stage_2_job_sequence[stage_id]

            for job_id, release_t in stage_2_job_2_release[stage_id].items():
                assert schedule.get_job_start_time(stage_id, job_id) >= release_t

    def test_get_schedule_by_two_way_stage_band_supports_strict_call_mode(
        self,
        five_stage_hfs_instance,
    ) -> None:
        dispatcher = BN2DDispatcher(five_stage_hfs_instance)
        option = BN2DOption(
            mixed_schedule_for_former_stages=False,
            mixed_schedule_for_later_stages=False,
        )
        jobs = list(five_stage_hfs_instance.job_id_list)
        anchor_stage_ids = five_stage_hfs_instance.stage_id_list[1:4]
        stage_2_job_sequence = {
            anchor_stage_ids[0]: [jobs[2], jobs[0], jobs[3], jobs[1]],
            anchor_stage_ids[1]: [jobs[1], jobs[3], jobs[0], jobs[2]],
            anchor_stage_ids[2]: [jobs[3], jobs[1], jobs[2], jobs[0]],
        }

        schedule = dispatcher.get_schedule_by_two_way_stage_band(
            anchor_stage_ids,
            stage_2_job_sequence,
            option,
            anchor_dispatch_mode="strict_call",
        )

        assert schedule is not None
        for job_id in five_stage_hfs_instance.job_id_list:
            for stage_id in five_stage_hfs_instance.stage_id_list:
                assert schedule.get_job_end_time(stage_id, job_id) is not None

        first_anchor_observed = [
            job_id
            for _mc_id, _start, _end, job_id in sorted(
                schedule.iter_operations_on_stage(anchor_stage_ids[0]),
                key=lambda row: (row[1], row[2], row[3]),
            )
        ]
        assert first_anchor_observed == stage_2_job_sequence[anchor_stage_ids[0]]
