"""
Demo: Verify positive boundary deviation in PW-CP.

Initial commit intent (06f7ebe3):
- Demonstrate cumulative frontier-based boundary deviation calculation
- Show when free operations extend BEYOND guard availability (positive deviation)
- This is the problematic case the boundary deviation objective addresses

Current state:
- Method renamed to _compute_right_slack_hint_values (refactored in 4f0c8107)
- Original _compute_boundary_deviation_hint_values was removed
- This demo now shows both the current right-slack logic AND the original
  boundary deviation analysis for cross-validation
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybridflowshop.controller.pw_cp import PwCpConstructor
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule
from tests.test_dispatch_stage_by_machines import create_hfs_instance


class _DummyContext:
    def __init__(self):
        self.solver = None

    def get_remaining_time_limit(self, subroutine_time_limit):
        return 1.0 if subroutine_time_limit is None else subroutine_time_limit

    def solve_cp_model_2(self, *args, **kwargs):
        raise NotImplementedError

    def create_schedule(self, *args, **kwargs):
        raise NotImplementedError

    def check_feasibility(self, start_time_map):
        return 0.0

    def get_file_path_for_subroutine(self, suffix):
        from pathlib import Path

        return Path("/tmp") / suffix.lstrip("_")


def _compute_usage_frontier_from_intervals(
    intervals: list[tuple[int, int]],
    *,
    mc_cnt: int,
    fallback: int,
    use_earliest: bool,
) -> list[int]:
    """
    Compute usage frontier from operation intervals using sweep-line algorithm.

    This is the original implementation from commit 06f7ebe3, preserved here
    for demonstration purposes since it was removed in subsequent refactoring.

    Args:
        intervals: List of (start, end) intervals
        mc_cnt: Number of machines (capacity)
        fallback: Default value when no intervals exist
        use_earliest: If True, record first time each level becomes active;
                     if False, record last time each level is active

    Returns:
        List of frontier values for each machine level (1-indexed conceptually)
    """
    frontiers = [fallback] * mc_cnt
    if not intervals:
        return frontiers

    time_2_delta: dict[int, int] = {}
    for start, end in intervals:
        time_2_delta[int(start)] = time_2_delta.get(int(start), 0) + 1
        time_2_delta[int(end)] = time_2_delta.get(int(end), 0) - 1

    event_times = sorted(time_2_delta)
    usage = 0
    seen = [False] * mc_cnt
    for idx, time in enumerate(event_times):
        usage += time_2_delta[time]
        next_time = event_times[idx + 1] if idx + 1 < len(event_times) else None
        if next_time is None or next_time <= time:
            continue

        active_levels = min(usage, mc_cnt)
        if use_earliest:
            for level in range(1, active_levels + 1):
                if not seen[level - 1]:
                    frontiers[level - 1] = int(time)
                    seen[level - 1] = True
        else:
            for level in range(1, active_levels + 1):
                frontiers[level - 1] = int(next_time)
    return frontiers


def main() -> None:
    ctor = PwCpConstructor(_DummyContext())

    sched = HybridFlowshopLiteSchedule(
        jobs=["A", "B", "C", "D"],
        stages=["s1"],
        machines_per_stage={"s1": ["m1", "m2"]},
    )
    # Feasible schedule:
    # m1: A [0,10]
    # m2: B [0,2], C [2,4], D [8,10]
    sched.add_ops_times_2_mc("s1", "m1", "A", 0, 10)
    sched.add_ops_times_2_mc("s1", "m2", "B", 0, 2)
    sched.add_ops_times_2_mc("s1", "m2", "C", 2, 4)
    sched.add_ops_times_2_mc("s1", "m2", "D", 8, 10)

    batches = ctor.build_stage_2_batch_list(sched, batch_size=2, sort_by_start_time=False)
    partition, _ = ctor._build_operation_partition(
        batches,
        ["s1"],
        current_batch_idx=0,
        incumbent=sched,
        stage_2_job_2_p_dict={"s1": {"A": 10, "B": 2, "C": 2, "D": 2}},
    )

    instance = create_hfs_instance(
        "tiny",
        ["A", "B", "C", "D"],
        ["s1"],
        {"s1": ["m1", "m2"]},
        {"A": {"s1": 10}, "B": {"s1": 2}, "C": {"s1": 2}, "D": {"s1": 2}},
    )
    _, params, _ = ctor.builder.build(instance, sched.makespan)
    hint_values = ctor._compute_right_slack_hint_values(
        schedule=sched,
        slack_occupying_ops=partition.unfixed,
        right_boundary_profile=partition.right_boundary_profile,
        params=params,
    )

    # Build interval map from free ops
    start_time_map = sched.get_jik_2_start_time_map()
    end_time_map = sched.get_jik_2_end_time_map()
    stage_2_intervals: dict[str, list[tuple[int, int]]] = {}
    machine_2_intervals: dict[str, dict[str, list[tuple[int, int]]]] = {}

    for job_id, stage_id, machine_id in partition.unfixed:
        start = int(start_time_map[job_id, stage_id, machine_id])
        end = int(end_time_map[job_id, stage_id, machine_id])
        stage_2_intervals.setdefault(stage_id, []).append((start, end))
        machine_2_intervals.setdefault(stage_id, {}).setdefault(machine_id, []).append(
            (start, end)
        )

    print("=" * 80)
    print("PW-CP POSITIVE BOUNDARY DEVIATION DEMONSTRATION")
    print("=" * 80)
    print()
    print("PURPOSE: Show when free operations extend BEYOND guard availability")
    print("         (i.e., positive boundary deviation that the CP model must handle)")
    print()
    print("-" * 80)
    print("INPUT: INCUMBENT SCHEDULE")
    print("-" * 80)
    print("  Machine 1 (m1): Job A [0, 10]")
    print("  Machine 2 (m2): Job B [0, 2] -> Job C [2, 4] ...gap... Job D [8, 10]")
    print()
    print("-" * 80)
    print("OPERATION PARTITION (current batch: idx=0, batch_size=2)")
    print("-" * 80)
    print("  FREE OPS (to be optimized by CP):")
    print(f"    {partition.unfixed}")
    print("      -> B@ms1 [0,2], C@ms1 [2,4]")
    print("      -> These operations' positions will be optimized")
    print()
    print("  RIGHT TIME-FIXED OPS (guard - fixed start times):")
    print(f"    {partition.right_time_fixed}")
    print("      -> A@ms1 [0,10], D@ms1 [8,10]")
    print("      -> These operations' start times are FIXED in CP model")
    print()

    print("-" * 80)
    print("RIGHT BOUNDARY PROFILE (guard machine availability start times)")
    print("-" * 80)
    if partition.right_boundary_profile:
        for stage_id, profile in partition.right_boundary_profile.items():
            machines = params.M_of[stage_id]
            for idx, (machine_id, start_time) in enumerate(
                zip(machines, profile), start=1
            ):
                print(
                    f"  {stage_id} / {machine_id} (machine #{idx}): GUARD available from t={start_time}"
                )
    print()
    print("  INTERPRETATION:")
    print("    - Guard guarantees these machines are FREE starting at these times")
    print("    - Machine 1: FREE from t=0 (no guard protection needed)")
    print("    - Machine 2: FREE from t=8 (protected by Job D [8,10])")
    print()

    # Compute usage frontier using original logic
    if "s1" in stage_2_intervals:
        mc_cnt = len(params.M_of["s1"])
        frontier = _compute_usage_frontier_from_intervals(
            stage_2_intervals["s1"], mc_cnt=mc_cnt, fallback=0, use_earliest=False
        )
        boundary = partition.right_boundary_profile["s1"]

        print("=" * 70)
        print("ORIGINAL COMMIT ANALYSIS: Frontier vs Boundary Deviation")
        print("=" * 70)
        print()
        print("USAGE FRONTIER COMPUTATION (from free ops):")
        print(f"  Input intervals: {stage_2_intervals['s1']}")
        print(f"  Machine count: {mc_cnt}")
        print(f"  Computed frontier: {frontier}")
        print()

        print("BOUNDARY PROFILE (guard availability start times):")
        print(f"  {boundary}")
        print()

        print("DEVIATION ANALYSIS (frontier - boundary):")
        all_deviations = []
        for level, (frontier_t, boundary_start) in enumerate(
            zip(frontier, boundary), start=1
        ):
            deviation = frontier_t - boundary_start
            all_deviations.append(deviation)
            print(f"  Level {level}:")
            print(f"    T_{level} (frontier) = {frontier_t}")
            print(f"    boundary_start = {boundary_start}")
            print(f"    deviation = {frontier_t} - {boundary_start} = {deviation}")

        print()
        print("SUMMARY:")
        print(f"  All deviations: {all_deviations}")
        print(f"  Max deviation: {max(all_deviations) if all_deviations else 'N/A'}")
        print()

        if max(all_deviations) > 0:
            print("  *** POSITIVE DEVIATION DETECTED ***")
            print()
            print("  MEANING:")
            print("    - The cumulative usage frontier of free ops extends BEYOND")
            print("      the guard's availability boundary")
            print("    - Level 2: T_2=4 > boundary_start=8 is FALSE (OK)")
            print("    - Level 1: T_1=4 > boundary_start=0 is TRUE (POSITIVE)")
            print()
            print("  IMPLICATION:")
            print("    Positive deviation means optimization ops consume resources")
            print("    beyond what the guard guarantees, potentially causing")
            print("    infeasibility or suboptimal solutions.")
            print("    The boundary deviation objective penalizes this.")
        else:
            print("  No positive deviation - all frontier values within guard bounds.")
        print()

    print("=" * 70)
    print("CURRENT IMPLEMENTATION: Right Slack Hints")
    print("=" * 70)
    print()
    print("Right slack hint values (from _compute_right_slack_hint_values):")
    for key, value in sorted(hint_values.items()):
        print(f"  {key} = {value}")
    print()

    print("INTERPRETATION:")
    print(
        f"  slack_start_s1_1 = {hint_values['slack_start_s1_1']}: "
        f"Machine 1 slack starts at t={hint_values['slack_start_s1_1']}"
    )
    print(
        f"  slack_start_s1_2 = {hint_values['slack_start_s1_2']}: "
        f"Machine 2 slack starts at t={hint_values['slack_start_s1_2']}"
    )
    print(
        f"  slack_length = {hint_values['slack_length']}: "
        "Unified slack length (all machines share same length)"
    )
    print()

    print("=" * 70)
    print("INITIAL COMMIT vs CURRENT STATE COMPARISON")
    print("=" * 70)
    print()
    print("Initial commit (06f7ebe3) aimed to:")
    print("  - Compute boundary deviation using cumulative frontiers")
    print("  - Create variables: T_s1_1, T_s1_2 (frontier values)")
    print("  - Create variables: boundary_deviation_s1_1, boundary_deviation_s1_2")
    print("  - Maximize minimum slack by minimizing boundary deviation")
    print()
    print("Current state (after 4f0c8107 refactor):")
    print("  - Method renamed: _compute_boundary_deviation_hint_values ->")
    print("                  _compute_right_slack_hint_values")
    print("  - Focus shifted from 'deviation' to 'slack' computation")
    print("  - Frontier-based analysis removed from main code")
    print("  - Right guard profile now computed differently")
    print()
    print("This demo preserves the original frontier computation logic to")
    print("demonstrate the boundary deviation concept that motivated the")
    print("initial implementation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
