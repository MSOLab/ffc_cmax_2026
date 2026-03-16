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

    batches = ctor._build_stage_batches(sched, batch_size=2, sort_by_start_time=False)
    partition = ctor._build_operation_partition(
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
    hint_values = ctor._compute_right_guard_hint_values(
        incumbent=sched,
        optimization_ops=partition.optimization,
        right_boundary_profile=partition.right_boundary_profile,
        params=params,
    )

    print("Feasible schedule:")
    print("  m1: A [0,10]")
    print("  m2: B [0,2], C [2,4], D [8,10]")
    print()
    print(f"Batches: {batches}")
    print(f"Optimization ops: {partition.optimization}")
    print(f"Right fixed ops: {partition.right_time_fixed_ops}")
    print(f"Right boundary profile: {partition.right_boundary_profile}")
    print()
    print("Right guard hint values:")
    print(f"  guard_start_s1_1 = {hint_values['guard_start_s1_1']}")
    print(f"  guard_start_s1_2 = {hint_values['guard_start_s1_2']}")
    print(f"  guard_extra_s1_1 = {hint_values['guard_extra_s1_1']}")
    print(f"  guard_extra_s1_2 = {hint_values['guard_extra_s1_2']}")
    print(f"  stage_guard_min_s1 = {hint_values['stage_guard_min_s1']}")
    print(f"  global_guard_min = {hint_values['global_guard_min']}")


if __name__ == "__main__":
    main()
