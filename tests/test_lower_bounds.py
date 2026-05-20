from types import SimpleNamespace

from hybridflowshop.lower_bounds import (
    bin_packing_body_lower_bound,
    chen_lb4_lower_bound,
    chen_lb4_stage_lower_bounds,
    santos_lower_bound,
    santos_stage_lower_bound,
    simple_job_lower_bound,
)


class _FakeProcessingTimeManager:
    def __init__(self, values: dict[tuple[str, str], int]) -> None:
        self.values = values

    def job_stage_2_value_map(
        self, jobs: list[str], stages: list[str]
    ) -> dict[tuple[str, str], int]:
        return {
            (job, stage): self.values[job, stage]
            for job in jobs
            for stage in stages
        }


def _paper_example_instance():
    jobs = ["j1", "j2", "j3", "j4"]
    stages = ["i1", "i2", "i3", "i4"]
    rows = {
        "j1": [66, 53, 20, 87],
        "j2": [37, 81, 40, 92],
        "j3": [54, 48, 37, 97],
        "j4": [52, 81, 23, 43],
    }
    values = {
        (job, stage): rows[job][stage_idx]
        for job in jobs
        for stage_idx, stage in enumerate(stages)
    }
    return SimpleNamespace(
        job_id_list=jobs,
        stage_id_list=stages,
        stage_2_machines_map={
            stage: [f"{stage}_m1", f"{stage}_m2", f"{stage}_m3"]
            for stage in stages
        },
        p_manager=_FakeProcessingTimeManager(values),
    )


def test_bin_packing_body_lower_bound_matches_paper_example_stages() -> None:
    instance = _paper_example_instance()
    jobs = instance.job_id_list
    p = instance.p_manager.job_stage_2_value_map(jobs, instance.stage_id_list)

    assert bin_packing_body_lower_bound([p[job, "i1"] for job in jobs], 3) == 89
    assert bin_packing_body_lower_bound([p[job, "i2"] for job in jobs], 3) == 101
    assert bin_packing_body_lower_bound([p[job, "i3"] for job in jobs], 3) == 43
    assert bin_packing_body_lower_bound([p[job, "i4"] for job in jobs], 3) == 130


def test_chen_lb4_matches_paper_example() -> None:
    instance = _paper_example_instance()

    assert chen_lb4_stage_lower_bounds(instance) == {
        "i1": 250,
        "i2": 236,
        "i3": 236,
        "i4": 269,
    }
    assert chen_lb4_lower_bound(instance) == 269


def test_santos_lower_bound_remains_available_for_comparison() -> None:
    instance = _paper_example_instance()

    assert simple_job_lower_bound(instance) == 250
    assert [
        santos_stage_lower_bound(instance, stage)
        for stage in instance.stage_id_list
    ] == [233, 237, 227, 251]
    assert santos_lower_bound(instance) == 251
    assert santos_lower_bound(instance) <= chen_lb4_lower_bound(instance)
