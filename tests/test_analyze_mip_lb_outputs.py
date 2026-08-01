import json

from routix.io.yaml import dump_yaml

from lb_bucket.analyze_mip_lb_outputs import analyze_scenario_dir


def test_analyze_mip_lb_outputs_writes_summary_files(tmp_path) -> None:
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    dump_yaml(
        [
            {"method": "set_random_seed", "seed": 42},
            {"method": "apply_mip_lb", "tl_nc_multiplier": 0.01, "delta": 1200},
        ],
        scenario_dir / "subroutine_flow.yaml",
    )

    instance_dir = scenario_dir / "1"
    (instance_dir / "mip_lb" / "solutions" / "1").mkdir(parents=True)
    (instance_dir / "mip_lb" / "dispatch").mkdir(parents=True)

    metadata = {
        "ins_name": "1",
        "job_count": 10,
        "stage_count": 5,
        "input_ub": 123.0,
        "time_limit_sec_used": 0.5,
        "total_runtime_sec": 0.25,
        "wall_runtime_sec": 0.35,
        "model_build_wall_sec": 0.1,
    }
    (instance_dir / "mip_lb" / "solutions" / "1" / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    dump_yaml(
        {
            "selected_variant": "best_of_mixed_dispatches_tail_ls_rank",
            "selected_makespan": 120.0,
            "dispatch_candidates": {
                "best_of_mixed_dispatches_tail_ls_rank": 120.0,
                "bn2d_all_stages": 122.0,
            },
            "dispatch_candidate_elapsed_sec": {
                "best_of_mixed_dispatches_tail_ls_rank": 0.4,
                "bn2d_all_stages": 0.2,
            },
        },
        instance_dir / "mip_lb" / "dispatch" / "dispatch_summary.yaml",
    )
    (instance_dir / "subroutine_controller.log").write_text(
        "\n".join(
            [
                "2026-04-08 00:00:00,000 - INFO - Completed instance 1: W*=120.000000, certified_final_lb=120.000000, total_runtime_sec=0.25, wall_runtime_sec=0.35, model_build_wall_sec=0.10",
                "2026-04-08 00:00:00,010 - INFO - {'method': 'apply_mip_lb', 'elapsed_sec': 2.0}",
            ]
        ),
        encoding="utf-8",
    )

    summary = analyze_scenario_dir(scenario_dir)

    analysis_dir = scenario_dir / "mip_lb_analysis"
    assert summary["analyzed_instance_count"] == 1
    assert summary["missing_instance_count"] == 0
    assert (analysis_dir / "mip_lb_dispatch_instance_summary.csv").is_file()
    assert (analysis_dir / "mip_lb_dispatch_candidate_long.csv").is_file()
    assert (analysis_dir / "mip_lb_dispatch_variant_summary.csv").is_file()
    assert (analysis_dir / "mip_lb_dispatch_analysis.yaml").is_file()
