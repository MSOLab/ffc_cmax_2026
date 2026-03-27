from hybridflowshop.painter.gantt import GanttPlotter


def test_draw_operation_bars_highlights_by_job_and_stage() -> None:
    plotter = GanttPlotter()
    calls = []

    def fake_draw_operation_bar(**kwargs):
        calls.append(kwargs)

    plotter.draw_operation_bar = fake_draw_operation_bar  # type: ignore[method-assign]

    plotter.draw_operation_bars(
        start_time_map={
            ("j1", "s1", "m2"): 0,
            ("j2", "s1", "m1"): 1,
        },
        end_time_map={
            ("j1", "s1", "m2"): 3,
            ("j2", "s1", "m1"): 4,
        },
        job_to_color={
            "j1": (1.0, 0.0, 0.0, 1.0),
            "j2": (0.0, 1.0, 0.0, 1.0),
        },
        machine_to_y={
            ("s1", "m1"): 0.0,
            ("s1", "m2"): 1.0,
        },
        job_list=["j1", "j2"],
        highlight_op_set={("j1", "s1")},
    )

    assert [call["highlight"] for call in calls] == [True, False]
