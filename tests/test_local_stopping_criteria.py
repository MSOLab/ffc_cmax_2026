from hybridflowshop.controller.reactive.local_stopping_criteria import (
    LocalStoppingCriteria,
)


def test_max_loop_count_triggers():
    c = LocalStoppingCriteria({"max_loop_count": 2})
    assert not c.is_loop_stopping_condition(
        loop_count=1,
        no_improvement_steps=0,
        global_remaining_sec=100.0,
        lb_gap=None,
        global_timelimit=100.0,
    )
    assert c.is_loop_stopping_condition(
        loop_count=2,
        no_improvement_steps=0,
        global_remaining_sec=100.0,
        lb_gap=None,
        global_timelimit=100.0,
    )


def test_stop_at_global_timelimit_minus():
    c = LocalStoppingCriteria({"stop_at_global_timelimit_minus": 5})
    # global_remaining_sec greater -> don't stop
    assert not c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=10.0,
        lb_gap=None,
        global_timelimit=100.0,
    )
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=5.0,
        lb_gap=None,
        global_timelimit=100.0,
    )
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=3.0,
        lb_gap=None,
        global_timelimit=100.0,
    )


def test_rho_no_improve_and_lb_gap():
    c = LocalStoppingCriteria(
        {
            "rho_is_geq": 0.5,
            "max_no_improvement_steps": 2,
            "lb_gap_is_leq": 0.1,
        }
    )

    # rho threshold
    assert not c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=100.0,
        lb_gap=1.0,
        global_timelimit=100.0,
    )
    # no improvement steps
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=2,
        global_remaining_sec=100.0,
        lb_gap=1.0,
        global_timelimit=100.0,
    )
    # lb gap
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=100.0,
        lb_gap=0.05,
        global_timelimit=100.0,
    )


def test_stop_at_global_timelimit_minus_percent():
    c = LocalStoppingCriteria({"stop_at_global_timelimit_minus_percent": 0.05})
    global_timelimit = 100.0

    # remaining time 10s > 5s threshold (5% of 100s), don't stop
    assert not c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=10.0,
        lb_gap=None,
        global_timelimit=global_timelimit,
    )

    # remaining time 5s == 5s threshold, stop
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=5.0,
        lb_gap=None,
        global_timelimit=global_timelimit,
    )

    # remaining time 3s < 5s threshold, stop
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=3.0,
        lb_gap=None,
        global_timelimit=global_timelimit,
    )


def test_stop_at_global_timelimit_minus_and_percent_combined():
    """Test that both criteria work together, with the stricter one triggering."""
    # 100s global timelimit, 10s absolute threshold, 5% relative threshold (5s)
    # The 5s relative threshold is stricter
    c = LocalStoppingCriteria(
        {
            "stop_at_global_timelimit_minus": 10,
            "stop_at_global_timelimit_minus_percent": 0.05,
        }
    )
    global_timelimit = 100.0

    # 7s remaining: above absolute threshold (10s), but below relative (5s) - should stop
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=7.0,
        lb_gap=None,
        global_timelimit=global_timelimit,
    )

    # 12s remaining: above both thresholds - don't stop
    assert not c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=12.0,
        lb_gap=None,
        global_timelimit=global_timelimit,
    )

    # 8s remaining: below absolute threshold (10s), but above relative (5s) - should stop due to absolute
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=8.0,
        lb_gap=None,
        global_timelimit=global_timelimit,
    )
