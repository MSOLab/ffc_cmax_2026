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
    )
    assert c.is_loop_stopping_condition(
        loop_count=2,
        no_improvement_steps=0,
        global_remaining_sec=100.0,
        lb_gap=None,
    )


def test_stop_at_global_timelimit_minus():
    c = LocalStoppingCriteria({"stop_at_global_timelimit_minus": 5})
    # global_remaining_sec greater -> don't stop
    assert not c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=10.0,
        lb_gap=None,
    )
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=5.0,
        lb_gap=None,
    )
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=3.0,
        lb_gap=None,
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
    )
    # no improvement steps
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=2,
        global_remaining_sec=100.0,
        lb_gap=1.0,
    )
    # lb gap
    assert c.is_loop_stopping_condition(
        loop_count=0,
        no_improvement_steps=0,
        global_remaining_sec=100.0,
        lb_gap=0.05,
    )
