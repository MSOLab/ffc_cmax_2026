import pytest
from routix import DynamicDataObject

from cpsat_solver_config import SolveConfig, configure_solver


def test_configure_solver_applies_extra_cp_sat_params() -> None:
    solver = configure_solver(
        SolveConfig(
            use_lns_only=True,
            cp_sat_params={
                "diversify_lns_params": True,
                "solution_pool_size": 8,
                "filter_subsolvers": ["*lns"],
            },
        )
    )

    assert solver.parameters.use_lns_only is True
    assert solver.parameters.diversify_lns_params is True
    assert solver.parameters.solution_pool_size == 8
    assert list(solver.parameters.filter_subsolvers) == ["*lns"]


def test_configure_solver_rejects_unknown_extra_cp_sat_param() -> None:
    with pytest.raises(ValueError, match="Unknown CP-SAT parameter"):
        configure_solver(SolveConfig(cp_sat_params={"not_a_cp_sat_param": True}))


def test_configure_solver_accepts_dynamic_data_object_extra_params() -> None:
    solver = configure_solver(
        SolveConfig(
            cp_sat_params=DynamicDataObject.from_obj(
                {"diversify_lns_params": True, "solution_pool_size": 16}
            )
        )
    )

    assert solver.parameters.diversify_lns_params is True
    assert solver.parameters.solution_pool_size == 16
