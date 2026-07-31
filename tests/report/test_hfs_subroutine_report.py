from mbls.cpsat import CpsatSolverReport, CpsatStatus

from hybridflowshop.report.hfs_subroutine_report import HfsCpsatSolverReport


def _make_base_report() -> HfsCpsatSolverReport:
    return HfsCpsatSolverReport(
        elapsed_time=10.0,
        obj_value=100.0,
        obj_bound=95.0,
        status=CpsatStatus.FEASIBLE,
        is_init=False,
        subroutine_name="solve_base_cp_model",
        call_context="5-solve_base_cp_model",
        progress_obj_value_records=((0.0, 110.0), (5.0, 105.0), (10.0, 100.0)),
        progress_time_basis="local",
        obj_value_records=((0.0, 110.0), (5.0, 105.0), (10.0, 100.0)),
        obj_bound_records=((0.0, 90.0), (5.0, 93.0), (10.0, 95.0)),
    )


def test_copy_preserves_hfs_metadata() -> None:
    report = _make_base_report()
    copy = report.copy()

    assert copy.subroutine_name == "solve_base_cp_model"
    assert copy.call_context == "5-solve_base_cp_model"
    assert copy.progress_obj_value_records == (
        (0.0, 110.0),
        (5.0, 105.0),
        (10.0, 100.0),
    )
    assert copy.progress_time_basis == "local"


def test_copy_preserves_hfs_metadata_when_updating_other_fields() -> None:
    report = _make_base_report()
    copy = report.copy(
        elapsed_time=20.0,
        obj_value=98.0,
    )

    assert copy.elapsed_time == 20.0
    assert copy.obj_value == 98.0
    assert copy.subroutine_name == "solve_base_cp_model"
    assert copy.call_context == "5-solve_base_cp_model"
    assert copy.progress_obj_value_records == (
        (0.0, 110.0),
        (5.0, 105.0),
        (10.0, 100.0),
    )
    assert copy.progress_time_basis == "local"


def test_copy_can_override_hfs_metadata() -> None:
    report = _make_base_report()
    copy = report.copy(
        subroutine_name="override_name",
        call_context="override_context",
        progress_time_basis="global",
    )

    assert copy.subroutine_name == "override_name"
    assert copy.call_context == "override_context"
    assert copy.progress_time_basis == "global"
    assert copy.progress_obj_value_records == (
        (0.0, 110.0),
        (5.0, 105.0),
        (10.0, 100.0),
    )


def test_from_other_correctly_maps_fields() -> None:
    other = CpsatSolverReport(
        elapsed_time=5.0,
        obj_value=100.0,
        obj_bound=90.0,
        status=CpsatStatus.FEASIBLE,
        obj_value_records=[(0.0, 120.0), (5.0, 100.0)],
        obj_bound_records=[(0.0, 80.0), (5.0, 90.0)],
    )

    report = HfsCpsatSolverReport.from_other(other, is_init=True)

    assert report.elapsed_time == 5.0
    assert report.obj_value == 100.0
    assert report.obj_bound == 90.0
    assert report.status == CpsatStatus.FEASIBLE
    assert report.is_init is True
    assert report.progress_obj_value_records == ((0.0, 120.0), (5.0, 100.0))
    assert report.progress_time_basis == "local"


def test_is_feasible_logic() -> None:
    # Feasible case
    r1 = _make_base_report()  # status=FEASIBLE, obj_value=100.0
    assert r1.is_feasible is True

    # Infeasible status case
    r2 = r1.copy(status=CpsatStatus.INFEASIBLE)
    assert r2.is_feasible is False

    # No objective value case
    r3 = r1.copy(obj_value=None)
    assert r3.is_feasible is False


def test_to_string_dict_formatting() -> None:
    report = _make_base_report()
    d = report.to_string_dict()

    assert "subroutine_name" in d
    assert d["subroutine_name"] == "solve_base_cp_model"
    assert "call_context" in d
    assert d["call_context"] == "5-solve_base_cp_model"
    # Ensure records are wrapped in double quotes for CSV
    assert d["obj_value_records"].startswith('"') and d["obj_value_records"].endswith('"')
    assert d["obj_bound_records"].startswith('"') and d["obj_bound_records"].endswith('"')
    # Check that status enum is converted to string
    assert isinstance(d["status"], str)
    assert d["status"] == CpsatStatus.FEASIBLE.to_solver_status_enum().value
