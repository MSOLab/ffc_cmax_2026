from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from routix.type_defs import RunMode

import hfs_multi_instance_runner as multi_runner_module
import hybridflowshop.resume.validator as validator_module
from hfs_multi_instance_runner import HfsMultiInstanceRunner
from hybridflowshop.resume import ResumeValidator


class _FakeStore:
    def __init__(self, obj_value: float):
        self.obj_value = obj_value

    def get_last_obj_value(self) -> float:
        return self.obj_value


class _FakeObjValueBoundStore:
    obj_value = 123.0

    @staticmethod
    def load_yaml(path: Path) -> _FakeStore:
        return _FakeStore(_FakeObjValueBoundStore.obj_value)


class _FakeController:
    def __init__(self, obj_value: float):
        self.obj_value = obj_value
        self.base_cp_model_initialized = False

    def set_cp_model_as_base_cp_model(self) -> None:
        self.base_cp_model_initialized = True

    def check_feasibility(self, start_time_map: dict) -> float:
        assert start_time_map == {"op1": 0}
        assert self.base_cp_model_initialized is True
        return self.obj_value


class _FakeSingleInstanceRunner:
    obj_value = 123.0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_controller(self) -> _FakeController:
        return _FakeController(self.obj_value)


def _write_resume_files(
    tmp_path: Path,
    instance_name: str,
    *,
    best_obj: float = 123.0,
    with_results_dir: bool = True,
) -> Path:
    instance_dir = tmp_path / instance_name
    target_dir = instance_dir / "results" if with_results_dir else instance_dir
    target_dir.mkdir(parents=True)

    with open(
        target_dir / f"{instance_name}_solution.yaml", "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            {"start_time_map": {"op1": 0}, "end_time_map": {"op1": 5}},
            f,
            sort_keys=False,
        )

    (target_dir / f"{instance_name}_obj_log.yaml").write_text(
        "obj_value: {}\n", encoding="utf-8"
    )

    pd.DataFrame([{"bestObj": best_obj, "initObj": 150.0}]).to_csv(
        target_dir / f"{instance_name}_summary.csv", index=False
    )
    return tmp_path


def _build_validator(resume_root: Path) -> ResumeValidator:
    return ResumeValidator(
        instances=[SimpleNamespace(name="ins1")],
        shared_param_dict={},
        subroutine_flow=[],
        stopping_criteria=SimpleNamespace(),
        output_dir=resume_root / "out",
        output_metadata={"resume_root": str(resume_root)},
        mode=RunMode.RESUME,
        runner_class=_FakeSingleInstanceRunner,
    )


def test_resume_validator_loads_and_injects_resume_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_root = _write_resume_files(tmp_path / "resume", "ins1")
    validator = _build_validator(resume_root)
    monkeypatch.setattr(validator_module, "ObjValueBoundStore", _FakeObjValueBoundStore)

    data = validator.load_resume_data()
    runners = [SimpleNamespace()]
    validator.inject_resume_data_into_runners(runners)

    assert data.start_time_map_by_instance["ins1"] == {"op1": 0}
    assert data.end_time_map_by_instance["ins1"] == {"op1": 5}
    assert data.obj_value_by_instance["ins1"] == pytest.approx(123.0)
    assert data.summary_by_instance["ins1"]["bestObj"] == pytest.approx(123.0)
    assert runners[0].resume_start_time_map == {"op1": 0}
    assert runners[0].resume_end_time_map == {"op1": 5}
    assert runners[0].resume_summary_dict["bestObj"] == pytest.approx(123.0)


def test_resume_validator_requires_feasibility_before_obj_log() -> None:
    validator = _build_validator(Path("D:/resume-root"))

    with pytest.raises(RuntimeError, match="feasibility has not been checked"):
        validator.load_obj_store_check_resume_solution_obj_value()


def test_resume_validator_requires_obj_log_before_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_root = _write_resume_files(tmp_path / "resume", "ins1")
    validator = _build_validator(resume_root)
    monkeypatch.setattr(validator_module, "ObjValueBoundStore", _FakeObjValueBoundStore)

    validator.load_resume_solution_check_feasibility()

    with pytest.raises(RuntimeError, match="objective log has not been checked"):
        validator.load_summary_check_obj_values()


def test_resume_validator_detects_summary_best_obj_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_root = _write_resume_files(tmp_path / "resume", "ins1", best_obj=130.0)
    validator = _build_validator(resume_root)
    monkeypatch.setattr(validator_module, "ObjValueBoundStore", _FakeObjValueBoundStore)

    validator.load_resume_solution_check_feasibility()
    validator.load_obj_store_check_resume_solution_obj_value()

    with pytest.raises(RuntimeError, match="summary bestObj value 130.0"):
        validator.load_summary_check_obj_values()


def test_multi_instance_runner_resume_methods_delegate_to_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeResumeValidator:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self.data = SimpleNamespace(
                start_time_map_by_instance={"ins1": {"op1": 0}},
                end_time_map_by_instance={"ins1": {"op1": 5}},
                obj_value_by_instance={"ins1": 123.0},
                obj_store_by_instance={"ins1": "store"},
                summary_by_instance={"ins1": {"bestObj": 123.0}},
            )

        def load_resume_solution_check_feasibility(self) -> None:
            captured["feasibility"] = True

        def load_obj_store_check_resume_solution_obj_value(self) -> None:
            captured["obj_log"] = True

        def load_summary_check_obj_values(self) -> None:
            captured["summary"] = True

        def inject_resume_data_into_runners(self, runners) -> None:
            captured["runners"] = runners

    monkeypatch.setattr(multi_runner_module, "ResumeValidator", _FakeResumeValidator)

    runner = HfsMultiInstanceRunner.__new__(HfsMultiInstanceRunner)
    runner.instances = [SimpleNamespace(name="ins1")]
    runner.shared_param_dict = {}
    runner.subroutine_flow = []
    runner.stopping_criteria = SimpleNamespace()
    runner.output_dir = Path("D:/out")
    runner.output_metadata = {"resume_root": "D:/resume"}
    runner.mode = RunMode.RESUME
    runner.runners = [SimpleNamespace()]

    runner._load_resume_solution_check_feasibility()
    runner._load_obj_store_check_resume_solution_obj_value()
    runner._load_summary_check_obj_values()
    runner._inject_resume_data_into_runners()

    assert captured["feasibility"] is True
    assert captured["obj_log"] is True
    assert captured["summary"] is True
    assert captured["runners"] == runner.runners
    assert runner.ins_name_to_start_time_map_map == {"ins1": {"op1": 0}}
    assert runner.ins_name_to_end_time_map_map == {"ins1": {"op1": 5}}
    assert runner.ins_name_to_obj_value_map == {"ins1": 123.0}
    assert runner.ins_name_to_obj_store_map == {"ins1": "store"}
    assert runner.ins_name_to_summary_map == {"ins1": {"bestObj": 123.0}}
