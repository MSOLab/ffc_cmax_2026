import pytest

from hybridflowshop.scheduling.base.activity import Activity


class DummyActivity(Activity):
    def __init__(self, name: str, start: int, end: int):
        self._name = name
        self._start = start
        self._end = end

    @property
    def name(self) -> str:
        return self._name

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._end


def test_activity_properties():
    act = DummyActivity("A", 10, 20)
    assert act.name == "A"
    assert act.start == 10
    assert act.end == 20
    assert act.duration == 10


def test_activity_duration_zero():
    act = DummyActivity("B", 5, 5)
    with pytest.raises(ValueError, match="zero duration"):
        _ = act.duration


def test_activity_negative_duration():
    act = DummyActivity("C", 7, 3)
    with pytest.raises(ValueError, match="invalid duration"):
        _ = act.duration
