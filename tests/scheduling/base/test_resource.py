import pytest

from hybridflowshop.scheduling.base.activity import Activity
from hybridflowshop.scheduling.base.resource import Resource


class DummyActivity(Activity):
    def __init__(self, name, start, end):
        self._name = name
        self._start = start
        self._end = end

    @property
    def name(self):
        return self._name

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end


class DummyResource(Resource):
    @property
    def name(self):
        return "dummy"


def test_add_and_activity_list():
    res = DummyResource()
    a1 = DummyActivity("A", 0, 5)
    a2 = DummyActivity("B", 10, 15)
    assert res.add_activity(a1) == a1
    assert res.add_activity(a2) == a2
    assert res.activity_list == [a1, a2]
    assert res.get_activity_count_started_before(0) == 0
    assert res.get_activity_count_started_before(6) == 1
    assert res.get_activity_count_started_before(20) == 2


def test_check_for_time_conflict():
    res = DummyResource()
    a1 = DummyActivity("A", 0, 5)
    a2 = DummyActivity("B", 10, 15)
    res.add_activity(a1)
    res.add_activity(a2)
    # No conflict
    assert not res.check_for_time_conflict(5, 10)
    # Overlap with a1
    assert res.check_for_time_conflict(4, 6)
    # Overlap with a2
    assert res.check_for_time_conflict(12, 16)


def test_add_activity_conflict():
    res = DummyResource()
    a1 = DummyActivity("A", 0, 5)
    a2 = DummyActivity("B", 4, 8)
    res.add_activity(a1)
    # Conflict, should return None
    assert res.add_activity(a2) is None
    # force_add=True, should add anyway
    assert res.add_activity(a2, force_add=True) == a2


def test_makespan_and_clear():
    res = DummyResource()
    a1 = DummyActivity("A", 0, 5)
    a2 = DummyActivity("B", 10, 15)
    res.add_activity(a1)
    res.add_activity(a2)
    assert res.makespan == 15
    res.clear()
    assert res.makespan == 0
    assert res.activity_list == []


def test_get_earliest_start_time():
    res = DummyResource()
    a1 = DummyActivity("A", 2, 5)
    a2 = DummyActivity("B", 10, 15)
    res.add_activity(a1)
    res.add_activity(a2)
    # Between a1 and a2
    assert res.get_earliest_start_time(3, 5) == 5
    # After a2
    assert res.get_earliest_start_time(2, 15) == 15
    # Before all
    assert res.get_earliest_start_time(2, 0) == 0
    # Not enough gap between a1 and a2
    assert res.get_earliest_start_time(6, 0) == 15
    # duration <= 0
    with pytest.raises(ValueError):
        res.get_earliest_start_time(0, 0)
