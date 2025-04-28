from typing import Any

from clad.dynamic_data_object import DynamicDataObject


class StoppingCriteria(DynamicDataObject):
    """Stopping criteria for the hybrid flow shop scheduling problem.

    Attributes:
        timelimit (int): Maximum time allowed for the algorithm to run.
    """

    timelimit: float

    def __init__(self, param_dict: dict[str, Any]):
        assert "timelimit" in param_dict, "timelimit is required in stopping criteria"
        super().__init__(param_dict)
