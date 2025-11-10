from typing import Hashable

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import IntervalVar, IntVar


class CpModelWithFixedInterval(CustomCpModel):
    # Parameter

    horizon: int
    """
    The horizon for the scheduling problem, which is the maximum time
    that any operation can start or end.
    This is used to define the domain of the start and end time variables.
    """

    # Variables

    var_op_start: dict[Hashable, IntVar]
    """
    Dictionary to store start time variables for each operation in a job.
    """
    var_op_end: dict[Hashable, IntVar]
    """
    Dictionary to store end time variables for each operation in a job.
    """
    var_op_intvl: dict[Hashable, IntervalVar]
    """
    Dictionary to store interval variables for each operation in a job.
    """

    # Interval variable name set
    var_op_intvl_fixed_name_set: set[str]

    def __init__(self, horizon: int):
        """Initialize the CpModelWithFixedInterval class.

        Args:
            horizon (int): The horizon for the scheduling problem,
                           which is the maximum time that any operation can end.
        """
        super().__init__()

        # Parameter
        self.horizon = horizon

        # Initialize dictionaries to store variables
        self.var_op_start = {}
        self.var_op_end = {}
        self.var_op_intvl = {}

        # Initialize the set to store interval variable names
        self.var_op_intvl_fixed_name_set = set()

    def define_fixed_interval_var(self, name: Hashable, processing_time: int):
        """Define a fixed interval variable for an operation.

        Args:
            name (Hashable): The name of the operation.
            processing_time (int): The processing time of the operation.
        """
        suffix = self._from_name_to_var_name_suffix(name)
        var_name = f"intvlFixed_{suffix}"
        # Check if the interval variable name already exists
        if var_name in self.var_op_intvl_fixed_name_set:
            raise ValueError(
                f"Fixed interval variable name '{var_name}' already exists."
            )

        # Create the fixed interval variable with fixed processing time
        start_var = self.NewIntVar(0, self.horizon, f"start_{suffix}")
        end_var = self.NewIntVar(0, self.horizon, f"end_{suffix}")
        intvl_var = self.new_interval_var(start_var, processing_time, end_var, var_name)

        # Add each variable to the corresponding dictionaries
        self.var_op_start[name] = start_var
        self.var_op_end[name] = end_var
        self.var_op_intvl[name] = intvl_var

        # Add name of the new variable to the set
        self.var_op_intvl_fixed_name_set.add(var_name)

    @staticmethod
    def _from_name_to_var_name_suffix(name: Hashable) -> str:
        """
        Convert an operation name to a variable name suffix.

        - If name is tuple, return "_".join(name).
        - If name is not a tuple, return str(name).

        Args:
            name (Hashable): The name of the operation.

        Returns:
            str: The variable name suffix.
        """
        if isinstance(name, tuple):
            return "_".join(map(str, name))
        return str(name)
