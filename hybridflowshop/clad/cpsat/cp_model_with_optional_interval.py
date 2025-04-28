from collections import defaultdict

from ortools.sat.python.cp_model import IntervalVar, IntVar

from .custom_cp_model import CustomCpModel


class CpModelWithOptionalInterval(CustomCpModel):

    # Parameter

    horizon: int
    """
    The horizon for the scheduling problem, which is the maximum time
    that any operation can start or end.
    This is used to define the domain of the start and end time variables.
    """

    # Variables

    var_op_start: dict[str, dict[str, dict[str, IntVar]]]
    """
    Dictionary to store start time variables for each operation in a job.
    The keys are job names, stage names, and machine numbers.
    """
    var_op_end: dict[str, dict[str, dict[str, IntVar]]]
    """
    Dictionary to store end time variables for each operation in a job.
    The keys are job names, stage names, and machine numbers.
    """
    var_op_is_present: dict[str, dict[str, dict[str, IntVar]]]
    """
    Dictionary to store presence indicator variables for each operation in a job.
    The keys are job names, stage names, and machine numbers.
    """
    var_op_intvl: dict[str, dict[str, dict[str, IntervalVar]]]
    """
    Dictionary to store interval variables for each operation in a job.
    The keys are job names, stage names, and machine numbers.
    """

    def __init__(self, horizon: int):
        """Initialize the CustomCpModelWithInterval class.

        Args:
            horizon (int): The horizon for the scheduling problem,
                           which is the maximum time that any operation can start or end.
        """  # noqa: E501
        super().__init__()

        # Parameter
        self.horizon = horizon

        # Initialize dictionaries to store variables
        self.var_op_start = defaultdict(lambda: defaultdict(dict))
        self.var_op_end = defaultdict(lambda: defaultdict(dict))
        self.var_op_is_present = defaultdict(lambda: defaultdict(dict))
        self.var_op_intvl = defaultdict(lambda: defaultdict(dict))

    def define_optional_interval_var(
        self, job_idx: str, stage_idx: str, mc_idx: str, processing_time: int
    ):
        """Define an optional interval variable for a job operation.

        Args:
            job_idx (str): _description_
            stage_idx (str): _description_
            mc_idx (str): _description_
            processing_time (int): _description_
        """
        # method var_optional_casts on line 184 in constraint_program_model.py
        suffix = f"_{job_idx}_{stage_idx}_{mc_idx}"
        start_var = self.new_int_var(0, self.horizon, f"start{suffix}")
        end_var = self.new_int_var(0, self.horizon, f"end{suffix}")
        is_present_var = self.new_bool_var(f"is_present{suffix}")
        interval_var = self.new_optional_interval_var(
            start_var,
            processing_time,
            end_var,
            is_present_var,
            f"interval{suffix}",
        )
        self.var_op_start[job_idx][stage_idx][mc_idx] = start_var
        self.var_op_end[job_idx][stage_idx][mc_idx] = end_var
        self.var_op_is_present[job_idx][stage_idx][mc_idx] = is_present_var
        self.var_op_intvl[job_idx][stage_idx][mc_idx] = interval_var
