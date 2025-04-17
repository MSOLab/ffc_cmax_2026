from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from ortools.sat.cp_model_pb2 import ConstraintProto
from ortools.sat.python.cp_model import Constraint, CpModel, IntVar, LinearExpr


class CustomCpModel(CpModel):
    r"""A custom CpModel class that extends the ortools CpModel class."""

    added_constraints: list[Constraint]
    """List of added constraints."""
    idx_added_constraints: list[tuple[int, int]]
    """List of tuples representing the indices of the added constraints."""

    def __init__(self):
        super().__init__()
        self.added_constraints = []
        self.idx_added_constraints = []

    # variable functions

    def change_domain(self, var: IntVar, domain: list[int]) -> None:
        """Changes the domain of a variable.

        Args:
            var (IntVar)
            domain (list[int]): A list of two integers representing the new domain.
        """
        assert (
            len(domain) == 2
        ), f"Domain must be a list of two integers; {domain} given."

        var.Proto().domain[:] = domain

    # constraint functions

    def _get_constraints(self) -> RepeatedCompositeFieldContainer[ConstraintProto]:
        return self._CpModel__model.constraints

    def get_next_constr_idx(self) -> int:
        """Returns the index of the next constraint.

        Returns:
            int: The index of the next constraint.
        """
        return len(self._get_constraints())

    # add constraint functions

    def add_linear_constraint_fast(
        self, var_list: list[IntVar], coeff_list: list[int], domain: tuple[int, int]
    ):
        r"""Adds a linear constraint
        (domain[0] <= coeff_list \cdot var_list <= domain[1]).

        Args:
            var_list (list[IntVar])
            coeff_list (list[int])
            domain (tuple[int, int]): A tuple of two integers representing the domain of the linear expression.
        """  # noqa: E501
        assert len(var_list) == len(
            coeff_list
        ), f"Length of var_list and coeff_list must be the same; {len(var_list)} vars and {len(coeff_list)} coeffs given."

        ct = Constraint(self)
        model_ct = self._get_constraints()[ct.Index()]
        model_ct.linear.vars.extend([var.Index() for var in var_list])
        model_ct.linear.coeffs.extend(coeff_list)
        model_ct.linear.domain.extend(domain)

    def add_temporal_linear_constraints(
        self, args_list: list[tuple[list[IntVar], list[int], tuple[int, int]]]
    ) -> None:
        r"""Adds a set of temporal linear constraints
        (domain[0] <= coeff_list \cdot var_list <= domain[1]).

        Args:
            args (list[tuple[list[IntVar], list[int], tuple[int, int]]]): collections of tuples
            (list of vars, coefficients, domain of the linear expression)
        """  # noqa: E501
        start_idx = self.get_next_constr_idx()
        count = 0
        for count, (var_list, coeff_list, domain) in enumerate(args_list, 1):
            self.add_linear_constraint_fast(var_list, coeff_list, domain)
            # FIXME: 윗 줄이 return하는 constraint를 self.added_constraints에 저장해야 하나?
        if count > 0:
            self.idx_added_constraints.append((start_idx, start_idx + count))

    def add_temporal_abs_equality_constraints(
        self, args_list: list[tuple[IntVar, LinearExpr]]
    ) -> None:
        """Adds temporal AbsEquality constraints (target == |expr|)

        Args:
            args (list[tuple[IntVar, LinearExpr]]): collections of tuples
            (target, expr)
        """  # noqa: E501
        start_idx = self.get_next_constr_idx()
        count = 0
        for count, (target, expr) in enumerate(args_list, 1):
            self.add_abs_equality(target, expr)  # TODO: this may be slow
            # FIXME: 윗 줄이 return하는 constraint를 self.added_constraints에 저장해야 하나?
        if count > 0:
            self.idx_added_constraints.append((start_idx, start_idx + count))

    # delete constraint functions

    def delete_constraints(self, idx_start: int, idx_end: int) -> None:
        del self._get_constraints()[idx_start:idx_end]

    # FIXME: temporal과 added의 차이는?
    def delete_added_constraints(self):
        """Deletes all added constraints."""
        for constraint in self.added_constraints:
            constraint.Proto().Clear()
        self.added_constraints.clear()

        for idx_start, idx_end in reversed(self.idx_added_constraints):
            self.delete_constraints(idx_start, idx_end)
        self.idx_added_constraints.clear()
