from typing import Any, Hashable, Mapping, Sequence

from ortools.sat.python import cp_model


def solve_selection_problem(
    jobs: Sequence[Hashable],
    r: Mapping[Hashable, int],
    t: Mapping[Hashable, int],
    K_L: int,
    K_R: int,
) -> dict[str, Any]:
    """
    Parameters:
    - jobs: list of job ids
    - r: dict {job_id: r_j value}
    - t: dict {job_id: t_j value}
    - K_L: number of jobs to select for L
    - K_R: number of jobs to select for R
    """
    model = cp_model.CpModel()

    # Variables
    x = {j: model.new_bool_var(f"x_{j}") for j in jobs}
    y = {j: model.new_bool_var(f"y_{j}") for j in jobs}

    # Constraints
    # 1. Each job can be in at most one set (L or R)
    for j in jobs:
        model.add(x[j] + y[j] <= 1)

    # 2. Exactly K_L jobs in L
    model.add(sum(x[j] for j in jobs) == K_L)

    # 3. Exactly K_R jobs in R
    model.add(sum(y[j] for j in jobs) == K_R)

    # Objective: minimize total cost
    primary_obj = sum(r[j] * x[j] + t[j] * y[j] for j in jobs)
    model.minimize(primary_obj)

    # First solve for optimal primary objective
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        return {"status": "INFEASIBLE"}

    optimal_cost = solver.objective_value
    optimal_cost_int = int(round(optimal_cost))

    # Add constraint: primary objective must equal optimal value
    model.add(primary_obj == optimal_cost_int)

    # Secondary objective: minimize sum of selected job indices (tie-breaking)
    j_idx: dict[Hashable, int] = {
        j: idx + 1 for idx, j in enumerate(jobs)
    }  # Map job id to index (1-based)
    secondary_obj = sum(j_idx[j] * (x[j] + y[j]) for j in jobs)
    model.minimize(secondary_obj)

    # Solve again with tie-breaking
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Extract solution
        L_set = [j for j in jobs if solver.value(x[j]) == 1]
        R_set = [j for j in jobs if solver.value(y[j]) == 1]
        total_cost = solver.objective_value

        return {
            "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
            "L_set": L_set,
            "R_set": R_set,
            "total_cost": total_cost,
            "solve_time": solver.wall_time,
        }
    else:
        return {"status": "INFEASIBLE"}


# Usage example
if __name__ == "__main__":
    jobs: list[Hashable] = [1, 2, 3, 4, 5, 6]
    r: Mapping[Hashable, int] = {1: 10, 2: 15, 3: 8, 4: 12, 5: 20, 6: 5}
    t: Mapping[Hashable, int] = {1: 7, 2: 9, 3: 11, 4: 6, 5: 8, 6: 14}
    K_L: int = 2
    K_R: int = 2

    result = solve_selection_problem(jobs, r, t, K_L, K_R)
    print(f"Status: {result['status']}")
    if result["status"] != "INFEASIBLE":
        print(f"L set: {result['L_set']}")
        print(f"R set: {result['R_set']}")
        print(f"Total cost: {result['total_cost']}")
        print(f"Solve time: {result['solve_time']:.4f}s")
