from ortools.sat.sat_parameters_pb2 import SatParameters

parameters = SatParameters()

print("log_search_progress: ", parameters.log_search_progress)  # default is False
print("max_time_in_seconds: ", parameters.max_time_in_seconds)  # default is inf
print("num_workers: ", parameters.num_workers)  # default is 0
print(
    "keep_all_feasible_solutions_in_presolve: ",
    parameters.keep_all_feasible_solutions_in_presolve,
)  # default is False
print("random_seed: ", parameters.random_seed)  # default is 1
print(
    "encode_cumulative_as_reservoir: ", parameters.encode_cumulative_as_reservoir
)  # default is False
print(
    "expand_reservoir_constraints: ", parameters.expand_reservoir_constraints
)  # default is True
print(
    "expand_reservoir_using_circuit: ", parameters.expand_reservoir_using_circuit
)  # default is False
print("interleave_search: ", parameters.interleave_search)  # default is False
print("use_lns_only: ", parameters.use_lns_only)  # default is False
print(
    "search_branching: ", parameters.search_branching
)  # default is 0(AUTOMATIC_SEARCH)
print("cp_model_probing_level: ", parameters.cp_model_probing_level)  # default is 2
