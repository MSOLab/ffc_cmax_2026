from ortools.sat.sat_parameters_pb2 import SatParameters

parameters = SatParameters()

print("log_search_progress: ", parameters.log_search_progress)
print("max_time_in_seconds: ", parameters.max_time_in_seconds)
print("num_workers: ", parameters.num_workers)
print("keep_all_feasible_solutions_in_presolve: ", parameters.keep_all_feasible_solutions_in_presolve)
print("random_seed: ", parameters.random_seed)
print("encode_cumulative_as_reservoir: ", parameters.encode_cumulative_as_reservoir)
print("expand_reservoir_constraints: ", parameters.expand_reservoir_constraints)
print("expand_reservoir_using_circuit: ", parameters.expand_reservoir_using_circuit)
print("interleave_search: ", parameters.interleave_search)
print("cp_model_probing_level: ", parameters.cp_model_probing_level)
