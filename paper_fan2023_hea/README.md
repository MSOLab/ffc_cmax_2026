# Fan 2023 HEA Track

This folder is a standalone implementation track for **A Hybrid Evolutionary
Algorithm Using Two Solution Representations for Hybrid Flow-Shop Scheduling
Problem**.  It intentionally does not plug into `main.py` or the existing
scenario-flow runner.

Run it with:

```bash
uv run python -m paper_fan2023_hea --config paper_fan2023_hea/configs/smoke_ff2020big.yaml
```

Outputs are created under `Outputs_paper_fan2023/<timestamp>/`:

- `seed_runs.csv`
- `all_scenarios_summary.csv`
- `baseline_comparison.csv`
- optional `convergence_<instance>.png`

The summary keeps both `name` and `insName`, plus `scenario` and `bestObj`, so
it can be read by the repo's `exp_compare` utilities.

## Paper Constants

Table II was visually checked from the provided PDF, not copied from OCR:

- `popSize = 50`
- `evoRep = 15`
- `tSize = 2`
- `noImprove = 5`
- `Pr = 0.04`
- `maxIterTS = 100*n`
- `lenTS = n`
- total termination time: `2*n*z` seconds
- tabu search activation time: `n*z` seconds

## Scope Notes

The solver uses the paper's two-part representation `(permutation,
decoding_flag)` with immutable `forward`/`backward` flags.  Forward decoding is
ASAP + first available machine.  Backward decoding is implemented as reversed
stage-order ASAP followed by reversing the time axis, which realizes the paper's
ALAP + latest available machine rule with the repository's schedule primitive.

The tabu-search control flow follows the paper's disjunctive-graph critical
block neighborhoods, N7-style forward moves, Theorem 1/2 backward moves,
k-insertion filters, tabu tenure, and global-best-only aspiration.  The PDF does
not fully disclose the fast closed-form neighbor evaluator, so this
implementation intentionally uses exact semi-active schedule recomputation for
neighbor evaluation.
