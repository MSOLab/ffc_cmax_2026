# Multi-Scenario Subroutine Flow Comparison HTML

## Summary
Add a top-level HTML report beside `multi_scenario_report.xlsx` that compares
subroutine flows across scenarios in one overlay chart.

The chart will:
- read each scenario's existing `summary_method_rpdf_and_norm_time_long.csv`
- aggregate per scenario and subroutine as `mean(norm_time)` vs `mean(rpd_f)`
- plot one trace per scenario/subroutine-flow on a single interactive screen
- use the scenario's `output_subdir` name as the flow label
- force both axes to start at 0, matching the existing SVG chart behavior
- be generated only for the full-run report level, because equivalent method-level
  timepoint inputs do not currently exist

## Key Changes
- Add a new multi-scenario HTML builder in `hfs_multi_scenario_runner.py` or a
  small helper module under `hybridflowshop/report` dedicated to top-level
  scenario comparison.
- Reuse per-scenario inputs already produced by
  `HfsMultiInstanceRunner._create_rpd_summary()`:
  - `scenario_dir / "summary_method_rpdf_and_norm_time_long.csv"`
- For each scenario:
  - load the long CSV
  - drop rows missing `subroutine_name`, `norm_time`, or `rpd_f`
  - aggregate by `subroutine_name` using mean `norm_time` and mean `rpd_f`
  - preserve first appearance order of subroutines within that scenario
  - attach scenario metadata using `scenario_configs[i]["output_subdir"]`
- Generate one HTML artifact at the same level as `multi_scenario_report.xlsx`:
  - `multi_scenario_subroutine_flow_comparison.html`
- Render as a single overlay chart:
  - one trace per scenario
  - markers connected in subroutine order
  - hover shows scenario, subroutine name, mean norm time, mean RPDf
  - x/y axes shown as percent
  - x-axis and y-axis both use zero-origin ranges
  - labels/legend use scenario `output_subdir` basename
- Keep it best-effort:
  - skip scenarios whose method summary file is missing
  - if no scenario has valid aggregated data, do not create the HTML and log a warning
- Call this generation from `HfsMultiScenarioRunner.post_run_process()`
  immediately after writing `multi_scenario_report.xlsx`
- Do not add timepoint variants now:
  - no `multi_scenario_subroutine_flow_comparison_<label>.html`
  - no changes to timepoint report flow until method-level timepoint data exists

## Test Plan
- Add focused tests for the new multi-scenario HTML generation path:
  - when two or more scenario directories contain valid
    `summary_method_rpdf_and_norm_time_long.csv`, one top-level HTML file is created
  - the HTML contains both scenario labels and expected subroutine labels
  - rows with missing `norm_time` or `rpd_f` are excluded before aggregation
  - missing scenario summary files are skipped without failing the whole report generation
  - when all scenario inputs are missing or invalid, no HTML file is written
  - the rendered chart config sets both x-axis and y-axis to zero-origin behavior
- Add one integration test around `HfsMultiScenarioRunner.post_run_process()`
  verifying:
  - `multi_scenario_report.xlsx` is still created as before
  - the new comparison HTML is created alongside it when scenario method summaries exist

## Assumptions
- "subroutine flow별" comparison means comparing scenario outputs, where each
  scenario corresponds to one subroutine flow configuration.
- The visible scenario identifier should be the `output_subdir` basename rather
  than the full path.
- Full-run only is the correct scope for now because there is no scenario-level
  method RPD timepoint CSV to drive equivalent timepoint HTML.
- A single overlay chart is preferred over small multiples for direct
  cross-flow comparison on one screen.
