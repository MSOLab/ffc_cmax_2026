# TODO.md — Known Issues (from PR review of 20260731_pw_to_sw → main)

## Critical

- [ ] **`hfs_single_instance_runner.py:253-258`** — `finally: return` silently swallows all exceptions, including re-raised ones. Caller never sees the error; gets `post_run_process()` return instead. Fix: move `post_run_process()` call out of `finally`, or restructure to avoid `return` in `finally`.
- [ ] **`hfs_single_instance_runner.py:253`** — Bare `except:` catches `SystemExit` / `KeyboardInterrupt`. Combined with above, Ctrl+C is silently swallowed. Fix: change to `except Exception:`.
- [ ] **`controller_core.py:606`** — `raise e` destroys original traceback chain. Fix: change to bare `raise`.
- [ ] **`hfs_multi_instance_runner.py:58`** — `traceback.print_exc()` bypasses logging framework; output lost in production log files. Fix: replace with `logging.error(..., exc_info=True)`.

## High

- [ ] **`main.py:28`** — `MAIN_METADATA_FILENAME` is hardcoded to `"main_metadata_p1_improv_ablation_full.yaml"` (experiment-specific). Should be a CLI argument or default to a generic name.
- [ ] **`hfs_multi_instance_runner.py:60`** — Error path in `_run_sequential` calls `self.results.append(result)` directly (bypasses `append_result()`), so CSV won't reflect failures.
- [ ] **`hfs_cp_lns.py:3285`** — `assert surrogate_sw_cp_batch_size_ratio is not None` used for runtime validation; stripped with `-O`. Replace with explicit `if ... is None: raise ValueError(...)`.
- [ ] **`report/log_processor.py:46`** and **`report/method_progression_report.py:15`** — Duplicate `_find_progression_json_path` function. Import from `log_processor` in `method_progression_report`.
- [ ] **`.claude/skills/algorithm-doc-kr/SKILL.md:123`** and **`references/format.md:3,37`** — Stale `pw_cp_constructor_run.md` references. Update to `sw_cp_constructor_run.md`.
- [ ] **`AGENTS.md:109`** and **`CLAUDE.md:109`** — Stale commit prefix `fix(pw-cp)` / `refactor(pw-cp)`. Update to `sw-cp`.

## Low

- [ ] Magic numbers (`10**9`, `10`, `1000`, `20000`) scattered across `hfs_cp_lns.py` should be named constants.
- [ ] Legacy YAML configs in `configs_20s/`, `configs_100s/`, `configs_600s/` reference non-existent controller methods (`initialize_by_cjq1`, `apply_apsc_lb`, `time_window_search`, etc.). Either archive or annotate as legacy.
