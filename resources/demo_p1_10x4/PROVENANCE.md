# PROVENANCE

FF instance 1's first 10 jobs x first 4 stages, machines forced to 2/stage,
p_ij original retained (uniform[1,99]). Explanation-only reduced instance,
UNRELATED to thesis experiment results (data/ffc_cmax/references.md). seed=42.

Source: resources/ff2020big/1.txt (40 jobs x 5 stages, machines "3 3 2 3 3").
Slice: first 10 job columns x first 4 stage rows.
Machine line forced to: "2 2 2 2".
Processing times are taken verbatim from the source; only the job/stage subset
and per-stage machine count were changed.

Regenerate with:
    uv run python scripts/p1_algo_explainer/make_demo_instance.py
