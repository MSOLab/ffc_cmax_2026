"""Build the explanation-only reduced demo instance (plan §1).

Slices FF instance 1 (``resources/ff2020big/1.txt``, 40 jobs x 5 stages,
machines ``3 3 2 3 3``) down to the first 10 jobs x first 4 stages and forces
2 machines per stage. Processing times are kept EXACTLY as in the original
(no modification). The result is written in the same FF 2020 text format.

Idempotent: re-running overwrites the outputs with identical content.
"""

from __future__ import annotations

from pathlib import Path

SEED = 42  # fixed for reproducibility (documented in PROVENANCE)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "resources" / "ff2020big" / "1.txt"
OUT_DIR = REPO_ROOT / "resources" / "demo_p1_10x4"
OUT_INSTANCE = OUT_DIR / "1.txt"
OUT_PROVENANCE = OUT_DIR / "PROVENANCE.md"

N_JOBS = 10
N_STAGES = 4
MACHINES_PER_STAGE = 2

PROVENANCE_TEXT = """\
# PROVENANCE

FF instance 1's first 10 jobs x first 4 stages, machines forced to 2/stage,
p_ij original retained (uniform[1,99]). Explanation-only reduced instance,
UNRELATED to thesis experiment results (data/ffc_cmax/references.md). seed=42.

Source: resources/ff2020big/1.txt (40 jobs x 5 stages, machines "3 3 2 3 3").
Slice: first {n_jobs} job columns x first {n_stages} stage rows.
Machine line forced to: "{mc_line}".
Processing times are taken verbatim from the source; only the job/stage subset
and per-stage machine count were changed.

Regenerate with:
    uv run python scripts/p1_algo_explainer/make_demo_instance.py
""".format(
    n_jobs=N_JOBS,
    n_stages=N_STAGES,
    mc_line=" ".join([str(MACHINES_PER_STAGE)] * N_STAGES),
)


def build_demo_lines() -> list[str]:
    """Read the source FF file and return the sliced FF-format lines."""
    raw_lines = SOURCE_PATH.read_text().splitlines()

    header = raw_lines[0].split()
    n_orig, c_orig = int(header[0]), int(header[1])
    if n_orig < N_JOBS or c_orig < N_STAGES:
        raise ValueError(
            f"Source instance too small: {n_orig} jobs x {c_orig} stages, "
            f"need at least {N_JOBS} x {N_STAGES}"
        )

    # raw_lines[1] is the original machine line; we discard it and force ours.
    stage_rows = raw_lines[2 : 2 + N_STAGES]

    out_lines = [
        f"{N_JOBS} {N_STAGES}",
        " ".join([str(MACHINES_PER_STAGE)] * N_STAGES),
    ]
    for row in stage_rows:
        p_values = row.split()[:N_JOBS]
        if len(p_values) != N_JOBS:
            raise ValueError(f"Stage row has fewer than {N_JOBS} values: {row!r}")
        out_lines.append(" ".join(p_values))
    return out_lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_lines = build_demo_lines()
    OUT_INSTANCE.write_text("\n".join(out_lines) + "\n")
    OUT_PROVENANCE.write_text(PROVENANCE_TEXT)
    print(f"Wrote {OUT_INSTANCE}")
    print(f"Wrote {OUT_PROVENANCE}")
    print("Instance contents:")
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
