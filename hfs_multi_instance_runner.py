from pathlib import Path
from typing import Any, Sequence

from routix.runner import MultiInstanceConcurrentRunner
from schore.parameters_examples.parallel_shop.identical_flow import (
    HybridFlowshopParameters,
)

from hfs_single_instance_runner import HfsSingleInstanceRunner


class HfsMultiInstanceRunner(
    MultiInstanceConcurrentRunner[HybridFlowshopParameters, HfsSingleInstanceRunner]
):
    """
    Orchestrates solving a set of Hybrid Flow Shop (HFS) instances with a given runner class.
    Inherits from MultiInstanceConcurrentRunner.
    """

    def __init__(
        self,
        s_i_runner_class: type[HfsSingleInstanceRunner],
        instances: Sequence[HybridFlowshopParameters],
        shared_params: dict,
        subroutine_flow: Any,
        stopping_criteria: Any,
        output_dir: Path,
        output_metadata: dict[str, Any],
    ):
        super().__init__(
            s_i_runner_class,
            instances,
            shared_params,
            subroutine_flow,
            stopping_criteria,
            output_dir,
            output_metadata,
        )

    def post_run_process(self):
        self.generate_instance_set_table()

    def generate_instance_set_table(self) -> None:
        """
        Generates a table of the instance set.
        This method is specific to Hybrid Flow Shop instances.
        """

        def quote_brace_field(field: str) -> str:
            field = field.strip()
            if field.startswith("{") and field.endswith("}"):
                if not (field.startswith('"') and field.endswith('"')):
                    return f'"{field}"'
            return field

        def smart_split(line: str) -> list[str]:
            result: list[str] = []
            buf = ""
            depth = 0
            for c in line:
                if c == "," and depth == 0:
                    result.append(quote_brace_field(buf))
                    buf = ""
                else:
                    buf += c
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
            result.append(quote_brace_field(buf))
            return result

        result_dir_name = str(self.output_metadata.get("result_dir_name", "results"))
        header: list[str] = []
        rows: list[list[str]] = []

        for instance in self.instances:
            ins_name = instance.name
            summary_filename = f"{ins_name}_summary.csv"
            if "summary_filename_format" in self.output_metadata:
                summary_filename_format = self.output_metadata[
                    "summary_filename_format"
                ]
                if isinstance(summary_filename_format, str):
                    summary_filename_format = summary_filename_format.strip()
                    summary_filename = summary_filename_format.format(ins_name)

            summary_file_path = (
                self.output_dir / ins_name / result_dir_name / summary_filename
            )
            if not summary_file_path.exists():
                continue
            with open(summary_file_path, encoding="utf-8") as f:
                lines = f.readlines()
                file_header = lines[0].strip()
                if not header:
                    header = smart_split(file_header)
                data_line = lines[1].strip()
                row = smart_split(data_line)
                rows.append(row)

        # Prepare the output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "merged.csv"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                f.write(",".join(row) + "\n")
