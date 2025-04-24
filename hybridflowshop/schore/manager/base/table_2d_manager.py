from typing import Self, TextIO, Type, TypeVar

import pandas as pd

from .df_manager import DfManager

T = TypeVar("T", bound=int)


class Table2DManager(DfManager):
    def __init__(self, name: str, df: pd.DataFrame) -> None:
        super().__init__(name, df)

    def __repr__(self) -> str:
        return f"Table2DManager(name='{self.name}', shape={self.df.shape})"

    @classmethod
    def from_text_stream(
        cls,
        stream: TextIO,
        row_count: int,
        dtype: Type[T] | None = None,
        sep: str | None = None,
        name: str = "Table2DManager",
    ) -> Self:
        return super().from_text_stream(stream, row_count, dtype, sep, name)

    def col_idx_2_row_idx_2_value_map(self) -> dict[int, dict[int, T]]:
        """
        Create a mapping from column index to row index and value.

        Returns:
            dict: Column index -> row index -> value.
        """
        return {
            col_idx: {
                row_idx: value for row_idx, value in self.df.iloc[:, col_idx].items()
            }
            for col_idx in range(self.col_count())
        }
