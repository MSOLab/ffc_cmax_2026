from pathlib import Path
from typing import Self, TextIO, Type, TypeVar, cast

import pandas as pd

from ...util.text_data_parser import TextDataParser

T = TypeVar("T", bound=int)


class DfManager:
    def __init__(self, name: str, df: pd.DataFrame) -> None:
        self.name = name
        self.df = df

    def __repr__(self) -> str:
        return f"DfManager(name='{self.name}', shape={self.df.shape})"

    @classmethod
    def from_text_stream(
        cls,
        stream: TextIO,
        row_count: int,
        dtype: Type[T] | None = None,
        sep: str | None = None,
        name: str = "DfManager",
    ) -> Self:
        """Create a DfManager instance from a text stream.

        Args:
            stream (TextIO): Input text stream.
            row_count (int): Number of rows to read from the stream.
            dtype (Type[T], optional): Data type to cast the values to. Defaults to int.
            sep (str | None, optional): Column separator. Defaults to any whitespace.
            name (str): The name of the table. Defaults to "DfManager".

        Returns:
            DfManager: instance with parsed table.
        """
        actual_dtype = dtype or int

        rows: list[list[T]] = TextDataParser.strip_list_of_a_typed_list(
            stream, row_count, cast(Type[T], actual_dtype), sep=sep
        )
        df = pd.DataFrame(rows)
        return cls(name, df)

    def row_count(self) -> int:
        return len(self.df)

    def col_count(self) -> int:
        return len(self.df.columns)

    def to_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        # print(f"[{self.name}] DataFrame exported to {path}")
