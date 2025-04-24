from typing import TextIO, Type, TypeVar

T = TypeVar("T", bound=int)


class TextDataParser:
    @staticmethod
    def strip_a_line(stream: TextIO) -> str:
        line = stream.readline()
        if not line:
            raise EOFError("Unexpected end of file while reading a line.")
        return line.strip()

    # Line as a value

    @staticmethod
    def strip_a_typed_value(stream: TextIO, dtype: Type[T]) -> T:
        try:
            return dtype(TextDataParser.strip_a_line(stream))
        except ValueError as e:
            raise ValueError(f"Failed to convert line to {dtype.__name__}: {e}") from e

    # Line as a list

    @staticmethod
    def strip_a_list(stream: TextIO, sep: str | None = None) -> list[str]:
        return TextDataParser.strip_a_line(stream).split(sep=sep)

    @staticmethod
    def strip_a_typed_list(
        stream: TextIO, dtype: Type[T], sep: str | None = None
    ) -> list[T]:
        return [dtype(x) for x in TextDataParser.strip_a_list(stream, sep=sep)]

    # Multiple lines as a list of lists

    @staticmethod
    def strip_list_of_a_typed_list(
        stream: TextIO, num_lists: int, dtype: Type[T], sep: str | None = None
    ) -> list[list[T]]:
        try:
            return [
                TextDataParser.strip_a_typed_list(stream, dtype, sep=sep)
                for _ in range(num_lists)
            ]
        except EOFError:
            raise EOFError(
                f"Unexpected end of file while reading {num_lists} lists "
                f"of type {dtype.__name__}."
            )
