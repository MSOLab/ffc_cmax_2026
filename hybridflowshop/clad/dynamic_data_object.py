import json
from pathlib import PurePath
from typing import Any


class DynamicDataObject(object):
    """
    A dynamic container for data that assigns attributes from a dictionary.

    This class enables the dynamic creation of attributes based on a provided dictionary.
    In addition, the class supports wrapping nested dictionaries and lists into
    DynamicDataObject instances via the from_obj() class method.

    The constructor expects a dictionary (with valid identifier keys) and transforms
    each key/value pair into an attribute. For data structured as lists or nested dictionaries,
    use the from_obj() method to recursively convert them.

    Features:
    - Dynamic attribute assignment from a dictionary.
    - Supports conversion of its attributes into plain dictionaries and lists.
    - Serialization to and deserialization from JSON files.
    - Recursive transformation of nested dictionaries and lists via from_obj().

    Usage:
    >>> data = DynamicDataObject({'name': 'John', 'age': 30, 'skills': ['Python', 'JSON']})
    >>> print(data.name)
    John
    >>> data.to_json_file('person.json')
    >>> loaded_data = DynamicDataObject.from_json_file('person.json')

    Note:
    This class is particularly useful for working with data from APIs or other sources where
    the structure might vary or evolve over time. For list-based data, the from_obj() method can be used.
    """  # noqa: E501

    def __init__(self, param_dict: dict[str, Any]):
        for key, value in param_dict.items():
            # Validate that key is a valid identifier
            if not isinstance(key, str) or not key.isidentifier():
                raise ValueError(
                    f"Invalid key: {key}. Keys must be valid string identifiers."
                )
            # Prevent conflicts with existing class attributes/methods
            if key in self.__class__.__dict__:
                raise ValueError(
                    f"Key '{key}' is reserved and cannot be used as an attribute."
                )
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.to_obj())})"

    @classmethod
    def from_obj(cls, obj: Any) -> Any:
        """Recursively converts dictionaries and lists into DynamicDataObject instances.

        Args:
            obj (Any): dictionary or list or any other object

        Returns:
            Any: a class instance
        """
        if isinstance(obj, dict):
            return cls(
                {key: DynamicDataObject.from_obj(value) for key, value in obj.items()}
            )
        if isinstance(obj, list):
            return [DynamicDataObject.from_obj(item) for item in obj]
        return obj

    def to_obj(self) -> Any:
        """Recursively converts the DynamicDataObject and any nested DynamicDataObject
        instances back into plain dictionaries and lists.

        Returns:
            Any: a plain object that can be serialized directly to JSON.
        """

        def _convert(value: Any) -> Any:
            if isinstance(value, DynamicDataObject):
                return value.to_obj()
            elif isinstance(value, list):
                return [_convert(item) for item in value]
            elif isinstance(value, dict):
                return {key: _convert(val) for key, val in value.items()}
            return value

        return _convert(self.__dict__)

    @classmethod
    def from_json(cls, file_path: PurePath, encoding="utf-8") -> Any:
        """Deserializes a JSON file into a DynamicDataObject instance.

        Args:
            file_path (PurePath)

        Raises:
            RuntimeError: If an error occurs while reading the file.
            ValueError: If an error occurs while parsing the JSON data.

        Returns:
            Any: a class instance created from the JSON data in the file.
        """
        try:
            with open(file_path, "r", encoding=encoding) as file:
                json_data = json.load(file)
            return cls.from_obj(json_data)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error reading from file {file_path}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON from file {file_path}: {e}")

    def to_json(self, file_path: PurePath) -> None:
        """Serializes the object's data to a JSON file at the specified file path.

        Args:
            file_path (PurePath)
        """
        try:
            with open(file_path, "w", encoding="utf-8") as writer:
                json.dump(self.to_obj(), writer, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error writing to file {file_path}: {e}")


def main():
    from pprint import pprint

    # Create an DynamicDataObject instance
    data = DynamicDataObject.from_obj(
        {
            "name": "John Doe",
            "age": 30,
            "job": {"title": "Software Engineer", "company": "Tech Corp"},
            "skills": ["Python", "JSON", "API Design"],
        }
    )
    pprint(data)


if __name__ == "__main__":
    main()
