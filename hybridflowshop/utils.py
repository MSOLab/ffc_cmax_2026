import re


def tuple_to_pyyaml_key(d: dict) -> dict:
    """
    tuple 형태의 key를 '!!python/tuple [j0,i0,i0_1]' 형태로 변환

    reference: solution_manager.py row 66
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(k, tuple):
            # 내부 요소를 쉼표로 연결하고 strip
            items = ", ".join(str(item).strip() for item in k)
            new_dict[f"!!python/tuple [{items}]"] = v
        else:
            new_dict[k] = v
    return new_dict


def pyyaml_key_to_tuple(d: dict) -> dict:
    """
    '!!python/tuple [j0,i0,i0_1]' 형태의 key를 tuple로 변환

    reference: solution_manager.py row 66
    """
    tuple_key_pattern = re.compile(r"^!!python/tuple \[(.*)\]$")
    new_dict = {}
    for k, v in d.items():
        m = tuple_key_pattern.match(k)
        if m:
            # 내부 요소를 쉼표로 분리하고 strip
            items = [item.strip() for item in m.group(1).split(",")]
            new_dict[tuple(items)] = v
        else:
            new_dict[k] = v
    return new_dict
