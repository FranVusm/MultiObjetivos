from __future__ import annotations

import re
import warnings
from pathlib import Path

try:
    from .data_model import ProblemData
except ImportError:  # Allows running src/main.py directly.
    from data_model import ProblemData


class DataConsistencyWarning(UserWarning):
    """Backward-compatible warning category for older tests/imports."""


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
INDEX_RE = r"[-+]?\d+"
REQUIRED_SETS = ("N", "P", "K", "V")
REQUIRED_PARAMS = ("O", "q", "d", "LI", "LS", "sigma", "e", "ee", "g", "gg", "T", "tt")


def load_ampl_dat(path: str) -> ProblemData:
    dat_path = Path(path)
    if not dat_path.exists():
        raise FileNotFoundError(f"AMPL .dat file not found: {dat_path}")

    text = _strip_comments(dat_path.read_text(encoding="utf-8"))
    sets = _extract_declarations(text, "set")
    params = _extract_declarations(text, "param")

    _ensure_required(sets, REQUIRED_SETS, "set")
    _ensure_required(params, REQUIRED_PARAMS, "param")

    data = ProblemData(
        N=_parse_set(sets["N"]),
        P=_parse_set(sets["P"]),
        K=_parse_set(sets["K"]),
        V=_parse_set(sets["V"]),
        O=_parse_int_scalar(params["O"], "O"),
        q=_parse_float_scalar(params["q"], "q"),
        d=_parse_1d_param(params["d"], "d"),
        LI=_parse_1d_param(params["LI"], "LI"),
        LS=_parse_1d_param(params["LS"], "LS"),
        sigma=_parse_2d_table_param(params["sigma"], "sigma"),
        e=_parse_3d_matrix_param(params["e"], "e"),
        ee=_parse_2d_pairs_param(params["ee"], "ee"),
        g=_parse_3d_matrix_param(params["g"], "g"),
        gg=_parse_2d_pairs_param(params["gg"], "gg"),
        T=_parse_3d_matrix_param(params["T"], "T"),
        tt=_parse_2d_pairs_param(params["tt"], "tt"),
    )

    _validate_problem_data(data)
    return data


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _extract_declarations(text: str, keyword: str) -> dict[str, str]:
    pattern = re.compile(rf"\b{keyword}\s+([A-Za-z_]\w*)\b(.*?);", re.IGNORECASE | re.DOTALL)
    return {match.group(1): match.group(2).strip() for match in pattern.finditer(text)}


def _ensure_required(declarations: dict[str, str], required: tuple[str, ...], kind: str) -> None:
    missing = [name for name in required if name not in declarations]
    if missing:
        raise ValueError(f"Missing required AMPL {kind}(s): {', '.join(missing)}")


def _numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(text)


def _as_int(token: str, context: str) -> int:
    value = float(token)
    if not value.is_integer():
        raise ValueError(f"Expected integer index in {context}, got {token!r}")
    return int(value)


def _as_float(token: str) -> float:
    return float(token)


def _after_assignment(body: str, name: str) -> str:
    if ":=" not in body:
        raise ValueError(f"Parameter {name!r} must contain ':='")
    return body.split(":=", 1)[1]


def _parse_set(body: str) -> list[int]:
    values = [_as_int(token, "set") for token in _numbers(_after_assignment(body, "set"))]
    return values


def _parse_float_scalar(body: str, name: str) -> float:
    tokens = _numbers(_after_assignment(body, name))
    if not tokens:
        raise ValueError(f"Scalar parameter {name!r} has no numeric value")
    return _as_float(tokens[0])


def _parse_int_scalar(body: str, name: str) -> int:
    tokens = _numbers(_after_assignment(body, name))
    if not tokens:
        raise ValueError(f"Scalar parameter {name!r} has no numeric value")
    return _as_int(tokens[0], name)


def _parse_1d_param(body: str, name: str) -> dict[int, float]:
    tokens = _numbers(_after_assignment(body, name))
    if len(tokens) % 2 != 0:
        raise ValueError(f"Parameter {name!r} must contain key/value pairs")

    parsed: dict[int, float] = {}
    for index in range(0, len(tokens), 2):
        key = _as_int(tokens[index], name)
        parsed[key] = _as_float(tokens[index + 1])
    return parsed


def _parse_2d_table_param(body: str, name: str) -> dict[tuple[int, int], float]:
    match = re.search(r":\s*(.*?)\s*:=\s*(.*)\Z", body, re.DOTALL)
    if not match:
        raise ValueError(f"Parameter {name!r} must use a 2D table format with a header")

    columns = [_as_int(token, name) for token in _numbers(match.group(1))]
    if not columns:
        raise ValueError(f"Parameter {name!r} has an empty column header")

    parsed: dict[tuple[int, int], float] = {}
    for line in match.group(2).splitlines():
        tokens = _numbers(line)
        if not tokens:
            continue
        row = _as_int(tokens[0], name)
        values = [_as_float(token) for token in tokens[1:]]
        _warn_if_row_width_mismatch(name, row, columns, values)
        for column, value in zip(columns, values):
            parsed[(row, column)] = value
    return parsed


def _parse_3d_matrix_param(body: str, name: str) -> dict[tuple[int, int, int], float]:
    pattern = re.compile(
        rf"\[\s*\*\s*,\s*\*\s*,\s*(?P<period>{INDEX_RE})\s*\]\s*:\s*(?P<columns>.*?)\s*:=",
        re.DOTALL,
    )
    matches = list(pattern.finditer(body))
    if not matches:
        raise ValueError(f"Parameter {name!r} must contain blocks like [*,*,p]: columns :=")

    parsed: dict[tuple[int, int, int], float] = {}
    for index, match in enumerate(matches):
        period = _as_int(match.group("period"), name)
        columns = [_as_int(token, name) for token in _numbers(match.group("columns"))]
        if not columns:
            raise ValueError(f"Parameter {name!r}, period {period}, has an empty column header")

        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        block = body[match.end() : next_start]
        for line in block.splitlines():
            tokens = _numbers(line)
            if not tokens:
                continue
            row = _as_int(tokens[0], name)
            values = [_as_float(token) for token in tokens[1:]]
            _warn_if_row_width_mismatch(f"{name}[*,*,{period}]", row, columns, values)
            for column, value in zip(columns, values):
                parsed[(row, column, period)] = value
    return parsed


def _parse_2d_pairs_param(body: str, name: str) -> dict[tuple[int, int], float]:
    block_pattern = re.compile(
        rf"\[\s*\*\s*,\s*(?P<period>{INDEX_RE})\s*\]\s*(?::)?\s*:=",
        re.DOTALL,
    )
    matches = list(block_pattern.finditer(body))
    if matches:
        return _parse_2d_pair_blocks(body, name, matches)

    tokens = _numbers(_after_assignment(body, name))
    if len(tokens) % 3 != 0:
        raise ValueError(f"Parameter {name!r} must contain triples: node period value")

    parsed: dict[tuple[int, int], float] = {}
    for index in range(0, len(tokens), 3):
        node = _as_int(tokens[index], name)
        period = _as_int(tokens[index + 1], name)
        parsed[(node, period)] = _as_float(tokens[index + 2])
    return parsed


def _parse_2d_pair_blocks(
    body: str,
    name: str,
    matches: list[re.Match[str]],
) -> dict[tuple[int, int], float]:
    parsed: dict[tuple[int, int], float] = {}
    for index, match in enumerate(matches):
        period = _as_int(match.group("period"), name)
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        block = body[match.end() : next_start]
        tokens = _numbers(block)
        if len(tokens) % 2 != 0:
            raise ValueError(f"Parameter {name!r}, period {period}, must contain node/value pairs")

        for token_index in range(0, len(tokens), 2):
            node = _as_int(tokens[token_index], name)
            parsed[(node, period)] = _as_float(tokens[token_index + 1])
    return parsed


def _warn_if_row_width_mismatch(
    name: str,
    row: int,
    columns: list[int],
    values: list[float],
) -> None:
    if len(values) != len(columns):
        warnings.warn(
            f"Parameter {name!r}, row {row}, has {len(values)} values but {len(columns)} columns",
            DataConsistencyWarning,
            stacklevel=3,
        )


def _validate_problem_data(data: ProblemData) -> None:
    if data.q <= 0:
        raise ValueError(f"Vehicle capacity q must be positive, got {data.q}")
    for set_name in ("P", "K", "V"):
        if not getattr(data, set_name):
            raise ValueError(f"Set {set_name} must not be empty")

    if 20 not in data.N:
        raise ValueError("Dummy depot 20 must be present in set N")

    nodes_in_n = set(data.N)
    _require_nodes_in_n("d", set(data.d), nodes_in_n)
    _require_nodes_in_n("e", _nodes_from_3d(data.e), nodes_in_n)
    _require_nodes_in_n("g", _nodes_from_3d(data.g), nodes_in_n)
    _require_nodes_in_n("T", _nodes_from_3d(data.T), nodes_in_n)
    _require_nodes_in_n("ee", {node for node, _ in data.ee}, nodes_in_n)
    _require_nodes_in_n("gg", {node for node, _ in data.gg}, nodes_in_n)
    _require_nodes_in_n("tt", {node for node, _ in data.tt}, nodes_in_n)

    positive_demand_not_in_n = sorted(node for node, demand in data.d.items() if demand > 0 and node not in nodes_in_n)
    if positive_demand_not_in_n:
        raise ValueError(
            "Positive-demand customers not present in set N: "
            + ", ".join(str(node) for node in positive_demand_not_in_n)
        )

    periods = set(data.P)
    _require_periods("LI", set(data.LI), periods)
    _require_periods("LS", set(data.LS), periods)
    _require_periods("e", {period for _, _, period in data.e}, periods)
    _require_periods("g", {period for _, _, period in data.g}, periods)
    _require_periods("T", {period for _, _, period in data.T}, periods)
    _require_periods("ee", {period for _, period in data.ee}, periods)
    _require_periods("gg", {period for _, period in data.gg}, periods)
    _require_periods("tt", {period for _, period in data.tt}, periods)
    _require_period_bounds(data)
    _require_period_complete_3d("e", data.e, nodes_in_n, periods)
    _require_period_complete_3d("g", data.g, nodes_in_n, periods)
    _require_period_complete_3d("T", data.T, nodes_in_n, periods)
    _require_period_complete_2d("ee", data.ee, nodes_in_n, periods)
    _require_period_complete_2d("gg", data.gg, nodes_in_n, periods)
    _require_period_complete_2d("tt", data.tt, nodes_in_n, periods)

    total_demand = sum(demand for node, demand in data.d.items() if node in nodes_in_n and demand > 0)
    total_capacity = len(data.K) * len(data.V) * data.q
    oversized_customers = sorted(
        (node, demand)
        for node, demand in data.d.items()
        if node in nodes_in_n and demand > data.q
    )
    if oversized_customers:
        sample = ", ".join(f"{node}={demand}" for node, demand in oversized_customers[:5])
        raise ValueError(f"Customer demand exceeds vehicle capacity q={data.q}: {sample}")
    if total_demand > total_capacity:
        raise ValueError(
            "Total customer demand exceeds total trip capacity: "
            f"demand={total_demand}, capacity={total_capacity}. "
            "Increase K, V, or q; or reduce positive-demand customers."
        )


def _nodes_from_3d(values: dict[tuple[int, int, int], float]) -> set[int]:
    nodes: set[int] = set()
    for origin, destination, _ in values:
        nodes.add(origin)
        nodes.add(destination)
    return nodes


def _require_nodes_in_n(param_name: str, used_nodes: set[int], nodes_in_n: set[int]) -> None:
    extra_nodes = sorted(used_nodes - nodes_in_n)
    if extra_nodes:
        raise ValueError(
            f"Parameter {param_name!r} uses nodes not present in set N: "
            + ", ".join(str(node) for node in extra_nodes)
        )


def _require_periods(param_name: str, used_periods: set[int], periods: set[int]) -> None:
    missing = sorted(periods - used_periods)
    extra = sorted(used_periods - periods)
    messages = []
    if missing:
        messages.append("missing periods: " + ", ".join(str(period) for period in missing))
    if extra:
        messages.append("uses periods not present in set P: " + ", ".join(str(period) for period in extra))
    if messages:
        raise ValueError(f"Parameter {param_name!r} has inconsistent periods; " + "; ".join(messages))


def _require_period_bounds(data: ProblemData) -> None:
    last_upper = None
    for period in sorted(data.P):
        lower = data.LI[period]
        upper = data.LS[period]
        if lower > upper:
            raise ValueError(f"Period {period} has LI > LS: LI={lower}, LS={upper}")
        if last_upper is not None and lower < last_upper:
            raise ValueError(
                f"Period bounds overlap or go backwards at period {period}: "
                f"LI={lower}, previous LS={last_upper}"
            )
        last_upper = upper


def _require_period_complete_3d(
    param_name: str,
    values: dict[tuple[int, int, int], float],
    nodes: set[int],
    periods: set[int],
) -> None:
    missing = [
        (origin, destination, period)
        for period in sorted(periods)
        for origin in sorted(nodes)
        for destination in sorted(nodes)
        if (origin, destination, period) not in values
    ]
    if missing:
        sample = ", ".join(str(key) for key in missing[:5])
        raise ValueError(f"Parameter {param_name!r} is missing arc-period values; examples: {sample}")


def _require_period_complete_2d(
    param_name: str,
    values: dict[tuple[int, int], float],
    nodes: set[int],
    periods: set[int],
) -> None:
    missing = [
        (node, period)
        for period in sorted(periods)
        for node in sorted(nodes)
        if (node, period) not in values
    ]
    if missing:
        sample = ", ".join(str(key) for key in missing[:5])
        raise ValueError(f"Parameter {param_name!r} is missing node-period values; examples: {sample}")
