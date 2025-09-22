"""Utilities for inspecting average-dual mappings and coverage."""

from __future__ import annotations

import argparse
import importlib
from typing import Any, Callable, Hashable, Iterable

MappingCallable = Callable[[float, float, float, float], Hashable]


def _extract_expected_types(mapping: Any) -> set[Hashable] | None:
    """Return the ``expected_types`` set from ``mapping`` when available."""

    expected = getattr(mapping, "expected_types", None)
    if expected is None:
        return None
    if isinstance(expected, set):
        return expected
    if isinstance(expected, Iterable):
        return set(expected)
    raise TypeError("Mapping 'expected_types' attribute must be iterable when provided")


def _load_mapping(spec: str) -> tuple[MappingCallable, set[Hashable] | None]:
    """Resolve a ``module:callable`` specification to a mapping function."""

    if ":" not in spec:
        raise ValueError("Mapping spec must be in 'module:callable' format")
    module_name, attr_name = spec.rsplit(":", 1)
    if not module_name or not attr_name:
        raise ValueError("Mapping spec must include both module and callable name")
    module = importlib.import_module(module_name)
    try:
        mapping = getattr(module, attr_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Module '{module_name}' lacks attribute '{attr_name}'") from exc
    if not callable(mapping):
        raise TypeError(f"Resolved mapping '{module_name}:{attr_name}' is not callable")
    expected = _extract_expected_types(mapping)
    return mapping, expected


def _format_expected_types(expected: set[Hashable] | None, *, limit: int | None = None) -> str:
    if not expected:
        return "Mapping does not define expected types."
    count = len(expected)
    header = f"Mapping reports {count} expected types."
    ordered = sorted(expected, key=repr)
    if limit is None or count <= limit:
        details = "\n".join(repr(item) for item in ordered)
        return f"{header}\n{details}" if details else header
    details = "\n".join(repr(item) for item in ordered[:limit])
    return f"{header}\n(first {limit} shown)\n{details}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        required=True,
        help="module:callable path to an average-dual mapping",
    )
    parser.add_argument(
        "--show-types",
        action="store_true",
        help="print the expected types reported by the mapping (if available)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="when showing types, print at most this many entries",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    mapping, expected = _load_mapping(args.mapping)
    print(f"Loaded mapping '{args.mapping}' as {mapping!r}")
    if args.show_types:
        print(_format_expected_types(expected, limit=args.limit))
    elif expected is None:
        print("Mapping does not define expected types.")
    else:
        print(f"Mapping reports {len(expected)} expected types.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
