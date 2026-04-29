"""Config flattening helper for ml_logger hparams.

Walks `@dataclass` / pydantic `BaseModel` trees and produces a flat
`dict[str, primitive]` suitable for `wandb.config`. Non-config-shaped
values (callables, runtime objects, etc.) are silently skipped.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel

_Primitive = (str, int, float, bool)


def flatten_config(obj: Any, prefix: str = "", sep: str = "/") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in _iter_fields(obj):
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if value is None or isinstance(value, _Primitive):
            out[full_key] = value
        elif isinstance(value, Path):
            out[full_key] = str(value)
        elif is_dataclass(value) or isinstance(value, BaseModel):
            out.update(flatten_config(value, prefix=full_key, sep=sep))
        # else: skip (AgentsSDKModel, callables, runtime handles, ...)
    return out


def _iter_fields(obj: Any) -> Iterator[tuple[str, Any]]:
    if is_dataclass(obj):
        for f in fields(obj):
            yield f.name, getattr(obj, f.name)
    elif isinstance(obj, BaseModel):
        for k in type(obj).model_fields:
            yield k, getattr(obj, k)
