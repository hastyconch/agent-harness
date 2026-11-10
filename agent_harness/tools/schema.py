"""A small JSON Schema validator covering what tool schemas actually use.

Supported keywords: type (incl. lists), properties, required,
additionalProperties (bool), enum, minimum/maximum, minLength/maxLength,
pattern, items, minItems/maxItems, default.

``coerce=True`` repairs the common ways models get types wrong ("5" for 5,
"true" for true) and fills defaults, returning the repaired value.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _coerce(value: Any, type_name: str) -> Tuple[bool, Any]:
    if not isinstance(value, str):
        return False, value
    s = value.strip()
    try:
        if type_name == "integer":
            return True, int(s)
        if type_name == "number":
            return True, float(s)
    except ValueError:
        return False, value
    if type_name == "boolean" and s.lower() in ("true", "false"):
        return True, s.lower() == "true"
    return False, value


def validate(value: Any, schema: Dict[str, Any], coerce: bool = True) -> Tuple[Any, List[str]]:
    """Return ``(possibly-coerced value, errors)``; no errors means valid."""
    errors: List[str] = []
    value = _validate(value, schema, "$", errors, coerce)
    return value, errors


def _validate(value: Any, schema: Dict[str, Any], path: str, errors: List[str], coerce: bool) -> Any:
    types = schema.get("type")
    if types is not None:
        type_list = types if isinstance(types, list) else [types]
        if not any(_TYPE_CHECKS[t](value) for t in type_list):
            fixed = False
            if coerce:
                for t in type_list:
                    fixed, value = _coerce(value, t)
                    if fixed:
                        break
            if not fixed:
                errors.append("%s: expected %s, got %s" % (path, " or ".join(type_list), type(value).__name__))
                return value

    if "enum" in schema and value not in schema["enum"]:
        errors.append("%s: must be one of %s" % (path, schema["enum"]))

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append("%s: shorter than %d characters" % (path, schema["minLength"]))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append("%s: longer than %d characters" % (path, schema["maxLength"]))
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append("%s: does not match pattern %r" % (path, schema["pattern"]))

    if _TYPE_CHECKS["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append("%s: must be >= %s" % (path, schema["minimum"]))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append("%s: must be <= %s" % (path, schema["maximum"]))

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append("%s: needs at least %d items" % (path, schema["minItems"]))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append("%s: allows at most %d items" % (path, schema["maxItems"]))
        if "items" in schema:
            value = [_validate(v, schema["items"], "%s[%d]" % (path, i), errors, coerce) for i, v in enumerate(value)]

    if isinstance(value, dict):
        props = schema.get("properties", {})
        value = dict(value)
        for name in schema.get("required", []):
            if name not in value:
                errors.append("%s: missing required property %r" % (path, name))
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    errors.append("%s: unexpected property %r (allowed: %s)" % (path, name, sorted(props)))
        for name, sub in props.items():
            if name in value:
                value[name] = _validate(value[name], sub, "%s.%s" % (path, name), errors, coerce)
            elif coerce and "default" in sub:
                value[name] = sub["default"]
    return value
