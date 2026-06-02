from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SchemaValidationError(ValueError):
    pass


def validate_protocol_payload(payload: dict[str, Any]) -> None:
    validate_payload("gemmanima_protocol.schema.json", payload)


def validate_conditioning_bundle_payload(payload: dict[str, Any]) -> None:
    validate_payload("conditioning_bundle.schema.json", payload)


def validate_conflict_report_payload(payload: dict[str, Any]) -> None:
    validate_payload("conflict_report.schema.json", payload)


def validate_run_manifest_payload(payload: dict[str, Any]) -> None:
    validate_payload("run_manifest.schema.json", payload)


def validate_cache_build_manifest_payload(payload: dict[str, Any]) -> None:
    validate_payload("cache_build_manifest.schema.json", payload)


def validate_payload(schema_name: str, payload: dict[str, Any]) -> None:
    schema = _load_schema(schema_name)
    _validate_node(payload, schema, path=schema.get("title", schema_name))


def _validate_node(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if "$ref" in schema:
        schema = _load_schema(str(schema["$ref"]))

    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path} must be {schema['const']!r}, got {value!r}")

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path} must be one of {schema['enum']!r}, got {value!r}")

    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        raise SchemaValidationError(f"{path} must be {expected_type}, got {type(value).__name__}")

    if schema.get("type") == "object":
        _validate_object(value, schema, path=path)
    elif schema.get("type") == "array":
        _validate_array(value, schema, path=path)

    if "minimum" in schema and value < schema["minimum"]:
        raise SchemaValidationError(f"{path} must be >= {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        raise SchemaValidationError(f"{path} must be <= {schema['maximum']}")


def _validate_object(value: dict[str, Any], schema: dict[str, Any], *, path: str) -> None:
    required = schema.get("required", ())
    missing = [key for key in required if key not in value]
    if missing:
        raise SchemaValidationError(f"{path} missing required fields: {', '.join(missing)}")

    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        extra = sorted(set(value) - set(properties))
        if extra:
            raise SchemaValidationError(f"{path} has unexpected fields: {', '.join(extra)}")

    for key, child_schema in properties.items():
        if key in value:
            _validate_node(value[key], child_schema, path=f"{path}.{key}")


def _validate_array(value: list[Any], schema: dict[str, Any], *, path: str) -> None:
    prefix_items = schema.get("prefixItems")
    if prefix_items is not None:
        if len(value) != len(prefix_items) and schema.get("items") is False:
            raise SchemaValidationError(f"{path} must have {len(prefix_items)} items")
        for index, child_schema in enumerate(prefix_items):
            if index >= len(value):
                raise SchemaValidationError(f"{path} missing item {index}")
            _validate_node(value[index], child_schema, path=f"{path}[{index}]")
        return

    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            _validate_node(item, item_schema, path=f"{path}[{index}]")


def _matches_type(value: Any, expected_type: str | list[str]) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_type(value, item) for item in expected_type)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((_schema_dir() / name).read_text(encoding="utf-8"))


def _schema_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas"
