"""Load form schemas by name and validate fields against them.

Nothing here knows about FNOL: fields, labels, follow-up questions, validators
and conditional requirements all come from the JSON Schema file.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from app.agent.validators import FORMAT_CHECKER, FORMAT_HINTS, SEMANTIC_VALIDATORS, ValidationContext, normalize_value
from app.config import get_settings

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_REQUIRED_MSG = re.compile(r"^'(.+?)' is a required property")
_MISSING = object()


class SchemaNotFound(LookupError):
    pass


@dataclass
class FieldSpec:
    path: str
    schema: dict[str, Any]
    required: bool
    labels: dict[str, str] = field(default_factory=dict)
    questions: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        t = self.schema.get("type", "string")
        if isinstance(t, list):
            t = next((x for x in t if x != "null"), "string")
        return t

    def label(self, lang: str) -> str:
        return self.labels.get(lang) or self.labels.get("en") or self.path

    def question(self, lang: str) -> str:
        return self.questions.get(lang) or self.questions.get("en") or f"Could you tell me the {self.label(lang)}?"


@dataclass
class FieldValidation:
    valid: bool
    value: Any
    errors: list[str]

    def as_dict(self, original: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"valid": self.valid, "errors": self.errors}
        if self.value != original:
            out["normalized_value"] = self.value
        return out


class FormSchema:
    def __init__(self, name: str, raw: dict[str, Any]):
        self.name = name
        self.raw = raw
        self.title: str = raw.get("title", name)
        self.pii_allow: list[str] = list(raw.get("x-pii-allow", []))
        self.fields: list[FieldSpec] = []
        self._flatten(raw, prefix="", parent_required=True)
        self.by_path = {f.path: f for f in self.fields}
        self._validator = Draft202012Validator(raw, format_checker=FORMAT_CHECKER)

    # --------------------------------------------------------------- structure
    def _flatten(self, node: dict[str, Any], prefix: str, parent_required: bool) -> None:
        required = set(node.get("required", []))
        for key, sub in node.get("properties", {}).items():
            path = f"{prefix}{key}"
            is_required = parent_required and key in required
            if sub.get("type") == "object" and "properties" in sub:
                self._flatten(sub, prefix=f"{path}.", parent_required=is_required)
                continue
            self.fields.append(
                FieldSpec(
                    path=path,
                    schema=sub,
                    required=is_required,
                    labels=dict(sub.get("x-label", {})),
                    questions=dict(sub.get("x-question", {})),
                )
            )

    def field(self, path: str) -> FieldSpec:
        try:
            return self.by_path[path]
        except KeyError:
            raise KeyError(f"Unknown field_path {path!r}. Valid paths: {', '.join(self.by_path)}") from None

    @staticmethod
    def get_value(payload: dict[str, Any], path: str, default: Any = None) -> Any:
        node: Any = payload
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @staticmethod
    def set_value(payload: dict[str, Any], path: str, value: Any) -> None:
        parts = path.split(".")
        node = payload
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    @staticmethod
    def delete_value(payload: dict[str, Any], path: str) -> None:
        parts = path.split(".")
        node: Any = payload
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                return
        if isinstance(node, dict):
            node.pop(parts[-1], None)

    # --------------------------------------------------------------- validation
    def validate_field(self, path: str, value: Any, ctx: ValidationContext) -> FieldValidation:
        spec = self.field(path)
        normalized = normalize_value(copy.deepcopy(value), spec.schema)
        errors = []
        for e in Draft202012Validator(spec.schema, format_checker=FORMAT_CHECKER).iter_errors(normalized):
            where = "/".join([path, *map(str, e.absolute_path)])
            hint = FORMAT_HINTS.get(e.validator_value) if e.validator == "format" else None
            errors.append(f"{where}: {e.message}" + (f" ({hint})" if hint else ""))
        for name in spec.schema.get("x-validators", []):
            errors += SEMANTIC_VALIDATORS[name](normalized, ctx)
        return FieldValidation(valid=not errors, value=normalized, errors=errors)

    def clean_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Keep only known fields with non-null values, normalized."""
        out: dict[str, Any] = {}
        for spec in self.fields:
            value = self.get_value(payload, spec.path, _MISSING)
            if value is _MISSING or value is None or value == "":
                continue
            self.set_value(out, spec.path, normalize_value(copy.deepcopy(value), spec.schema))
        return out

    def missing_required(self, payload: dict[str, Any]) -> list[str]:
        """Required paths (including conditional if/then requirements) absent from payload."""
        missing: list[str] = []
        for err in self._validator.iter_errors(payload):
            for e in [err, *(err.context or [])]:
                if e.validator != "required":
                    continue
                m = _REQUIRED_MSG.match(e.message)
                if not m:
                    continue
                prefix = ".".join(str(p) for p in e.absolute_path)
                path = f"{prefix}.{m.group(1)}" if prefix else m.group(1)
                if path in self.by_path and path not in missing:
                    missing.append(path)
        # Nested objects entirely absent: report their required leaves.
        for spec in self.fields:
            if spec.required and self.get_value(payload, spec.path, _MISSING) is _MISSING and spec.path not in missing:
                missing.append(spec.path)
        order = {f.path: i for i, f in enumerate(self.fields)}
        return sorted(missing, key=lambda p: order.get(p, 1_000))

    # --------------------------------------------------------------- prompting
    def describe_fields(self) -> list[dict[str, Any]]:
        out = []
        for spec in self.fields:
            s = spec.schema
            item: dict[str, Any] = {"field_path": spec.path, "type": spec.type, "required": spec.required}
            for key in ("description", "format", "pattern", "enum"):
                if key in s:
                    item[key] = s[key]
            if spec.type == "array" and isinstance(s.get("items"), dict):
                item["items"] = {k: v for k, v in s["items"].items() if not k.startswith("x-")}
            out.append(item)
        return out

    def conditional_rules(self) -> list[str]:
        rules = []
        for clause in self.raw.get("allOf", []):
            cond = clause.get("if", {}).get("properties", {})
            then = clause.get("then", {}).get("required", [])
            if cond and then:
                cond_txt = " and ".join(f"{k} == {json.dumps(v.get('const'))}" for k, v in cond.items())
                rules.append(f"if {cond_txt} then {', '.join(then)} required")
        return rules


def schemas_dir() -> Path:
    return get_settings().schemas_dir


def list_schemas() -> list[str]:
    return sorted(p.stem for p in schemas_dir().glob("*.json") if _NAME.match(p.stem))


def load_schema(name: str) -> FormSchema:
    if not _NAME.match(name or ""):
        raise SchemaNotFound(f"Invalid schema name {name!r}")
    path = schemas_dir() / f"{name}.json"
    if not path.is_file():
        raise SchemaNotFound(f"Schema {name!r} not found in {schemas_dir()} (available: {', '.join(list_schemas())})")
    raw = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(raw)
    return FormSchema(name, raw)
