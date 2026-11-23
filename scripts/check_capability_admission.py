#!/usr/bin/env python3
"""Prototype v2 capability admission without changing production selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "nvcr.capability-cases.v1"
STATUSES = {"exact", "native", "unsupported", "untested"}
MAXIMUM_BYTES = 1024 * 1024


class InputError(ValueError):
    """The prototype input is malformed."""


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise InputError(f"{where} must be an array")
    return value


def _strings(value: Any, where: str) -> list[str]:
    items = _list(value, where)
    if not all(isinstance(item, str) and item for item in items):
        raise InputError(f"{where} must contain non-empty strings")
    return items


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise InputError(f"missing capability document: {path}")
    if path.stat().st_size > MAXIMUM_BYTES:
        raise InputError(f"capability document exceeds {MAXIMUM_BYTES} bytes: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InputError(f"cannot parse {path}: {error}") from error
    root = _object(document, "document")
    if root.get("schema") != SCHEMA:
        raise InputError(f"document.schema must be {SCHEMA}")
    cases = [
        _object(item, f"cases[{index}]")
        for index, item in enumerate(_list(root.get("cases"), "cases"))
    ]
    ids = [case.get("id") for case in cases]
    if not all(isinstance(case_id, str) and case_id for case_id in ids):
        raise InputError("every case needs a non-empty id")
    if len(ids) != len(set(ids)):
        raise InputError("case ids must be unique")
    return cases


def _result(
    case: dict[str, Any], decision: str, status: str, reason: str,
    incompatibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "case_id": case["id"],
        "decision": decision,
        "status": status,
        "reason": reason,
        "alternatives": case.get("alternatives", []),
    }
    if incompatibility is not None:
        output["incompatibility"] = incompatibility
    return output


def _reject(
    case: dict[str, Any], owner: str, requirement: str, required: Any,
    available: Any, reason: str, stage_id: str | None = None,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "owner": owner,
        "requirement": requirement,
        "required": required,
        "available": available,
    }
    if stage_id is not None:
        detail["stage_id"] = stage_id
    return _result(case, "reject", "unsupported", reason, detail)


def assess(case: dict[str, Any]) -> dict[str, Any]:
    codec = _object(case.get("codec"), "codec")
    provider = _object(case.get("provider"), "provider")
    artifact = _object(case.get("artifact"), "artifact")
    workload = _object(case.get("workload"), "workload")
    conformance = _object(case.get("conformance"), "conformance")
    capabilities = _object(provider.get("capabilities"), "provider.capabilities")

    codec_id = codec.get("id")
    provider_id = provider.get("id")
    if artifact.get("codec_id") != codec_id:
        return _reject(case, "artifact", "codec_id", codec_id, artifact.get("codec_id"),
                       "artifact belongs to another codec")
    if artifact.get("provider_id") != provider_id:
        return _reject(case, "artifact", "provider_id", provider_id, artifact.get("provider_id"),
                       "artifact belongs to another provider")

    direction = workload.get("direction")
    directions = _strings(artifact.get("directions"), "artifact.directions")
    if direction not in directions:
        return _reject(case, "artifact", "direction", direction, directions,
                       "artifact does not implement the requested direction")

    provider_sets = {
        name: set(_strings(capabilities.get(name), f"provider.capabilities.{name}"))
        for name in ("tensor_types", "layouts", "shape_modes", "memory_domains",
                     "synchronization", "numerics_profiles", "model_formats")
    }
    artifact_stages = set(_strings(artifact.get("stage_ids"), "artifact.stage_ids"))
    stages = _list(codec.get("stages"), "codec.stages")
    selected_stages = 0
    for index, raw_stage in enumerate(stages):
        stage = _object(raw_stage, f"codec.stages[{index}]")
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            raise InputError(f"codec.stages[{index}].id must be a non-empty string")
        stage_directions = _strings(stage.get("directions"), f"codec.stages[{index}].directions")
        if direction not in stage_directions:
            continue
        selected_stages += 1
        if stage_id not in artifact_stages:
            return _reject(case, "artifact", "stage_id", stage_id, sorted(artifact_stages),
                           "artifact is missing a required codec stage", stage_id)
        for field in ("tensor_types", "layouts"):
            required = set(_strings(stage.get(field), f"codec.stages[{index}].{field}"))
            if not required <= provider_sets[field]:
                return _reject(
                    case, "provider", field, sorted(required), sorted(provider_sets[field]),
                    f"provider cannot satisfy {stage_id} {field}", stage_id,
                )
        for field, capability in {
            "shape_mode": "shape_modes",
            "synchronization": "synchronization",
            "numerics_profile": "numerics_profiles",
        }.items():
            required = stage.get(field)
            if required not in provider_sets[capability]:
                return _reject(case, "provider", field, required, sorted(provider_sets[capability]),
                               f"provider cannot satisfy {stage_id} {field}", stage_id)
        memory = set(_strings(stage.get("memory_domains_any_of"),
                              f"codec.stages[{index}].memory_domains_any_of"))
        if not memory & provider_sets["memory_domains"]:
            return _reject(case, "provider", "memory_domain", sorted(memory),
                           sorted(provider_sets["memory_domains"]),
                           f"provider has no usable memory domain for {stage_id}", stage_id)
    if selected_stages == 0:
        raise InputError(f"codec has no stages for workload direction {direction!r}")

    if artifact.get("format") not in provider_sets["model_formats"]:
        return _reject(case, "provider", "model_format", artifact.get("format"),
                       sorted(provider_sets["model_formats"]),
                       "provider cannot load the artifact format")
    if (artifact.get("operators_validated") is not True or
            artifact.get("digest_verified") is not True):
        return _reject(case, "artifact", "validation", True, False,
                       "artifact operators and digest must be validated before admission")

    coverage = _object(artifact.get("visible_size"), "artifact.visible_size")
    coverage_fields = ("min_width", "max_width", "min_height", "max_height")
    if not all(isinstance(coverage.get(field), int) and coverage[field] > 0
               for field in coverage_fields):
        raise InputError("artifact.visible_size limits must be positive integers")
    if (coverage["min_width"] > coverage["max_width"] or
            coverage["min_height"] > coverage["max_height"]):
        raise InputError("artifact.visible_size minimums must not exceed maximums")
    width, height = workload.get("width"), workload.get("height")
    if not all(isinstance(value, int) and value > 0 for value in (width, height)):
        raise InputError("workload width and height must be positive integers")
    if not (coverage["min_width"] <= width <= coverage["max_width"] and
            coverage["min_height"] <= height <= coverage["max_height"]):
        return _reject(case, "artifact", "visible_size", [width, height], coverage,
                       "artifact does not cover the requested visible size")

    status = conformance.get("status")
    if status not in STATUSES:
        raise InputError(f"conformance.status must be one of {', '.join(sorted(STATUSES))}")
    if status == "unsupported":
        return _result(case, "reject", status, conformance.get("reason", "known incompatibility"))
    if status == "untested":
        return _result(case, "hold", status,
                       "requirements match, but no retained conformance evidence exists")
    return _result(case, "admit", status, f"retained evidence marks this cell {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--case", dest="case_id")
    args = parser.parse_args()
    try:
        cases = load_cases(args.document)
        if args.case_id:
            cases = [case for case in cases if case["id"] == args.case_id]
            if not cases:
                raise InputError(f"unknown case id: {args.case_id}")
        failed = False
        for case in cases:
            result = assess(case)
            print(json.dumps(result, sort_keys=True))
            failed |= result["decision"] != case.get("expected_decision")
        return 1 if failed else 0
    except InputError as error:
        print(f"capability-admission: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
