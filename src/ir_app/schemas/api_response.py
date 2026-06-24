"""Consistent Flask API response helpers."""

from __future__ import annotations

from typing import Any

from flask import jsonify


def api_success(data: dict[str, Any] | None = None, meta: dict[str, Any] | None = None, **legacy: Any):
    """Return a success response with transitional legacy fields.

    Complexity:
        Time: O(k) where k is the number of legacy fields
        Space: O(k)
    """
    payload: dict[str, Any] = {
        "ok": True,
        "success": True,
        "data": data or {},
        "meta": meta or {},
    }
    payload.update(legacy)
    return jsonify(payload)


def api_error(
    code: str,
    message: str,
    status: int = 400,
    details: dict[str, Any] | None = None,
    **legacy: Any,
):
    """Return a structured error response with transitional legacy fields.

    Complexity:
        Time: O(k) where k is the number of details and legacy fields
        Space: O(k)
    """
    error = {"code": code, "message": message}
    if details:
        error["details"] = details

    payload: dict[str, Any] = {
        "ok": False,
        "success": False,
        "error": error,
        "message": message,
    }
    payload.update(legacy)
    return jsonify(payload), status
