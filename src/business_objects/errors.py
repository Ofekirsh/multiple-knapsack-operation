# -*- coding: utf-8 -*-
"""
Common exceptions for the business objects layer.
"""


class SchemaError(ValueError):
    """Raised when an input file (CSV/YAML) violates the expected schema."""


class StateValidationError(ValueError):
    """Raised when the in-memory state violates domain constraints."""
