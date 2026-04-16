"""
This package has been renamed to arc-sentry.

    pip uninstall bendex
    pip install arc-sentry
"""
import warnings
warnings.warn(
    "The bendex package has been renamed to arc-sentry. "
    "Please run: pip uninstall bendex && pip install arc-sentry",
    DeprecationWarning, stacklevel=2
)
