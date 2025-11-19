"""Risk management module.

This package exposes common risk control helpers. The original layout
expected a ``controls.py`` module in the package. Some installations of
the project omit that file which caused an ImportError on ``import risk``.

We attempt to import helpers from a local ``controls`` module first; if
that fails we provide fallbacks by importing the functions from
``runner.py`` (where many of the risk helpers also live). This keeps
``import risk`` safe for callers in the repo and for the unit tests.
"""

try:
    from .controls import (
        check_exposure_limit,
        check_kill_switch,
        calculate_max_position_size_for_risk,
        check_vix_filter,
        get_vix_level,
    )
except Exception:
    # Fallback: import from runner to avoid breaking callers when
    # risk/controls.py is absent. Import inside the except so the
    # package import never raises on missing optional module.
    try:
        from ..runner import (
            check_exposure_limit,
            check_kill_switch,
            calculate_max_position_size_for_risk,
            check_vix_filter,
            get_vix_level,
        )
    except Exception:
        # Last-resort: provide placeholder functions that raise a clear
        # ImportError when used — keeps import-time safe while giving
        # actionable error messages at call time.
        def _missing(name):
            def _raiser(*args, **kwargs):
                raise ImportError(
                    f"Risk helper '{name}' is unavailable: missing 'risk.controls' "
                    "and could not import from runner. Ensure the package is complete."
                )

            return _raiser

        check_exposure_limit = _missing("check_exposure_limit")
        check_kill_switch = _missing("check_kill_switch")
        calculate_max_position_size_for_risk = _missing("calculate_max_position_size_for_risk")
        check_vix_filter = _missing("check_vix_filter")
        get_vix_level = _missing("get_vix_level")

__all__ = [
    "check_exposure_limit",
    "check_kill_switch",
    "calculate_max_position_size_for_risk",
    "check_vix_filter",
    "get_vix_level",
]

