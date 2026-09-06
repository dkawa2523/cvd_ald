"""Reporting package for deposition simulation outputs."""

__version__ = "0.1.0"

from .run_report import write_run_report
from .fit_plots import write_fit_diagnostic_plots

__all__ = ["__version__", "write_fit_diagnostic_plots", "write_run_report"]
