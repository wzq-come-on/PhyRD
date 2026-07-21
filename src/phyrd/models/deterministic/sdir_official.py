"""Compatibility import for the pre-v11 SDIR module path.

The implementation now lives in :mod:`phyrd.models.deterministic.sdir_official`.
This file remains so existing scripts and checkpoints can keep importing the
old path during the migration.
"""

from .deterministic.sdir_official import OfficialSDIRForecast

__all__ = ["OfficialSDIRForecast"]
