"""
based-thermals application package.
"""

import os

# Opt out before native runtime initialization; probes and workers inherit this policy.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

__version__ = "0.1.0"
APP_NAME = "based-thermals"
