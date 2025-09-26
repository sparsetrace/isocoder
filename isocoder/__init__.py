"""
isocoder: tiny client for the isocoder_backend (Modal) TVAE service.
"""

from .api import TVAE, run_tvae
from .errors import IsocoderError, AuthError, RemoteJobError, TimeoutError

__all__ = ["TVAE", "run_tvae", "IsocoderError", "AuthError", "RemoteJobError", "TimeoutError"]
__version__ = "0.1.0"