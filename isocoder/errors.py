class IsocoderError(Exception):
    """Base error for isocoder client."""

class AuthError(IsocoderError):
    """Authentication / authorization error."""

class RemoteJobError(IsocoderError):
    """Remote GPU job failed on the backend."""

class TimeoutError(IsocoderError):
    """Polling timed out."""
