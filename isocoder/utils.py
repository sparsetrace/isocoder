from __future__ import annotations
import base64, io
import numpy as np

def np_to_base64_npz(arr: np.ndarray) -> str:
    """Encode numpy array as base64(np.savez_compressed(..., x=arr)) string."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("Data must be a numpy.ndarray")
    buf = io.BytesIO()
    np.savez_compressed(buf, x=arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def clean_base_url(url: str) -> str:
    url = url.strip()
    return url[:-1] if url.endswith("/") else url
