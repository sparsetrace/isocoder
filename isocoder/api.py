from __future__ import annotations
import time
from typing import Any, Dict, Optional, Union

import requests
import numpy as np

from .utils import np_to_base64_npz, clean_base_url
from .errors import IsocoderError, AuthError, RemoteJobError, TimeoutError as ClientTimeout

DEFAULT_TIMEOUT_S = 60 * 60         # overall wall clock cap
DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_REQUEST_TIMEOUT = 60        # per-HTTP call timeout in seconds

class TVAEResult(dict):
    def __str__(self) -> str:
        return str(dict(self))

class TVAE:
    """
    Blocking convenience wrapper:

    TVAE(Data=np_array, Modal_ID="https://<modal-app>.modal.run",
         Modal_Key="your-bearer", HF_key="hf_xxx", HF_repo="you/tvae-results")

    Optional:
      - dataset_filename: use file inside HF repo instead of raw array
      - gpu: "L4" | "A10G" | "A100" | "H100"
      - config: forwarded to backend TVAE(**config)
      - timeout_s / poll_interval_s
    """

    def __init__(
        self,
        Data: Optional[np.ndarray] = None,
        Modal_ID: str = "",
        Modal_Key: str = "",
        HF_key: Optional[str] = None,
        HF_repo: str = "",
        dataset_filename: Optional[str] = None,
        gpu: str = "L4",
        config: Optional[Dict[str, Any]] = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT,
    ):
        if not HF_repo:
            raise ValueError("HF_repo is required")
        if not Modal_ID:
            raise ValueError("Modal_ID (backend base URL) is required")
        if Modal_Key is None or Modal_Key == "":
            raise ValueError("Modal_Key (bearer) is required")

        if Data is None and dataset_filename is None:
            raise ValueError("Provide either Data (numpy array) or dataset_filename")

        base_url = clean_base_url(Modal_ID)
        run_url = f"{base_url}/run"
        status_url = f"{base_url}/status"

        headers = {"Authorization": f"Bearer {Modal_Key}", "Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "hf_repo": HF_repo,
            "hf_token": HF_key,
            "dataset": dataset_filename,
            "data_b64": None,
            "config": config or {},
            "gpu": gpu,
            "timeout_s": timeout_s,
        }

        if Data is not None:
            payload["data_b64"] = np_to_base64_npz(Data)
            payload["dataset"] = None  # prefer raw data

        # --- submit job ---
        try:
            r = requests.post(run_url, json=payload, headers=headers, timeout=request_timeout_s)
        except requests.RequestException as e:
            raise IsocoderError(f"Failed to contact backend /run: {e}") from e

        if r.status_code in (401, 403):
            raise AuthError(f"Auth failed: {r.text}")
        if r.status_code >= 400:
            raise IsocoderError(f"Backend /run error {r.status_code}: {r.text}")

        job_id = r.json().get("job_id")
        if not job_id:
            raise IsocoderError("Backend /run missing job_id")

        # --- poll for completion ---
        deadline = time.time() + timeout_s
        result: Optional[Dict[str, Any]] = None
        last_state = None

        while time.time() < deadline:
            try:
                s = requests.get(f"{status_url}/{job_id}", headers=headers, timeout=request_timeout_s)
            except requests.RequestException as e:
                # brief backoff then keep polling (transient net errors)
                time.sleep(min(5.0, poll_interval_s))
                continue

            if s.status_code in (401, 403):
                raise AuthError(f"Auth failed during polling: {s.text}")
            if s.status_code == 404:
                raise IsocoderError("Unknown job_id (was state lost on backend?)")
            if s.status_code >= 400:
                raise IsocoderError(f"Backend /status error {s.status_code}: {s.text}")

            js = s.json()
            state = js.get("state")
            if state != last_state:
                last_state = state  # (you could hook a logger here)

            if state == "succeeded":
                result = js.get("result") or {}
                break
            if state == "failed":
                err = js.get("error") or "unknown error"
                raise RemoteJobError(f"Remote job failed: {err}")

            time.sleep(poll_interval_s)

        if result is None:
            raise ClientTimeout("TVAE remote job timeout")

        self.result = TVAEResult(result)

    def __repr__(self) -> str:
        return f"TVAE({dict(self.result)})"

    def to_dict(self) -> dict:
        return dict(self.result)

# Functional helper (non-class)
def run_tvae(
    Data: Optional[np.ndarray],
    Modal_ID: str,
    Modal_Key: str,
    HF_key: Optional[str],
    HF_repo: str,
    dataset_filename: Optional[str] = None,
    gpu: str = "L4",
    config: Optional[Dict[str, Any]] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """
    Convenience function; returns the plain result dict.
    """
    client = TVAE(
        Data=Data,
        Modal_ID=Modal_ID,
        Modal_Key=Modal_Key,
        HF_key=HF_key,
        HF_repo=HF_repo,
        dataset_filename=dataset_filename,
        gpu=gpu,
        config=config,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        request_timeout_s=request_timeout_s,
    )
    return client.to_dict()
