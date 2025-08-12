from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time

from utils.logger import emit_alert


@dataclass
class ErrorInfo:
    code: str
    message: str
    provider: Optional[str] = None
    retryable: bool = False
    details: Optional[Dict[str, Any]] = None
    ts: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # human-friendly timestamp
        d["ts"] = int(self.ts * 1000)
        return d


def safe_response(payload: Dict[str, Any], error: Optional[ErrorInfo] = None) -> Dict[str, Any]:
    """Attach structured error_info without breaking existing response shape.

    Also emits an alert for ops visibility.
    """
    out = dict(payload or {})
    if error is not None:
        out["error_info"] = error.to_dict()
        try:
            emit_alert(
                title=f"Agent Error: {error.code}",
                body=error.message,
                level="error" if not error.retryable else "warning",
                kv={"provider": error.provider or "", **(error.details or {})},
            )
        except Exception:
            pass
    return out


