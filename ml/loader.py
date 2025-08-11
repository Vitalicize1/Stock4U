from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import os


@dataclass
class SklearnArtifact:
    model: Any
    feature_names: List[str]
    metadata: Dict[str, Any]


def load_latest_model(models_dir: str = os.path.join("ml", "models")) -> SklearnArtifact | None:
    try:
        import joblib  # type: ignore
        if not os.path.isdir(models_dir):
            return None
        # pick latest .joblib file
        files = [
            os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".joblib")
        ]
        if not files:
            return None
        latest = max(files, key=lambda p: os.path.getmtime(p))
        payload = joblib.load(latest)
        model = payload.get("model")
        feature_names = payload.get("feature_names") or []
        metadata = payload.get("metadata") or {}
        if model is None or not feature_names:
            return None
        return SklearnArtifact(model=model, feature_names=feature_names, metadata=metadata)
    except Exception:
        return None


