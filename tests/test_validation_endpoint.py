from __future__ import annotations

def test_validation_endpoint_importable():
    # Basic import test; endpoint wiring is in FastAPI app
    import utils.validate_agent as va
    assert hasattr(va, "validate_live")


