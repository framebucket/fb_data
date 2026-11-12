"""Studio editing look: learn from references, preview, settings."""
import io

import pytest
from PIL import Image, ImageEnhance

from conftest import jpeg_bytes


def graded(seed: int) -> bytes:
    image = Image.open(io.BytesIO(jpeg_bytes(seed))).convert("RGB")
    image = ImageEnhance.Color(image).enhance(0.6)
    image = ImageEnhance.Contrast(image).enhance(1.25)
    r, g, b = image.split()
    image = Image.merge("RGB", (r.point(lambda v: min(255, v + 12)), g, b.point(lambda v: max(0, v - 10))))
    out = io.BytesIO()
    image.save(out, "JPEG", quality=90)
    return out.getvalue()


def test_style_learn_preview_and_clear(service, ids):
    pid = ids["pid"]
    state = service.studio_style(pid)
    assert not state["has_style"] and state["strength"] == 80
    with pytest.raises(ValueError):
        service.save_style_references(pid, [graded(1)] * 2)
    state = service.save_style_references(pid, [graded(i) for i in range(6)])
    assert state["has_style"] and state["reference_count"] == 6
    state = service.update_style_settings(pid, 60, None, False)
    assert state["strength"] == 60 and state["apply_style"] and not state["stamp_logo"]
    preview = service.style_preview(pid, "sample.jpg", jpeg_bytes(42), None)
    assert preview[:2] == b"\xff\xd8"
    state = service.clear_style(pid)
    assert not state["has_style"]
