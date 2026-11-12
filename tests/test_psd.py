"""The layered PSD writer produces files Photoshop-compatible readers can open."""
from PIL import Image

from app import psd


def test_layered_psd_round_trip(tmp_path):
    background = Image.new("RGBA", (600, 240), (240, 230, 200, 255))
    photo = Image.new("RGBA", (200, 150), (200, 40, 60, 255))
    document = psd.Document(600, 240, dpi=300, layers=[
        psd.layer_from(background, "Background", 0, 0),
        psd.layer_from(photo, "Photo 00001 - IMG_0001.jpg", 50, 40),
    ], guides=[psd.Guide(300, True)])
    composite = background.copy()
    composite.alpha_composite(photo, (50, 40))
    out = tmp_path / "spread.psd"
    psd.save(document, out, composite.convert("RGB"))
    with Image.open(out) as opened:
        assert opened.format == "PSD" and opened.size == (600, 240)
        assert getattr(opened, "n_frames", 1) >= 2           # Pillow exposes layers as frames
        layers = getattr(opened, "layers", [])
        assert any("Photo 00001" in str(layer[0]) for layer in layers), layers
