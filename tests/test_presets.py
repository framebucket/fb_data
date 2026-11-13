"""Editing presets: parsing Lightroom/LUT files, compiling to a 3D LUT and picking one per photo."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app import presets

XMP = b"""<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
 crs:IncrementalTemperature="+20" crs:Exposure2012="+0.40" crs:Contrast2012="-10" crs:Clarity2012="+15" crs:Vibrance="+10">
 <crs:Name><rdf:Alt><rdf:li xml:lang="x-default">Warm Airy</rdf:li></rdf:Alt></crs:Name>
 <crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 20</rdf:li><rdf:li>255, 250</rdf:li></rdf:Seq></crs:ToneCurvePV2012>
</rdf:Description></rdf:RDF></x:xmpmeta>"""

LRTEMPLATE = b"""s = { title = "Moody", value = { settings = { Exposure2012 = -0.4, Saturation = -30,
 ToneCurvePV2012 = { 0, 25, 255, 235, }, }, }, version = 0, }"""


def _cube(fn, size=9):
    axis = np.linspace(0, 1, size)
    rows = [f"LUT_3D_SIZE {size}"]
    for b in axis:
        for g in axis:
            for r in axis:
                rows.append(" ".join(f"{v:.5f}" for v in fn(r, g, b)))
    return "\n".join(rows).encode()


def _photo(level=150):
    rng = np.random.default_rng(1)
    base = np.clip(rng.normal(level, 40, (240, 320, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(base)


def test_xmp_is_compiled_with_name_and_skipped_local_effects():
    name, lut, summary = presets.parse("airy.xmp", XMP)
    assert name == "Warm Airy" and lut.shape == (33, 33, 33, 3)
    assert summary["kind"] == "xmp" and summary["skipped"] == ["clarity"]
    assert "brighter" in summary["words"] and "warm" in summary["words"]
    mid = presets.apply_lut_array(np.array([0.5, 0.5, 0.5], dtype=np.float32), lut)
    assert mid[0] > mid[2]                                   # warmer: more red than blue


def test_lrtemplate_and_identity_cube():
    name, lut, summary = presets.parse("moody.lrtemplate", LRTEMPLATE)
    assert name == "Moody" and "darker" in summary["words"] and "muted colour" in summary["words"]
    _, identity, _ = presets.parse("id.cube", _cube(lambda r, g, b: (r, g, b)))
    values = np.random.default_rng(2).random((500, 3)).astype(np.float32)
    assert np.abs(presets.apply_lut_array(values, identity) - values).max() < 1e-3


def test_cube_axis_order_red_fastest():
    _, lut, _ = presets.parse("swap.cube", _cube(lambda r, g, b: (b, g, r)))
    out = presets.apply_lut_array(np.array([1.0, 0.0, 0.0], dtype=np.float32), lut)
    assert out[2] > 0.99 and out[0] < 0.01


@pytest.mark.parametrize("filename,data", [
    ("notes.txt", b"hello"),
    ("short.cube", b"LUT_3D_SIZE 2\n0 0 0\n"),
    ("empty.xmp", b'<x:xmpmeta xmlns:crs="c"><d crs:Version="15"/></x:xmpmeta>'),
    ("huge.cube", b"0" * (presets.MAX_FILE_BYTES + 1)),
])
def test_bad_files_are_refused(filename, data):
    with pytest.raises(presets.PresetError):
        presets.parse(filename, data)


def test_pack_round_trip_and_apply_strength():
    _, lut, _ = presets.parse("airy.xmp", XMP)
    assert np.abs(presets.unpack(presets.pack(lut)) - lut).max() < 1e-4
    photo = _photo()
    assert np.array_equal(np.asarray(presets.apply(photo, lut, 0.0)), np.asarray(photo))
    assert np.asarray(presets.apply(photo, lut, 1.0)).mean() > np.asarray(photo).mean()


def test_choose_prefers_scene_and_sane_exposure():
    _, bright, _ = presets.parse("airy.xmp", XMP)
    _, dark, _ = presets.parse("moody.lrtemplate", LRTEMPLATE)
    options = [presets.Preset("a", "Airy", "any", bright), presets.Preset("m", "Moody", "night", dark)]
    night = _photo(level=25)
    chosen, why = presets.choose(night, options, original=night)
    assert why["scene"] == "night" and chosen.preset_id == "m"
    day = _photo(level=80)          # a bit dark: the brighter preset suits it
    chosen, why = presets.choose(day, [presets.Preset("a", "Airy", "any", bright),
                                       presets.Preset("m", "Moody", "any", dark)], original=day)
    assert chosen.preset_id == "a" and set(why["scores"]) == {"a", "m"}
    assert presets.choose(day, [])[0] is None
