"""AI album auto-design: validation, job, previews, archives, publishing to the gallery for proofing."""
import zipfile

import pytest

from app.service import DesignNotReady
from conftest import drain


@pytest.fixture
def submission(service, event, photos):
    ids = [service.ingest_file(f, f.name, event["eid"], event["pid"], event["cid"])["photo_id"] for f in photos(16)]
    drain(service)
    service.submit_album(event["eid"], event["pid"], event["cid"], ids)
    drain(service)
    listed = service.album_submissions(event["eid"], event["pid"], event["cid"])
    assert len(listed) == 1 and listed[0]["photo_count"] == 16
    return event | {"sid": listed[0]["submission_id"], "photo_ids": ids}


def test_unknown_event_lists_nothing(service, ids):
    assert service.album_submissions(ids["eid"], ids["pid"], ids["cid"]) == []


@pytest.mark.parametrize("bad", [
    dict(height_in=12, width_in=10, sheets=5),
    dict(height_in=12, width_in=30, sheets=0),
    dict(height_in=12, width_in=30, sheets=5, dpi=72),
])
def test_rejects_bad_specs(service, submission, bad):
    s = submission
    with pytest.raises(ValueError):
        service.create_album_design(s["eid"], s["pid"], s["cid"], s["sid"], **bad)


def test_owner_check(service, submission):
    s = submission
    with pytest.raises(ValueError):
        service.create_album_design(s["eid"], s["pid"], "other-customer", s["sid"], 12, 30, 5)


def test_design_end_to_end(service, submission):
    s = submission
    design = service.create_album_design(s["eid"], s["pid"], s["cid"], s["sid"], 12, 36, 5, dpi=150, size_label="12x36")
    assert design["status"] == "QUEUED" and design["version"] == 1 and design["size_label"] == "12x36"
    with pytest.raises(ValueError):            # one design at a time per event
        service.create_album_design(s["eid"], s["pid"], s["cid"], s["sid"], 12, 30, 5)
    with pytest.raises(DesignNotReady):
        service.album_design_download(s["eid"], design["design_id"], "psd", s["pid"], s["cid"])

    ran = drain(service, {"design_album"})
    assert ran and all(r["ok"] for r in ran), ran
    status = service.album_design_status(s["eid"], design["design_id"], s["pid"], s["cid"])
    assert status["status"] == "READY" and status["progress"] == 100
    assert len(status["spreads"]) == 5
    used = sorted(n for spread in status["spreads"] for n in spread["photos"])
    assert len(used) == 16 and len(set(used)) == 16          # every selected photo placed exactly once

    preview, cleanup = service.album_design_preview(s["eid"], design["design_id"], 0, s["pid"], s["cid"])
    from PIL import Image
    with Image.open(preview) as image:
        assert abs(image.width / image.height - 3.0) < 0.02   # 12x36 open spread
    with pytest.raises(FileNotFoundError):
        service.album_design_preview(s["eid"], design["design_id"], 5, s["pid"], s["cid"])

    for kind, ext in (("jpeg", ".jpg"), ("psd", ".psd")):
        path, filename, cleanup = service.album_design_download(s["eid"], design["design_id"], kind, s["pid"], s["cid"])
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        assert len([n for n in names if n.endswith(ext)]) == 5 and "README.txt" in names
        assert filename.endswith(".zip")

    published = service.publish_album_design(s["eid"], design["design_id"], s["pid"], s["cid"])
    assert published["spreads"] == 5
    spreads = service.list_gallery(s["eid"], s["cid"], published["folder_id"])
    assert len(spreads) == 5 and all(m["status"] == "READY" for m in spreads)
    assert service.publish_album_design(s["eid"], design["design_id"], s["pid"], s["cid"])["folder_id"] == published["folder_id"]

    second = service.create_album_design(s["eid"], s["pid"], s["cid"], s["sid"], 12, 30, 4, dpi=150)
    assert second["version"] == 2
