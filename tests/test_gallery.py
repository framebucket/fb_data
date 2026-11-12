"""Core gallery flow: ingest -> process -> list -> album selection -> album ZIP link."""
import zipfile

import pytest

from conftest import drain


def test_ingest_process_and_list(service, event, photos):
    files = photos(3)
    ids = [service.ingest_file(f, f.name, event["eid"], event["pid"], event["cid"])["photo_id"] for f in files]
    results = drain(service)
    assert results and all(r["ok"] for r in results), results
    gallery = service.list_gallery(event["eid"], event["cid"])
    assert [m["photo_id"] for m in gallery] == ids
    assert all(m["status"] == "READY" and m["width"] and m["height"] for m in gallery)


def test_gallery_is_private_to_the_customer(service, event, photos):
    service.ingest_file(photos(1)[0], "a.jpg", event["eid"], event["pid"], event["cid"])
    with pytest.raises(ValueError):
        service.list_gallery(event["eid"], "someone-else")


def test_album_submission_builds_zip(service, event, photos):
    ids = [service.ingest_file(f, f.name, event["eid"], event["pid"], event["cid"])["photo_id"] for f in photos(4)]
    drain(service)
    result = service.submit_album(event["eid"], event["pid"], event["cid"], ids[:3])
    assert result["photo_count"] == 3 and result["status"] == "SUBMITTED"
    drain(service)
    token = result["album_design_link"].rsplit("/", 1)[1]
    path, filename, cleanup = service.resolve_album_link(token)
    with zipfile.ZipFile(path) as archive:
        assert len([n for n in archive.namelist() if not n.endswith("/")]) >= 3
    assert filename.endswith(".zip")
    if cleanup:
        path.unlink(missing_ok=True)
