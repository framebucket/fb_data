"""Customer wedding invitations: details, photo, five designs, regenerate, finalize to MP4."""
import shutil

import pytest

from app import invitation as engine
from app.invitation import InvitationError
from conftest import drain, jpeg_bytes

DETAILS = {
    "bride": "Varsha", "groom": "Nitish", "invocation": "|| Shri Ganeshaya Namah ||",
    "bride_family": "D/o Smt. Lakshmi & Sri. Ramesh", "groom_family": "S/o Smt. Shobha & Sri. Prakash",
    "events": [{"title": "Muhurtham", "date": "Sunday, 25 January 2026", "time": "10:30 AM",
                "venue": "Sri Krishna Kalyana Mantapa", "address": "Jayanagar, Bengaluru"},
               {"title": "Reception", "date": "Saturday, 24 January 2026", "time": "7:00 PM", "venue": "Same venue"}],
    "hosts": "With best compliments from family & friends", "rsvp": "RSVP +91 98450 12345",
}


def test_details_validation():
    with pytest.raises(InvitationError):
        engine.normalise_details({"bride": "A"})
    with pytest.raises(InvitationError):
        engine.normalise_details({"bride": "A", "groom": "B", "events": []})
    clean = engine.normalise_details({**DETAILS, "kind": "nonsense", "bride": "  Varsha\x00 "})
    assert clean["kind"] == "WEDDING" and clean["bride"] == "Varsha" and len(clean["events"]) == 2


def test_choose_designs_are_distinct_and_regenerate_avoids_previous():
    first = engine.choose_designs(1, has_photo=True)
    assert len(first) == 5 and len({c.theme for c in first}) == 5
    second = engine.choose_designs(2, has_photo=True, avoid={c.theme for c in first})
    assert len({c.theme for c in second} - {c.theme for c in first}) >= 3
    assert all(c.layout == "medallion" for c in engine.choose_designs(3, has_photo=False))


@pytest.mark.parametrize("theme", sorted(engine.THEMES))
def test_every_theme_renders(theme):
    details = engine.normalise_details(DETAILS)
    from PIL import Image
    photo = Image.new("RGB", (1200, 1600), (120, 90, 80))
    for layout, picture in (("arch", photo), ("medallion", None)):
        poster = engine.render_poster(engine.build_scene(details, engine.DesignChoice(theme, layout, 7), picture, []))
        assert poster.size == (engine.W, engine.H)


def test_indian_script_names_render():
    details = engine.normalise_details({**DETAILS, "bride": "ವರ್ಷ", "groom": "ನಿತೀಶ್"})
    poster = engine.render_poster(engine.build_scene(details, engine.DesignChoice("royal_maroon", "medallion", 3)))
    assert poster.size == (engine.W, engine.H)


def test_invitation_flow(service, ids):
    pid, cid, eid = ids["pid"], ids["cid"], ids["eid"]
    state = service.invitation(eid, pid, cid)
    assert not state["exists"] and state["regenerations_left"] == 40
    with pytest.raises(InvitationError):
        service.generate_invitation_designs(eid, pid, cid)

    state = service.save_invitation_details(eid, pid, cid, DETAILS)
    assert state["exists"] and state["details"]["bride"] == "Varsha"
    with pytest.raises(ValueError):
        service.invitation(eid, pid, "someone-else")

    state = service.generate_invitation_designs(eid, pid, cid)
    assert len(state["designs"]) == 5 and all(d["layout"] == "medallion" for d in state["designs"])
    first = {d["theme"] for d in state["designs"]}

    with pytest.raises(InvitationError):
        service.save_invitation_photo(eid, pid, cid, b"not an image")
    state = service.save_invitation_photo(eid, pid, cid, jpeg_bytes(5))
    assert state["has_photo"]
    state = service.generate_invitation_designs(eid, pid, cid)
    assert len({d["theme"] for d in state["designs"]} - first) >= 3
    assert all(d["layout"] != "medallion" for d in state["designs"])
    assert state["regenerations_left"] == 38

    preview = service.invitation_preview(eid, cid, 0)
    assert preview[:2] == b"\xff\xd8"
    with pytest.raises(FileNotFoundError):
        service.invitation_preview(eid, cid, 9)
    with pytest.raises(FileNotFoundError):
        service.invitation_file(eid, cid, "video")

    state = service.finalize_invitation(eid, pid, cid, 1)
    assert state["status"] == "RENDERING" and state["chosen"]["theme"] == state["designs"][1]["theme"]
    with pytest.raises(InvitationError):
        service.generate_invitation_designs(eid, pid, cid)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_invitation_video(service, ids):
    pid, cid, eid = ids["pid"], ids["cid"], ids["eid"]
    service.save_invitation_details(eid, pid, cid, DETAILS)
    service.save_invitation_photo(eid, pid, cid, jpeg_bytes(9))
    service.generate_invitation_designs(eid, pid, cid)
    service.finalize_invitation(eid, pid, cid, 0)
    ran = drain(service, {"render_invitation"})
    assert ran and all(r["ok"] for r in ran), ran
    state = service.invitation(eid, pid, cid)
    assert state["status"] == "READY" and state["video_ready"]
    video, _, content_type = service.invitation_file(eid, cid, "video")
    assert content_type == "video/mp4" and video.read_bytes()[4:8] == b"ftyp"
    poster, _, content_type = service.invitation_file(eid, cid, "poster")
    assert content_type == "image/jpeg"
