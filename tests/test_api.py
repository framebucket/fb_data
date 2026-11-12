"""HTTP layer: service-key auth and a few routes through FastAPI's test client."""
import pytest

pytest.importorskip("httpx")


@pytest.fixture(scope="module")
def client(service):
    from fastapi.testclient import TestClient

    from app.api import app
    with TestClient(app) as test_client:
        yield test_client


HEADERS = {"X-FrameBucket-Service-Key": "test-service-token"}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_service_key_required(client, ids):
    url = f"/api/v1/events/{ids['eid']}/album-designs"
    params = {"photographer_id": ids["pid"], "customer_id": ids["cid"]}
    assert client.get(url, params=params).status_code == 401
    assert client.get(url, params=params, headers={"X-FrameBucket-Service-Key": "wrong"}).status_code == 401
    response = client.get(url, params=params, headers=HEADERS)
    assert response.status_code == 200 and response.json() == {"submissions": []}


def test_invitation_routes(client, ids):
    base = f"/api/v1/events/{ids['eid']}/invitation"
    owner = {"photographer_id": ids["pid"], "customer_id": ids["cid"]}
    assert client.get(base, params=owner, headers=HEADERS).json()["exists"] is False
    bad = client.put(base, json={**owner, "details": {"bride": "A"}}, headers=HEADERS)
    assert bad.status_code == 422
    good = client.put(base, json={**owner, "details": {"bride": "A", "groom": "B",
                                                       "events": [{"date": "1 Jan", "venue": "Hall"}]}}, headers=HEADERS)
    assert good.status_code == 200 and good.json()["exists"]
    designs = client.post(base + "/designs", json=owner, headers=HEADERS)
    assert designs.status_code == 200 and len(designs.json()["designs"]) == 5
    image = client.get(base + "/designs/0", params={"customer_id": ids["cid"]}, headers=HEADERS)
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert client.get(base + "/video", params={"customer_id": ids["cid"]}, headers=HEADERS).status_code == 404
