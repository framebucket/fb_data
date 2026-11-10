# FrameBucket Data Service

Python microservice for FrameBucket gallery media. The first working slice is local image upload, original preservation, preview processing, and album-design handoff.

## Implemented

- Creates separate storage areas for private photographer originals and customer event previews.
- Copies each uploaded source byte-for-byte while independently processing a JPEG preview in parallel.
- Reads JPEG, PNG, TIFF, WebP, BMP, and common RAW formats (ARW, CR2/CR3, DNG, NEF, ORF, RAF, RW2).
- Applies restrained exposure, color, contrast, and sharpening adjustments. Output keeps source pixel dimensions; JPEG optimization tries to reduce file size without downscaling.
- Supports an optional ONNX image-enhancement model. No model is bundled; without one, the response reports ai_enhanced=false.
- Places the photographer logo in a low-texture corner that does not overlap detected faces. If face detection is unavailable or no safe corner exists, the logo is skipped.
- Submits selected photo IDs for album design, moves those originals into a private album-design folder, and creates a random expiring ZIP link.
- Provides a protected FastAPI API for the existing app backend.

The local gallery is integrated with the FrameBucket Spring Boot app. It supports photographer folders, image and video upload, parallel original/preview writes, customer preview galleries, photo selection, and album-design ZIP handoff. WhatsApp delivery uses the app's existing console mode locally; configure an approved WhatsApp template for live delivery. Production queueing and object storage are described in [INFRASTRUCTURE.md](INFRASTRUCTURE.md).

## Local setup

Use Python 3.11 or newer. The requested data directory is:

~~~text
C:\Users\srini\OneDrive\Documents\FrameBucket\data_output
~~~

1. Create and activate a virtual environment from this repository:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
~~~

2. Copy `.env.example` to `.env`. Set a long random `FB_SERVICE_TOKEN` and `FB_DATA_ROOT`. Set the same token in the Java backend as `DATA_SERVICE_TOKEN`. Optionally set `FB_LOGO_PATH` to a transparent PNG and `FB_PUBLIC_BASE_URL` to a URL the album designer can reach. The service loads `.env` automatically.

3. Start the API in a separate terminal after installing the package:

~~~powershell
uvicorn app.api:app --host 127.0.0.1 --port 8001
~~~

4. Start the Spring Boot backend with `DATA_SERVICE_TOKEN` set to the same token, then start the React frontend as usual. The Data tab requires a plan with Data enabled. Events can be provisioned when created; if the Python service was offline, opening that event gallery provisions its storage.

To run the standalone local batch processor instead, use `python -m app.cli --event-id DEMO_EVENT --photographer-id DEMO_PHOTOGRAPHER --customer-id DEMO_CUSTOMER`. Use `--input <folder>`, `--workers 4`, or `--json` to customize it.

## Optional AI enhancement

Install ONNX Runtime:

~~~powershell
pip install -e ".[ai]"
~~~

Set FB_ENHANCER_MODEL to an ONNX model file that accepts one RGB float32 NCHW tensor normalized to [0, 1] and returns an RGB image in the same layout/range. The model must use dynamic spatial dimensions or exactly match each image's dimensions. The processor still applies conservative color/detail finishing and never downsizes the result. OpenCV's frontal-face detector is used for logo placement; the logo is skipped when face detection is unavailable.

This hook allows a locally supplied enhancement model to be evaluated without binding the service to a single vendor. Benchmark a production model for color fidelity, skin tones, image size, throughput, memory, and licensing before deployment.

## Local API

The API requires FB_SERVICE_TOKEN. Keep it private and bind the local service to loopback:

~~~powershell
uvicorn app.api:app --host 127.0.0.1 --port 8001
~~~

Send the X-FrameBucket-Service-Key header from the trusted Spring Boot backend. The token protects internal endpoints; the customer-facing application must authenticate the customer before it asks this service for that customer's event gallery.

- GET /health
- POST /api/v1/events — create event storage areas.
- GET/POST /api/v1/events/{event_id}/folders — list or create folders (requires photographer_id and customer_id query parameters).
- POST /api/v1/events/{event_id}/photos?photographer_id=...&customer_id=...&folder_id=... — multipart field `file`; accepts supported images and videos.
- GET /api/v1/events/{event_id}/photos?customer_id=...&folder_id=... — list customer-visible previews.
- GET /api/v1/events/{event_id}/album-selection — load the customer's temporary album selection.
- PUT /api/v1/events/{event_id}/album-selection — save selected photo IDs to the temporary selection store.
- POST /api/v1/events/{event_id}/downloads — JSON `{"customer_id":"...","photo_ids":[]}` returns one customer-visible item directly, or multiple/all items as a ZIP; pass photo IDs to download only selected items.
- GET /api/v1/events/{event_id}/photos/{photo_id}/preview?customer_id=... — fetch an event preview.
- GET /api/v1/events/{event_id}/photos/{photo_id}/original?photographer_id=...&customer_id=... — photographer-only original.
- POST /api/v1/events/{event_id}/album-submissions — JSON: {"photographer_id":"...","customer_id":"...","photo_ids":["..."]}.
- GET /api/v1/album-design/{token} — download the expiring album ZIP.

Example upload from PowerShell:

~~~powershell
curl.exe -H "X-FrameBucket-Service-Key: $env:FB_SERVICE_TOKEN" -F "file=@C:\path\to\sample.jpg" "http://127.0.0.1:8001/api/v1/events/DEMO_EVENT/photos?photographer_id=DEMO_PHOTOGRAPHER&customer_id=DEMO_CUSTOMER"
~~~

A successful album submission returns album_design_link, photo_numbers, and an expiry. The Spring Boot app stores the returned link in the photographer event list and sends the configured WhatsApp template. The customer API returns only whether a selection was submitted, never the designer link. The local ZIP URL is a bearer link: anyone holding it can download the selected originals until it expires.

## Storage layout

~~~text
data_output/
  input/                                  # local source photos to process
  photographer/<photographer>/<event>/originals/<folder>/
  events/<event>/photos/<folder>/         # customer-visible image derivatives only
  events/<event>/videos/<folder>/         # customer-visible video copies
  album_design/<event>/<submission>/       # selected originals and ZIP
  metadata/events/<event>/                # event, folder, and media manifests
  tmp/                                    # temporary uploads, customer ZIPs, and album-selection drafts
~~~

The filesystem adapter is for development only. The private photographer path is never returned by the customer gallery API. See INFRASTRUCTURE.md for the production design.
