# Data module infrastructure plan

## Local integrated setup

The Spring Boot backend is the only browser-facing gateway to the Python service. It checks the signed-in photographer or customer against the event record, then calls Python with the shared service token. The token stays in environment configuration and is never sent to React.

For local development, run FastAPI on `127.0.0.1:8001`, set `DATA_SERVICE_BASE_URL=http://127.0.0.1:8001` and `DATA_SERVICE_TOKEN` for Spring Boot, and use the same value for `FB_SERVICE_TOKEN` in the Python `.env`. Configure `FB_PUBLIC_BASE_URL` to a URL reachable by the album designer; loopback is usable only from the same computer.

## Production service boundaries

- **Application API:** Run the existing Java app and the Python data API as separate private services. Only Java handles user sessions, event ownership, and customer versus photographer authorization. Keep FastAPI on private networking with service-to-service authentication.
- **Upload path:** Accept large uploads as streaming multipart requests. For production, let Java authorize an event and issue short-lived, event-scoped upload grants to private object storage; have the Python service verify completion before processing. This avoids buffering multi-gigabyte videos in the Java API.
- **Original storage:** Use a private, versioned object-store bucket with encryption, lifecycle rules, and access limited to the data workers and authorized backend roles. Key by photographer and event. Customer roles must never receive original-object grants.
- **Customer previews:** Keep processed images and customer-viewable videos in a separate private bucket or isolated prefix. Serve them through short-lived event-scoped URLs or an authenticated media proxy after Java checks the customer's event relationship.
- **Processing workers:** Put media jobs on a durable queue. Scale CPU workers for JPEG/RAW transforms and separate GPU-capable workers only if the chosen enhancement model benefits from a GPU. Workers should be stateless and fetch inputs from private storage, write outputs atomically, and update job status. Video should be copied or transcoded by an explicit video pipeline; image enhancement models should not run on video files.
- **Metadata:** Keep event-to-media mappings, folder records, processing status, photo numbers, album submissions, and expiration timestamps in a transactional database. Use idempotency keys so retries cannot duplicate photo records or album submissions.
- **Album handoff:** Copy selected originals into a restricted album-design location, record the photo numbers and audit trail, and issue a time-limited download link. Do not expose the link to the customer portal. Store it in the photographer event record and notify the photographer using the existing WhatsApp adapter.
- **Operations:** Track upload failures, queue age, processing duration, disk/object-store capacity, and failed notifications. Set request size/time limits, rate limits, malware scanning policy, retention periods, and backup/restore procedures before enabling public uploads.

The data service includes a local filesystem adapter, an S3-compatible object-storage adapter (including Railway Buckets), and a Google Drive adapter. `FB_STORAGE_BACKEND` selects the adapter. These adapters keep the same logical storage keys, so event metadata and galleries stay grouped consistently. For high traffic or multiple data-service instances, move manifests and album submissions to a transactional database and add distributed locking and a durable processing queue; the current object adapters use process-local locks.

Google Drive requires OAuth credentials for an account with access to the selected folder. A shared folder URL is not sufficient for server authentication. Railway Bucket configuration requires the endpoint URL, bucket name, region, access key, and secret key. Keep these values in Railway's encrypted service variables, not in the frontend or source control.
