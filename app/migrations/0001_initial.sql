-- FrameBucket data-service metadata. Owned by the Python service (schema fb_data);
-- the Java application's Flyway migrations never touch this schema.

CREATE TABLE fb_data.event
(
    event_id          TEXT        PRIMARY KEY,
    event_code        TEXT        NOT NULL,
    photographer_id   TEXT        NOT NULL,
    customer_id       TEXT        NOT NULL,
    next_photo_number INTEGER     NOT NULL DEFAULT 1 CHECK (next_photo_number >= 1),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE fb_data.folder
(
    event_id   TEXT        NOT NULL REFERENCES fb_data.event (event_id) ON DELETE CASCADE,
    folder_id  TEXT        NOT NULL,
    name       TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, folder_id)
);

CREATE UNIQUE INDEX ux_fb_data_folder_name ON fb_data.folder (event_id, lower(name));

-- status: UPLOADING -> QUEUED -> PROCESSING -> READY | PROCESSING_FAILED; READY -> SUBMITTED_FOR_ALBUM
CREATE TABLE fb_data.media
(
    photo_id            TEXT        PRIMARY KEY,
    event_id            TEXT        NOT NULL REFERENCES fb_data.event (event_id) ON DELETE CASCADE,
    photo_number        INTEGER,
    folder_id           TEXT        NOT NULL,
    media_type          TEXT        NOT NULL CHECK (media_type IN ('image', 'video')),
    original_filename   TEXT        NOT NULL,
    content_type        TEXT,
    original_key        TEXT        NOT NULL,
    processed_key       TEXT        NOT NULL,
    status              TEXT        NOT NULL,
    size_bytes          BIGINT,
    width               INTEGER,
    height              INTEGER,
    processed_bytes     BIGINT,
    watermarked         BOOLEAN     NOT NULL DEFAULT FALSE,
    ai_enhanced         BOOLEAN     NOT NULL DEFAULT FALSE,
    error               TEXT,
    album_submission_id TEXT,
    face_embeddings     JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ux_fb_data_media_event_number UNIQUE (event_id, photo_number)
);

CREATE INDEX ix_fb_data_media_event_folder ON fb_data.media (event_id, folder_id, photo_number);
CREATE INDEX ix_fb_data_media_uploading ON fb_data.media (created_at) WHERE status = 'UPLOADING';

CREATE TABLE fb_data.album_selection
(
    event_id    TEXT        NOT NULL REFERENCES fb_data.event (event_id) ON DELETE CASCADE,
    customer_id TEXT        NOT NULL,
    photo_ids   JSONB       NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, customer_id)
);

-- status: PREPARING -> READY | FAILED
CREATE TABLE fb_data.album_submission
(
    submission_id   TEXT        PRIMARY KEY,
    event_id        TEXT        NOT NULL REFERENCES fb_data.event (event_id) ON DELETE CASCADE,
    photographer_id TEXT        NOT NULL,
    customer_id     TEXT        NOT NULL,
    token_hash      TEXT        NOT NULL UNIQUE,
    archive_key     TEXT        NOT NULL,
    status          TEXT        NOT NULL,
    photo_ids       JSONB       NOT NULL,
    photo_numbers   JSONB       NOT NULL,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ready_at        TIMESTAMPTZ,
    expires_at      TIMESTAMPTZ NOT NULL
);

CREATE TABLE fb_data.face_guest
(
    event_id   TEXT        NOT NULL REFERENCES fb_data.event (event_id) ON DELETE CASCADE,
    guest_id   TEXT        NOT NULL,
    embedding  JSONB       NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, guest_id)
);

-- Durable work queue consumed by `python -m app.worker` with FOR UPDATE SKIP LOCKED.
CREATE TABLE fb_data.job
(
    id           BIGSERIAL   PRIMARY KEY,
    kind         TEXT        NOT NULL,
    payload      JSONB       NOT NULL,
    status       TEXT        NOT NULL DEFAULT 'QUEUED' CHECK (status IN ('QUEUED', 'RUNNING', 'DONE', 'FAILED')),
    attempts     INTEGER     NOT NULL DEFAULT 0,
    max_attempts INTEGER     NOT NULL DEFAULT 5,
    run_after    TIMESTAMPTZ NOT NULL DEFAULT now(),
    locked_by    TEXT,
    locked_at    TIMESTAMPTZ,
    last_error   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_fb_data_job_queued ON fb_data.job (run_after, id) WHERE status = 'QUEUED';
CREATE INDEX ix_fb_data_job_running ON fb_data.job (locked_at) WHERE status = 'RUNNING';
CREATE INDEX ix_fb_data_job_finished ON fb_data.job (updated_at) WHERE status IN ('DONE', 'FAILED');

-- Notifications for the Java application (face matches, album archives ready/failed).
CREATE TABLE fb_data.outbox
(
    id            BIGSERIAL   PRIMARY KEY,
    kind          TEXT        NOT NULL,
    event_id      TEXT        NOT NULL,
    payload       JSONB       NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_until TIMESTAMPTZ,
    delivered_at  TIMESTAMPTZ
);

CREATE INDEX ix_fb_data_outbox_pending ON fb_data.outbox (id) WHERE delivered_at IS NULL;
