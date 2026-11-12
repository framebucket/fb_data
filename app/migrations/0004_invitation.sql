-- Customer wedding invitations (one per event): details, optional couple photo, design batches, final video.
CREATE TABLE IF NOT EXISTS fb_data.invitation
(
    event_id         TEXT        PRIMARY KEY,
    photographer_id  TEXT        NOT NULL,
    customer_id      TEXT        NOT NULL,
    details          JSONB       NOT NULL,
    photo_key        TEXT,
    photo_faces      JSONB,
    designs          JSONB,
    batches          INTEGER     NOT NULL DEFAULT 0,
    chosen           JSONB,
    status           TEXT        NOT NULL DEFAULT 'DRAFT' CHECK (status IN ('DRAFT', 'RENDERING', 'READY', 'FAILED')),
    progress         INTEGER     NOT NULL DEFAULT 0,
    poster_key       TEXT,
    video_key        TEXT,
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    ready_at         TIMESTAMPTZ
);
