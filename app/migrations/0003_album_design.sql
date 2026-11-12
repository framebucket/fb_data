-- Auto-designed album spreads generated from a customer's album selection.
CREATE TABLE IF NOT EXISTS fb_data.album_design
(
    design_id         TEXT          PRIMARY KEY,
    event_id          TEXT          NOT NULL REFERENCES fb_data.event (event_id) ON DELETE CASCADE,
    submission_id     TEXT          NOT NULL,
    photographer_id   TEXT          NOT NULL,
    version           INTEGER       NOT NULL,
    size_label        TEXT          NOT NULL,
    height_in         NUMERIC(5, 2) NOT NULL,
    width_in          NUMERIC(5, 2) NOT NULL,
    sheets            INTEGER       NOT NULL CHECK (sheets BETWEEN 1 AND 120),
    dpi               INTEGER       NOT NULL DEFAULT 300,
    seed              BIGINT        NOT NULL,
    status            TEXT          NOT NULL CHECK (status IN ('QUEUED', 'PROCESSING', 'READY', 'FAILED')),
    progress          INTEGER       NOT NULL DEFAULT 0,
    spreads           JSONB,
    jpeg_zip_key      TEXT,
    psd_zip_key       TEXT,
    gallery_folder_id TEXT,
    error             TEXT,
    created_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
    ready_at          TIMESTAMPTZ,
    UNIQUE (event_id, version)
);

CREATE INDEX IF NOT EXISTS album_design_event_idx ON fb_data.album_design (event_id, created_at DESC);
