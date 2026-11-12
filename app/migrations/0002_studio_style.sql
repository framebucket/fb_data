-- Per-studio editing look (learned from reference photos) and logo stamping preferences.
CREATE TABLE IF NOT EXISTS fb_data.studio_style
(
    photographer_id  TEXT        PRIMARY KEY,
    profile          JSONB,
    reference_count  INTEGER     NOT NULL DEFAULT 0,
    baseline         JSONB,
    strength         INTEGER     NOT NULL DEFAULT 80 CHECK (strength BETWEEN 0 AND 100),
    apply_style      BOOLEAN     NOT NULL DEFAULT TRUE,
    stamp_logo       BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
