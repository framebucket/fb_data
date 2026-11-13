-- Editing presets (Lightroom .xmp/.lrtemplate or .cube LUTs, compiled to a 3D LUT) that the service picks
-- from per photo, and reference album spreads whose layouts the album designer follows.
CREATE TABLE IF NOT EXISTS fb_data.studio_preset
(
    preset_id        TEXT        PRIMARY KEY,
    photographer_id  TEXT        NOT NULL,
    name             TEXT        NOT NULL,
    filename         TEXT        NOT NULL,
    kind             TEXT        NOT NULL CHECK (kind IN ('xmp', 'lrtemplate', 'cube')),
    scene            TEXT        NOT NULL DEFAULT 'any' CHECK (scene IN ('any', 'outdoor', 'indoor', 'night', 'portrait')),
    lut              BYTEA       NOT NULL,
    summary          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_studio_preset_owner ON fb_data.studio_preset (photographer_id, created_at);

ALTER TABLE fb_data.studio_style ADD COLUMN IF NOT EXISTS use_presets BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE fb_data.media ADD COLUMN IF NOT EXISTS preset_name TEXT;
ALTER TABLE fb_data.media ADD COLUMN IF NOT EXISTS scene TEXT;

CREATE TABLE IF NOT EXISTS fb_data.album_reference
(
    reference_id     TEXT        PRIMARY KEY,
    photographer_id  TEXT        NOT NULL,
    filename         TEXT        NOT NULL,
    name             TEXT        NOT NULL,
    thumb            BYTEA       NOT NULL,
    layout           JSONB       NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_album_reference_owner ON fb_data.album_reference (photographer_id, created_at);
