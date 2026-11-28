-- Migration 20: Incident Classifications (fusion-v2)
-- Stores advisory multi-modal report classifications combining citizen label, text and image models.

CREATE TABLE IF NOT EXISTS incident_classifications (
    classification_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id         UUID NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
    version             INT  NOT NULL,
    is_current          BOOLEAN NOT NULL DEFAULT TRUE,
    status              TEXT NOT NULL
        CHECK (status IN ('QUEUED','RUNNING','COMPLETED','DEGRADED','FAILED')),
    trigger_event       TEXT NOT NULL
        CHECK (trigger_event IN ('SUBMITTED','PHOTO_ADDED','INFO_REPLY','MANUAL_RERUN','SWEEP')),
    pipeline_version    TEXT NOT NULL DEFAULT 'fusion-v2',
    input_hash          CHAR(64) NOT NULL,   -- sha256(description + sorted photo sha256s + user label + pipeline_version)
    attempts            INT  NOT NULL DEFAULT 0,
    next_attempt_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    user_label          TEXT CHECK (user_label IN ('FLOOD','FIRE','CYCLONE')),
    text_result         JSONB,
    image_results       JSONB,               -- one entry per photo
    fused_probabilities JSONB,               -- {FLOOD,FIRE,CYCLONE,OTHER}

    final_label         TEXT CHECK (final_label IN ('FLOOD','FIRE','CYCLONE','UNDETERMINED')),
    label_source        TEXT CHECK (label_source IN ('FUSED','ML_ONLY','USER_ONLY')),
    model_confidence    NUMERIC(4,3) CHECK (model_confidence BETWEEN 0 AND 1),
    confidence_level    TEXT CHECK (confidence_level IN ('HIGH','MEDIUM','LOW')),
    margin              NUMERIC(4,3) CHECK (margin BETWEEN 0 AND 1),
    agreement           TEXT CHECK (agreement IN ('AGREE','MAJORITY','CONFLICT','SINGLE_SOURCE')),
    human_review_required BOOLEAN,
    review_reasons      TEXT[] NOT NULL DEFAULT '{}',
    mapped_scenario     scenario_type,       -- NULL only when UNDETERMINED
    models_used         JSONB,
    error_code          TEXT,

    reviewed_by         UUID REFERENCES users(user_id) ON DELETE SET NULL,
    reviewed_label      TEXT CHECK (reviewed_label IN
        ('FLOOD','FIRE','CYCLONE','OTHER_DISASTER','NOT_A_DISASTER')),
    reviewed_at         TIMESTAMPTZ,
    review_note         TEXT CHECK (char_length(review_note) <= 2000),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ,
    UNIQUE (incident_id, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_classification_current
    ON incident_classifications (incident_id) WHERE is_current;
CREATE UNIQUE INDEX IF NOT EXISTS uq_classification_input
    ON incident_classifications (incident_id, input_hash, pipeline_version)
    WHERE status IN ('QUEUED','RUNNING','COMPLETED');
CREATE INDEX IF NOT EXISTS idx_classification_queue
    ON incident_classifications (next_attempt_at) WHERE status = 'QUEUED';
CREATE INDEX IF NOT EXISTS idx_classification_review
    ON incident_classifications (human_review_required, final_label) WHERE is_current;
