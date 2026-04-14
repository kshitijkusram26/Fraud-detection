-- Create separate database for MLflow
SELECT 'CREATE DATABASE mlflow_db'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'mlflow_db'
)\gexec

GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO fraud_user;

CREATE Table if not exists predictions(
    id                serial primary key,
    transaction_id    varchar(64) not null UNIQUE,
    created_at        TIMESTAMPTZ  not null DEFAULT now(),

    amount            numeric(12,2) not null,
    time_seconds      numeric(12,2),
    v1                numeric(10,6),
    v2                numeric(10,6),
    v3                numeric(10,6),
    v4                numeric(10,6),
    v14               numeric(10,6),
    v17               numeric(10,6),

    fraud_probability  numeric(6,5) not null,
    prediction         varchar(16)  not null,
    confidence         VARCHAR(16),
    model_version      varchar(32),
    latency_ms         numeric(8,2),

    actual_label       SMALLINT  DEFAULT null

);


-- 2. BATCH JOBS--
CREATE Table  if not exists batch_jobs(
    id           serial primary key,
    job_id       varchar(64)  not null UNIQUE,
    created_at   TIMESTAMPTZ   not null DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status       varchar(16)   not null DEFAULT 'PENDING',
    total_rows   INTEGER,
    fraud_count  INTEGER,
    legit_count  INTEGER,
    filename     VARCHAR(256)
);


--3.Model versions--
DROP TABLE IF EXISTS model_versions CASCADE;
CREATE Table if not exists model_versions(
    id        serial primary key,
    registered_at   TIMESTAMPTZ    not null DEFAULT now(),
    Version         VARCHAR(32)  not null,
    run_id          VARCHAR(64),
    pr_auc          numeric(6,5),
    roc_auc         numeric(6,5),
    f1_score        numeric(6,5),
    threshold       numeric(6,5),
    is_active       BOOLEAN   not null DEFAULT false,
    notes           text
);




--4. Daily Stats--
CREATE Table if not exists daily_stats(
    stat_date      date      primary key,
    total_txns     INTEGER   not null DEFAULT 0,
    fraud_count    INTEGER   not null DEFAULT 0,
    avg_amount     numeric(12,2),
    avg_fraud_prob  numeric(6,5),
    updated_at      TIMESTAMPTZ   not null DEFAULT now()
);



-- Indexes --
CREATE INDEX if NOT EXISTS idx_predictions_created on predictions  (created_at DESC);
CREATE INDEX if not EXISTS idx_predictions_predcition  on predictions (prediction);
CREATE INDEX if not EXISTS idx_predictions_prob  on predictions (fraud_probability DESC);

-- SEED: --
INSERT into model_versions (Version,is_active,threshold,notes)
values('v0.0-placeholder',TRUE,0.50,'Placeholder until first training run')
on conflict do nothing;

SELECT 'Schema intialised Successfully' as status;