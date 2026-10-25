CREATE TABLE IF NOT EXISTS users (
    id            TEXT    PRIMARY KEY,
    password_hash TEXT    NOT NULL,
    sex           TEXT,
    age           INTEGER,
    height_cm     REAL,
    weight_kg     REAL,
    activity      TEXT,
    goal          TEXT,
    tdee          INTEGER,
    daily_target  INTEGER,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

ALTER TABLE meals ADD COLUMN user_id TEXT REFERENCES users(id);
CREATE INDEX IF NOT EXISTS idx_meals_user ON meals(user_id);
