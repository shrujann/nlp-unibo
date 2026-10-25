CREATE TABLE IF NOT EXISTS meals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    total_calories REAL NOT NULL,
    ingredients_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_meals_created_at ON meals(created_at);
