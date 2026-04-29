"""
SQLite database for preset reference images.

Stores 10 images per category from training_data/reference.
"""

import sqlite3
import random
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = Path(__file__).parent.parent / "data" / "references.db"
TRAINING_REF_DIR = Path(__file__).parent.parent / "test_images" / "training_data" / "reference"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

CATEGORY_MAP = {
    "1. Natural-Clean-Film": "Natural Clean Film",
    "2. Moody-Cinematic": "Moody Cinematic",
    "3. Bright-Airy-Cream-Whites": "Bright Airy",
}

PER_CATEGORY_LIMIT = 3


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preset_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            display_name TEXT NOT NULL,
            category TEXT NOT NULL,
            folder_name TEXT NOT NULL,
            UNIQUE(filename, folder_name)
        )
    """)
    conn.commit()
    conn.close()


def seed_db(per_category: int = PER_CATEGORY_LIMIT):
    """Pick `per_category` images from each category folder and insert into DB."""
    init_db()
    conn = get_db()

    # Check if already seeded
    count = conn.execute("SELECT COUNT(*) FROM preset_references").fetchone()[0]
    if count > 0:
        conn.close()
        return

    for folder_name, display_category in CATEGORY_MAP.items():
        folder_path = TRAINING_REF_DIR / folder_name
        if not folder_path.exists():
            continue

        images = [
            f.name for f in sorted(folder_path.iterdir())
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        selected = images[:per_category] if len(images) <= per_category else random.sample(images, per_category)

        for fname in selected:
            display_name = Path(fname).stem.replace("_", " ").replace("-", " ")
            # Truncate long names
            if len(display_name) > 30:
                display_name = display_name[:27] + "..."
            conn.execute(
                "INSERT OR IGNORE INTO preset_references (filename, display_name, category, folder_name) VALUES (?, ?, ?, ?)",
                (fname, display_name, display_category, folder_name),
            )

    conn.commit()
    conn.close()


def get_all_presets() -> List[Dict]:
    init_db()
    seed_db()
    conn = get_db()
    allowed = tuple(CATEGORY_MAP.values())
    placeholders = ",".join("?" * len(allowed))
    rows = conn.execute(
        f"SELECT * FROM preset_references WHERE category IN ({placeholders}) "
        f"ORDER BY category, display_name",
        allowed,
    ).fetchall()
    conn.close()

    # Limit to PER_CATEGORY_LIMIT per category (in case DB was seeded earlier with more)
    trimmed: List[Dict] = []
    counts: Dict[str, int] = {}
    for r in rows:
        d = dict(r)
        cat = d["category"]
        if counts.get(cat, 0) >= PER_CATEGORY_LIMIT:
            continue
        counts[cat] = counts.get(cat, 0) + 1
        trimmed.append(d)
    return trimmed


def get_presets_by_category(category: str) -> List[Dict]:
    init_db()
    seed_db()
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM preset_references WHERE category = ? ORDER BY display_name LIMIT ?",
        (category, PER_CATEGORY_LIMIT),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_categories() -> List[str]:
    init_db()
    seed_db()
    conn = get_db()
    allowed = tuple(CATEGORY_MAP.values())
    placeholders = ",".join("?" * len(allowed))
    rows = conn.execute(
        f"SELECT DISTINCT category FROM preset_references "
        f"WHERE category IN ({placeholders}) ORDER BY category",
        allowed,
    ).fetchall()
    conn.close()
    return [r["category"] for r in rows]


def reseed_db(per_category: int = PER_CATEGORY_LIMIT):
    """Force reseed — drops existing data."""
    init_db()
    conn = get_db()
    conn.execute("DELETE FROM preset_references")
    conn.commit()
    conn.close()
    seed_db(per_category)
