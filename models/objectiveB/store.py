"""
SQLite store for objective B (stdlib only). Two tables:

    node_summaries   what each hospital node shares: one row per (hospital, day, disease) with the case count and
                     the 28 symptom counts among those cases. disease_id "ATYP" holds the cases that objective A
                     flagged as atypical / novel. No patient rows ever reach this table.
    alerts           one row per detected outbreak, with its lifecycle (open -> acknowledged -> resolved).

Default file: data/outbreak.db (gitignored). Override with the OUTBREAK_DB environment variable.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in HERE.parents if (p / "models").is_dir())
DB_PATH = Path(os.getenv("OUTBREAK_DB") or PROJECT_ROOT / "data" / "outbreak.db")

SYMPTOMS = [f"S{i:03d}" for i in range(1, 29)]
ATYPICAL_ID = "ATYP"                       # pseudo-disease row: cases flagged atypical / novel by objective A
SUMMARY_COLUMNS = ["hospital_id", "region", "day", "disease_id", "n", *SYMPTOMS]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS node_summaries (
    hospital_id TEXT NOT NULL, region TEXT NOT NULL, day TEXT NOT NULL, disease_id TEXT NOT NULL,
    n INTEGER NOT NULL, {", ".join(f"{s} INTEGER NOT NULL DEFAULT 0" for s in SYMPTOMS)},
    PRIMARY KEY (hospital_id, day, disease_id)
);
CREATE INDEX IF NOT EXISTS ix_summaries_day ON node_summaries(day);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_key TEXT NOT NULL,             -- disease_id, or "UNKNOWN" for the possible-novel-outbreak path
    status TEXT NOT NULL DEFAULT 'open',   -- open | acknowledged | resolved
    severity TEXT NOT NULL,                -- watch | warning | critical
    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    payload TEXT NOT NULL                  -- full alert as JSON
);
CREATE INDEX IF NOT EXISTS ix_alerts_status ON alerts(status);
"""

SEVERITY_RANK = {"watch": 0, "warning": 1, "critical": 2}
MERGE_WITHIN_DAYS = 14                     # a new signal for the same disease joins an alert seen this recently


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path=None):
        self.path = Path(path or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            with c:                        # commit / rollback
                yield c
        finally:
            c.close()

    # ------------------------------------------------------------ node summaries
    def put_summaries(self, df):
        """Insert or replace summary rows (a DataFrame with SUMMARY_COLUMNS)."""
        rows = df[SUMMARY_COLUMNS].astype(object).values.tolist()
        marks = ",".join("?" * len(SUMMARY_COLUMNS))
        with self._conn() as c:
            c.executemany(f"INSERT OR REPLACE INTO node_summaries ({','.join(SUMMARY_COLUMNS)}) VALUES ({marks})", rows)
        return len(rows)

    def clear_summaries(self):
        with self._conn() as c:
            c.execute("DELETE FROM node_summaries")

    def load_summaries(self, as_of=None):
        q, args = "SELECT * FROM node_summaries", []
        if as_of is not None:
            q, args = q + " WHERE day <= ?", [str(as_of)[:10]]
        with self._conn() as c:
            df = pd.read_sql_query(q, c, params=args)
        df["day"] = pd.to_datetime(df["day"])
        return df

    def day_range(self):
        with self._conn() as c:
            r = c.execute("SELECT MIN(day) lo, MAX(day) hi, COUNT(DISTINCT hospital_id) h FROM node_summaries").fetchone()
        return (r["lo"], r["hi"], r["h"]) if r["lo"] else (None, None, 0)

    def last_shared_day(self, min_hospitals=2):
        """Latest day that at least `min_hospitals` nodes reported, or None. A lone node (e.g. this clinic's live
        assessments) cannot be cross-checked, so the dashboard and default scans use this day, not the latest."""
        with self._conn() as c:
            r = c.execute("SELECT MAX(day) FROM (SELECT day FROM node_summaries GROUP BY day "
                          "HAVING COUNT(DISTINCT hospital_id) >= ?)", (min_hospitals,)).fetchone()
        return r[0]

    # ------------------------------------------------------------ alerts
    @staticmethod
    def _row(r):
        a = json.loads(r["payload"])
        a.update(id=r["id"], status=r["status"], severity=r["severity"], first_seen=r["first_seen"],
                 last_seen=r["last_seen"], updated_at=r["updated_at"])
        return a

    def upsert_alert(self, alert, rebuild=None):
        """Merge `alert` into a live alert for the same disease seen within MERGE_WITHIN_DAYS, else insert it.
        `rebuild(alert)` (detect.describe) refreshes regions / scope / severity / title / message from the merged sites.
        Returns (alert with id/status, change) where change is 'new', 'escalated', 'updated', or 'known' (the day is
        already inside a resolved outbreak of the same disease; nothing is stored)."""
        day = alert["day"]
        floor = (pd.Timestamp(day) - pd.Timedelta(days=MERGE_WITHIN_DAYS)).strftime("%Y-%m-%d")
        with self._conn() as c:
            r = c.execute("SELECT * FROM alerts WHERE disease_key=? AND status IN ('open','acknowledged') "
                          "AND last_seen>=? ORDER BY id DESC LIMIT 1", (alert["disease_key"], floor)).fetchone()
            if r is None:
                # Replaying a day an already-resolved outbreak covers finds the same outbreak again: it is in the history
                # once, so report it as known instead of adding a duplicate.
                past = c.execute("SELECT * FROM alerts WHERE disease_key=? AND status='resolved' AND first_seen<=? "
                                 "AND last_seen>=? ORDER BY id DESC LIMIT 1", (alert["disease_key"], day, day)).fetchone()
                if past is not None:
                    return self._row(past), "known"
                cur = c.execute("INSERT INTO alerts (disease_key,status,severity,first_seen,last_seen,created_at,updated_at,"
                                "payload) VALUES (?,?,?,?,?,?,?,?)",
                                (alert["disease_key"], "open", alert["severity"], day, day, _now(), _now(), json.dumps(alert)))
                r = c.execute("SELECT * FROM alerts WHERE id=?", (cur.lastrowid,)).fetchone()
                return self._row(r), "new"
            old = json.loads(r["payload"])
            hospitals = {s["hospital_id"] for s in old["sites"]}
            merged = dict(alert)
            # keep every site seen during the episode; the current evidence replaces the old for sites seen again
            by_id = {s["hospital_id"]: s for s in old["sites"]}
            by_id.update({s["hospital_id"]: s for s in alert["sites"]})
            merged["sites"] = list(by_id.values())
            if rebuild:
                merged.update(rebuild(merged))
            else:
                merged["regions"] = sorted({s["region"] for s in merged["sites"]})
            sev = max(r["severity"], merged["severity"], key=SEVERITY_RANK.get)
            merged["severity"] = sev
            # status: an acknowledged alert stays acknowledged unless it escalates or spreads to new sites
            escalated = SEVERITY_RANK[sev] > SEVERITY_RANK[r["severity"]] or bool({s["hospital_id"] for s in alert["sites"]} - hospitals)
            status = "open" if escalated else r["status"]
            c.execute("UPDATE alerts SET status=?, severity=?, last_seen=?, updated_at=?, payload=? WHERE id=?",
                      (status, sev, max(day, r["last_seen"]), _now(), json.dumps(merged), r["id"]))
            return self._row(c.execute("SELECT * FROM alerts WHERE id=?", (r["id"],)).fetchone()), \
                ("escalated" if escalated else "updated")

    def resolve_stale(self, as_of, after_days):
        """Close live alerts with no new signal for `after_days`. Returns how many were closed."""
        floor = (pd.Timestamp(as_of) - pd.Timedelta(days=after_days)).strftime("%Y-%m-%d")
        with self._conn() as c:
            return c.execute("UPDATE alerts SET status='resolved', updated_at=? WHERE status IN ('open','acknowledged') "
                             "AND last_seen<?", (_now(), floor)).rowcount

    def list_alerts(self, status=None, limit=100):
        q, args = "SELECT * FROM alerts", []
        if status:
            q, args = q + " WHERE status=?", [status]
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY last_seen DESC, id DESC LIMIT ?", [*args, limit]).fetchall()
        return [self._row(r) for r in rows]

    def acknowledge(self, alert_id):
        """open -> acknowledged. Returns the alert, or None when the id is unknown."""
        with self._conn() as c:
            c.execute("UPDATE alerts SET status='acknowledged', updated_at=? WHERE id=? AND status='open'", (_now(), alert_id))
            r = c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return self._row(r) if r else None

    def clear_alerts(self):
        with self._conn() as c:
            c.execute("DELETE FROM alerts")
