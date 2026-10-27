"""
Objective B demo - push one more week of summaries for EVERY hospital and disease, with an outbreak hidden in them.

Normal days are drawn at each hospital's own recent rate (last 28 days in the store), with each disease's usual
symptom mix, so the Surveillance grid is full instead of empty. On top of that, the chosen disease surges at the chosen
hospitals. Days start right after the last day in the store, so no real row is replaced.
Standard library only; the backend must be running.

    python models/objectiveB/demo_outbreak.py --dry-run     # print what would be sent, send nothing
    python models/objectiveB/demo_outbreak.py               # push the week, then run a scan on its last day
    python models/objectiveB/demo_outbreak.py --cleanup     # remove every row and alert after 2025-12-30 (the real data's last day)

Options: --disease D004 --hospitals H001,H002,H003 --factor 2.7 --days 7 --seed 7 --url http://localhost:8000
The API key comes from OBJECTIVE_E_API_KEY (environment or .env). Needs a hospital's rows in the store: run
node.py --build first.
"""

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in HERE.parents if (p / "models").is_dir())
DB_PATH = Path(os.getenv("OUTBREAK_DB") or PROJECT_ROOT / "data" / "outbreak.db")
SYMPTOMS = [f"S{i:03d}" for i in range(1, 29)]
BASELINE_DAYS = 28
REAL_DATA_END = "2025-12-30"                # last day of the generated data; --cleanup removes everything after it


def poisson(rng, mean):
    """Knuth's method; fine for the small daily counts here."""
    limit, k, p = math.exp(-mean), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def load_baseline(db_path):
    """(last day in the store, {hospital: region}, {(hospital, disease): mean cases/day}, {disease: {symptom: share}})."""
    c = sqlite3.connect(db_path)
    try:
        last = c.execute("SELECT MAX(day) FROM node_summaries").fetchone()[0]
        if last is None:
            sys.exit("No node summaries in the store. Build them first:  python models/objectiveB/node.py --build")
        since = (date.fromisoformat(last) - timedelta(days=BASELINE_DAYS - 1)).isoformat()
        regions = dict(c.execute("SELECT DISTINCT hospital_id, region FROM node_summaries"))
        rates = {(h, d): n / BASELINE_DAYS for h, d, n in
                 c.execute("SELECT hospital_id, disease_id, SUM(n) FROM node_summaries WHERE day >= ? GROUP BY 1, 2", (since,))}
        sums = ", ".join(f"SUM({s})" for s in SYMPTOMS)
        shares = {}
        for row in c.execute(f"SELECT disease_id, SUM(n), {sums} FROM node_summaries WHERE day >= ? GROUP BY 1", (since,)):
            d, total = row[0], row[1]
            shares[d] = {s: v / total for s, v in zip(SYMPTOMS, row[2:]) if total and v} if total else {}
        return last, regions, rates, shares
    finally:
        c.close()


def build_payloads(db_path, disease="D004", surge_hospitals=("H001", "H002", "H003"), factor=2.7, days=7, seed=7):
    """One summaries batch per hospital: {hospital_id: {"hospital_id", "region", "rows"}}, plus the first and last demo day."""
    last, regions, rates, shares = load_baseline(db_path)
    rng = random.Random(seed)
    first = date.fromisoformat(last) + timedelta(days=1)
    hospitals = sorted(regions)
    diseases = sorted({d for _, d in rates})
    batches = {}
    for h in hospitals:
        rows = []
        for i in range(days):
            day = first + timedelta(days=i)
            day_rows = []
            for d in diseases:
                mean = rates.get((h, d), 0.0) * (factor if d == disease and h in surge_hospitals else 1.0)
                n = poisson(rng, mean)
                if n == 0:
                    continue
                sym = {s: sum(rng.random() < p for _ in range(n)) for s, p in shares.get(d, {}).items()}
                day_rows.append({"day": day.isoformat(), "disease_id": d, "n": n, "symptoms": {s: v for s, v in sym.items() if v}})
            if not day_rows:                                   # a day with no row would read as "did not report"
                day_rows.append({"day": day.isoformat(), "disease_id": disease, "n": 0, "symptoms": {}})
            rows += day_rows
        batches[h] = {"hospital_id": h, "region": regions[h], "rows": rows}
    return batches, first, first + timedelta(days=days - 1)


def api_key():
    key = (os.getenv("OBJECTIVE_E_API_KEY") or "").strip()
    env = PROJECT_ROOT / ".env"
    if not key and env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("OBJECTIVE_E_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        sys.exit("Set OBJECTIVE_E_API_KEY (environment or .env).")
    return key


def post(url, key, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"X-API-Key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"{url}: HTTP {e.code} {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach {url}: {e.reason}. Is the backend running?")


def cleanup(db_path, real_until):
    c = sqlite3.connect(db_path)
    try:
        with c:
            rows = c.execute("DELETE FROM node_summaries WHERE day > ?", (real_until,)).rowcount
            alerts = c.execute("DELETE FROM alerts WHERE last_seen > ?", (real_until,)).rowcount
    finally:
        c.close()
    print(f"Removed {rows} summary rows after {real_until} and {alerts} alert(s) seen after it.")


def main():
    ap = argparse.ArgumentParser(description="Push a week of data with an outbreak hidden in it")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--disease", default="D004", help="disease that surges (default D004, influenza)")
    ap.add_argument("--hospitals", default="H001,H002,H003", help="hospitals where it surges")
    ap.add_argument("--factor", type=float, default=2.7, help="surge size, times the hospital's normal rate")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="print the data, send nothing")
    ap.add_argument("--cleanup", action="store_true", help="delete every summary row and alert after --real-until, from the store file")
    ap.add_argument("--real-until", default=REAL_DATA_END, help=f"last day of real data (default {REAL_DATA_END})")
    args = ap.parse_args()

    if args.cleanup:
        cleanup(DB_PATH, args.real_until)
        return
    batches, first, last = build_payloads(DB_PATH, args.disease, tuple(args.hospitals.split(",")), args.factor, args.days, args.seed)
    surge = set(args.hospitals.split(","))
    print(f"{first} to {last}: {len(batches)} hospitals, {sum(len(b['rows']) for b in batches.values())} rows. "
          f"{args.disease} surges x{args.factor} at {', '.join(sorted(surge))}.\n")
    print(f"{'hospital':9s}{'region':9s}{'cases/week':>11s}{args.disease + ' cases':>12s}")
    for h, b in batches.items():
        total = sum(r["n"] for r in b["rows"])
        one = sum(r["n"] for r in b["rows"] if r["disease_id"] == args.disease)
        print(f"{h:9s}{b['region']:9s}{total:11d}{one:12d}{'   <- surge' if h in surge else ''}")
    if args.dry_run:
        print("\nDry run: nothing sent. Payloads, as the API receives them:")
        print(json.dumps(batches[sorted(surge)[0]], indent=1)[:1500], "...")
        return

    key = api_key()
    for h, b in batches.items():
        res = post(f"{args.url}/api/outbreak/summaries", key, b)
        print(f"pushed {h}: stored {res['stored']} rows")
    res = post(f"{args.url}/api/outbreak/scan", key, {"as_of": last.isoformat()})
    print(f"\nScan as of {res['as_of']}: {len(res['alerts'])} alert(s), {res['new']} new, {res['resolved']} resolved")
    for a in res["alerts"]:
        print(f"  [{a['severity']}] {a['title']}")


if __name__ == "__main__":
    main()
