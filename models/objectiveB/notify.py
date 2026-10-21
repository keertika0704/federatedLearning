"""
Objective B notification hook. Called by detect.run_scan for every NEW or ESCALATED alert.

In-app delivery needs nothing here: the alert is already in the store, and the frontend polls /api/outbreak/alerts.
Optionally, set OUTBREAK_WEBHOOK_URL in .env to also POST each alert to a chat / paging / email-gateway webhook:
    {"event": "outbreak.new" | "outbreak.escalated", "alert": {...}}
The webhook is off by default. Alerts hold aggregate counts only, so nothing patient-level is sent.
"""

import logging
import os

import httpx

log = logging.getLogger("uvicorn.error")
WEBHOOK_TIMEOUT = 5.0


def notify(alert, change="new"):
    """Never raises: a failing webhook must not break a scan. Returns {"logged": True, "webhook": sent | None}."""
    log.warning("OUTBREAK %s [%s] %s", change.upper(), alert["severity"], alert["title"])
    url = (os.getenv("OUTBREAK_WEBHOOK_URL") or "").strip()
    if not url:
        return {"logged": True, "webhook": None}
    try:
        r = httpx.post(url, json={"event": f"outbreak.{change}", "alert": alert}, timeout=WEBHOOK_TIMEOUT)
        r.raise_for_status()
        return {"logged": True, "webhook": True}
    except Exception as e:                                   # network error, bad status, bad URL ...
        log.error("Outbreak webhook failed: %s", e)
        return {"logged": True, "webhook": False}
