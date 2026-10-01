"""Combined API application for disease prediction and treatment recommendation."""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.disease_prediction_api import router as disease_router
from backend.security import API_KEY, UI_SESSION_TOKEN
from backend.treatment_prediction_api import router as treatment_router

app = FastAPI(title="Patient Disease and Treatment API", version="1.0.0", docs_url=None)
app.include_router(disease_router)
app.include_router(treatment_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.post("/ui/session", status_code=204, include_in_schema=False)
def create_ui_session(request: Request, response: Response):
    if request.client is None or request.client.host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Automatic UI sessions are local-only")
    response.set_cookie(
        "carepath_session",
        UI_SESSION_TOKEN,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        max_age=3600,
        path="/",
    )

DOCS_AUTOFILL = os.getenv("DOCS_AUTOFILL_KEY", "0") == "1"
LOCALHOST = {"127.0.0.1", "::1", "localhost"}


@app.get("/docs", include_in_schema=False)
def docs(request: Request):
    autofill = DOCS_AUTOFILL and API_KEY and request.client and request.client.host in LOCALHOST
    preauth = f"ui.preauthorizeApiKey('APIKeyHeader', {json.dumps(API_KEY)});" if autofill else ""
    return HTMLResponse(f"""<!doctype html>
<html><head><title>{app.title} - docs</title><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"></head>
<body><div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
	const ui = SwaggerUIBundle({{
		url: "{app.openapi_url}", dom_id: "#swagger-ui",
		persistAuthorization: true,
		onComplete: () => {{ {preauth} }}
	}});
</script></body></html>""")


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True),
    name="frontend",
)