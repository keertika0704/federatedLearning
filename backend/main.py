"""Combined API application for disease prediction and treatment recommendation."""

import json
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from backend.disease_prediction_api import API_KEY, router as disease_router
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