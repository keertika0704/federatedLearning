macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
.venv/bin/uvicorn backend.main:app --reload --port 8000
```

Windows (PowerShell):
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\.venv\Scripts\uvicorn.exe backend.main:app --reload --port 8000
```

Windows (Command Prompt):
```cmd
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
.venv\Scripts\uvicorn.exe backend.main:app --reload --port 8000
```
## Objective B: outbreak detection

Flags a disease (or a cluster of similar atypical presentations) that is rising at several hospitals at once.
Each hospital node shares only daily counts (`node.py`); the central detector (`detect.py`) never sees patient rows.
Run from the project root, with the venv active:

```bash
python models/objectiveB/node.py --build      # per-hospital daily summaries -> data/outbreak.db
python models/objectiveB/train.py             # calibrate thresholds, print held-out metrics
python models/objectiveB/detect.py --as-of 2025-06-26   # replay a day from the terminal
```

The app serves it under `/api/outbreak/*` and the **Surveillance** tab (replay a day, acknowledge alerts, hospital grid).
Optional: set `OUTBREAK_WEBHOOK_URL` in `.env` to also POST each new alert to a webhook (off by default).
