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