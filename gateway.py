"""
Gateway server: exposes all decision-model services behind one public port.
Each model runs as its own standalone process (separate venv/deps, avoiding
package name collisions between them); this just proxies requests to the
right internal port.
"""
import os
os.environ.setdefault("USE_TF", "0")

import json
import urllib.request
import urllib.error
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Decision Model Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model_name -> internal service URL
BACKENDS = {
    "nox4b": "http://localhost:18076/api/nox4b/predict",
}


def _proxy(url: str, payload: Dict[str, Any]):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=e.code, detail=e.read().decode())
    except urllib.error.URLError as e:
        raise HTTPException(status_code=503, detail=f"backend unreachable: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "backends": list(BACKENDS.keys())}


@app.post("/api/nox4b/predict")
def nox4b_predict(req: Dict[str, Any]):
    return _proxy(BACKENDS["nox4b"], req)
