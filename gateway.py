"""
Gateway server: exposes all decision-model services behind one public port.
Each model runs as its own standalone process (separate venv/deps, avoiding
package name collisions between them); this just proxies requests to the
right internal port.

Prometheus metrics (ticket #253) are served on a separate port bound to
127.0.0.1 (METRICS_PORT, default 18090), so they are not reachable through the
public URL. The gateway sees every request, so all metrics are recorded here
and the model servers stay unchanged.
"""
import os
os.environ.setdefault("USE_TF", "0")

import json
import subprocess
import time
import urllib.request
import urllib.error
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from prometheus_client.core import REGISTRY, GaugeMetricFamily

METRICS_PORT = int(os.environ.get("METRICS_PORT", "18090"))

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
    "lux9b": "http://localhost:18077/api/lux9b/predict",
    "lfm25rlcd": "http://localhost:18074/api/lfm25rlcd/predict",
}

# Latency buckets span ~30 ms model calls up to the 120 s backend timeout.
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
                   1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0, 120.0)

REQUESTS = Counter(
    "decision_requests_total",
    "Requests handled by the gateway, by model and HTTP status code.",
    ["model", "code"],
)
LATENCY = Histogram(
    "decision_request_duration_seconds",
    "Gateway-side E2E latency per request, including time queued in the backend.",
    ["model"],
    buckets=LATENCY_BUCKETS,
)
IN_FLIGHT = Gauge(
    "decision_requests_in_flight",
    "Requests currently inside the gateway for this model (running + waiting).",
    ["model"],
)
MODEL_TIME = Histogram(
    "decision_model_compute_seconds",
    "Model compute time reported by the backend's elapsed_ms field (LFM only). "
    "Excludes queueing, so E2E minus this is roughly the wait.",
    ["model"],
    buckets=LATENCY_BUCKETS,
)
for _m in BACKENDS:
    IN_FLIGHT.labels(_m).set(0)


class ScrapeTimeCollector:
    """Backend liveness and GPU memory, read fresh on every scrape."""

    def collect(self):
        up = GaugeMetricFamily("decision_backend_up", "1 if the backend's /health answers, else 0.", labels=["model"])
        for model, url in BACKENDS.items():
            health = url.split("/api/")[0] + "/health"
            try:
                with urllib.request.urlopen(health, timeout=2) as r:
                    up.add_metric([model], 1.0 if r.status == 200 else 0.0)
            except Exception:
                up.add_metric([model], 0.0)
        yield up

        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except Exception:
            return
        used = GaugeMetricFamily("gpu_memory_used_bytes", "GPU memory in use (nvidia-smi).", labels=["gpu"])
        total = GaugeMetricFamily("gpu_memory_total_bytes", "GPU memory total (nvidia-smi).", labels=["gpu"])
        util = GaugeMetricFamily("gpu_utilization_percent", "GPU utilization (nvidia-smi).", labels=["gpu"])
        for line in out.strip().splitlines():
            idx, mem_used, mem_total, gpu_util = [x.strip() for x in line.split(",")]
            used.add_metric([idx], float(mem_used) * 1024 * 1024)
            total.add_metric([idx], float(mem_total) * 1024 * 1024)
            util.add_metric([idx], float(gpu_util))
        yield used
        yield total
        yield util


REGISTRY.register(ScrapeTimeCollector())


@app.on_event("startup")
def start_metrics_server():
    start_http_server(METRICS_PORT, addr="127.0.0.1")
    print(f"Prometheus metrics on http://127.0.0.1:{METRICS_PORT}/metrics")


def _proxy(model: str, payload: Dict[str, Any]):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        BACKENDS[model], data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    code = 500
    IN_FLIGHT.labels(model).inc()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
        code = 200
        if isinstance(result.get("elapsed_ms"), (int, float)):
            MODEL_TIME.labels(model).observe(result["elapsed_ms"] / 1000)
        return result
    except urllib.error.HTTPError as e:
        code = e.code
        body = e.read().decode()
        try:
            detail = json.loads(body).get("detail", body)
        except ValueError:
            detail = body
        raise HTTPException(status_code=e.code, detail=detail)
    except TimeoutError as e:
        code = 504
        raise HTTPException(status_code=504, detail=f"backend timed out: {e}")
    except urllib.error.URLError as e:
        code = 504 if isinstance(e.reason, TimeoutError) else 503
        raise HTTPException(status_code=code, detail=f"backend unreachable: {e}")
    finally:
        LATENCY.labels(model).observe(time.perf_counter() - t0)
        REQUESTS.labels(model, str(code)).inc()
        IN_FLIGHT.labels(model).dec()


@app.get("/health")
def health():
    return {"status": "ok", "backends": list(BACKENDS.keys())}


@app.post("/api/nox4b/predict")
def nox4b_predict(req: Dict[str, Any]):
    return _proxy("nox4b", req)


@app.post("/api/lux9b/predict")
def lux9b_predict(req: Dict[str, Any]):
    return _proxy("lux9b", req)


@app.post("/api/lfm25rlcd/predict")
def lfm25rlcd_predict(req: Dict[str, Any]):
    return _proxy("lfm25rlcd", req)
