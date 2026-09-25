"""
Gateway server: exposes all decision-model services behind one public port.
Each model runs as its own standalone process (separate venv/deps, avoiding
package name collisions between them); this just proxies requests to the
right internal port.

Prometheus metrics (ticket #253) are served at GET /metrics on this same port.
Scrapes must send "Authorization: Bearer <token>", where the token is read from
METRICS_TOKEN_FILE; anything else gets 404, so the public URL shows nothing.
If the token file is missing or empty, /metrics is off. The gateway sees every
request, so all metrics are recorded here and the model servers stay unchanged.

Request count, latency and in-flight are recorded in an HTTP middleware, which
runs before FastAPI hands a sync endpoint to its 40-thread pool. So requests
waiting for a free thread are counted too.

The same middleware sheds load with HTTP 429 (Retry-After: 1) when a model is
actually overloaded. Each model has an adaptive concurrency limit (AIMD):
- about once a second, look at the P90 gateway latency of its recent
  successful requests;
- P90 above LATENCY_TARGET_SECONDS (default 2.0): limit *= 0.8;
- P90 under target and the limit was actually reached: limit += 1;
- limit stays within [CONCURRENCY_MIN, CONCURRENCY_MAX] (default 1..32).
A request over the limit gets 429 straight away, without taking a gateway
thread. A fixed request count doesn't work here: one long input can keep the
GPU at 100% on its own, while many short ones fit easily. Latency covers both.
Starting limits are the load-test thresholds (docs/decision-endpoints-load-
test.html). CONCURRENCY_LIMITS="laya=0" turns shedding off for a model.
"""
import os
os.environ.setdefault("USE_TF", "0")

import hmac
import json
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import urllib.error
from typing import Any, Dict
import anyio.to_thread
from anyio import CapacityLimiter
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import REGISTRY, GaugeMetricFamily

METRICS_TOKEN_FILE = os.environ.get(
    "METRICS_TOKEN_FILE", "/home/ecs-user/services/monitoring/metrics_token"
)

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
    "laya": "http://localhost:18075/api/laya/predict",
    "decider2b": "http://localhost:18078/api/decider2b/predict",
}

# Starting concurrency limit per model (0 = never send 429 for this model).
# The limit then adapts to measured latency; see AdaptiveLimit.
# Defaults are the load-test thresholds from ticket #253.
DEFAULT_CONCURRENCY_LIMITS = {
    "nox4b": 4,
    "lux9b": 4,
    "lfm25rlcd": 8,
    "laya": 8,
    "decider2b": 4,
}


def _concurrency_limits() -> Dict[str, int]:
    limits = dict(DEFAULT_CONCURRENCY_LIMITS)
    for item in os.environ.get("CONCURRENCY_LIMITS", "").split(","):
        if "=" in item:
            model, value = item.split("=", 1)
            if model.strip() in BACKENDS:
                limits[model.strip()] = max(0, int(value))
    return limits


CONCURRENCY_LIMITS = _concurrency_limits()
LATENCY_TARGET_SECONDS = float(os.environ.get("LATENCY_TARGET_SECONDS", "2.0"))
CONCURRENCY_MIN = max(1, int(os.environ.get("CONCURRENCY_MIN", "1")))
CONCURRENCY_MAX = max(CONCURRENCY_MIN, int(os.environ.get("CONCURRENCY_MAX", "32")))
RETRY_AFTER_SECONDS = 1


class AdaptiveLimit:
    """AIMD concurrency limit driven by gateway-side latency of successful requests.

    Only touched from the event loop (the middleware), so no lock is needed.
    """

    ADJUST_EVERY = 1.0   # seconds between adjustments
    MIN_SAMPLES = 5      # don't adjust on fewer completed requests than this
    DECREASE = 0.8

    def __init__(self, initial: int):
        self.limit = float(min(max(initial, CONCURRENCY_MIN), CONCURRENCY_MAX))
        self.started = {}          # token -> start time, for requests in progress
        self.peak_active = 0
        self.samples = deque(maxlen=50)
        self.last_adjust = time.monotonic()

    @property
    def active(self) -> int:
        return len(self.started)

    @property
    def allowed(self) -> int:
        return int(self.limit)

    def start(self) -> object:
        token = object()
        self.started[token] = time.monotonic()
        self.peak_active = max(self.peak_active, self.active)
        return token

    def finish(self, token: object, latency: float, ok: bool):
        self.started.pop(token, None)
        if ok:
            self.samples.append(latency)
        self.adjust()

    def adjust(self):
        """Called on every arrival and completion, acts at most once per ADJUST_EVERY.

        Overload shows up two ways: finished requests were slow (P90), or a request
        still in progress has already waited past the target. The second matters
        when requests are slow, because then few finish and P90 alone reacts late.
        """
        now = time.monotonic()
        if now - self.last_adjust < self.ADJUST_EVERY:
            return
        oldest = now - min(self.started.values()) if self.started else 0.0
        p90 = None
        if len(self.samples) >= self.MIN_SAMPLES:
            ordered = sorted(self.samples)
            p90 = ordered[int(0.9 * (len(ordered) - 1))]
        if oldest > LATENCY_TARGET_SECONDS or (p90 is not None and p90 > LATENCY_TARGET_SECONDS):
            self.limit = max(CONCURRENCY_MIN, self.limit * self.DECREASE)
        elif p90 is not None and self.peak_active >= self.allowed:
            # Only grow while the limit is really in use, or it creeps up while idle.
            self.limit = min(CONCURRENCY_MAX, self.limit + 1)
        else:
            return   # not enough evidence either way; keep collecting samples
        self.samples.clear()
        self.peak_active = self.active
        self.last_adjust = now


# model -> AdaptiveLimit, or None when shedding is off for that model.
_limits = {m: (AdaptiveLimit(CONCURRENCY_LIMITS.get(m, 0)) if CONCURRENCY_LIMITS.get(m, 0) else None)
           for m in BACKENDS}

# Latency buckets span ~30 ms model calls up to the 120 s backend timeout.
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
                   1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0, 120.0)

REQUESTS = Counter(
    "decision_requests_total",
    "Predict requests received by the gateway, by model and HTTP status code.",
    ["model", "code"],
)
LATENCY = Histogram(
    "decision_request_duration_seconds",
    "Gateway-side E2E latency per request, including time waiting for a gateway "
    "thread and time queued in the backend.",
    ["model"],
    buckets=LATENCY_BUCKETS,
)
IN_FLIGHT = Gauge(
    "decision_requests_in_flight",
    "Requests currently inside the gateway for this model (running + waiting, "
    "including waiting for a gateway thread).",
    ["model"],
)
MODEL_TIME = Histogram(
    "decision_model_compute_seconds",
    "Model compute time reported by the backend's elapsed_ms field (LFM only). "
    "Excludes queueing, so E2E minus this is roughly the wait.",
    ["model"],
    buckets=LATENCY_BUCKETS,
)
INPUT_BYTES = Histogram(
    "decision_request_input_bytes",
    "Size of the JSON request body forwarded to the backend (sizes only, no content).",
    ["model"],
    buckets=(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576),
)
INPUT_TOKENS = Histogram(
    "decision_request_input_tokens",
    "Input tokens reported by the backend (usage.input_tokens; models that report it).",
    ["model"],
    buckets=(32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
)
QUESTIONS = Histogram(
    "decision_request_questions",
    "Questions per request (for LFM: fields in the schema).",
    ["model"],
    buckets=(1, 2, 3, 4, 6, 8, 12, 16, 32, 64),
)
LIMIT = Gauge(
    "decision_concurrency_limit",
    "Current adaptive limit: requests allowed in progress per model before new "
    "ones get 429 (0 = shedding off for this model).",
    ["model"],
)
for _m in BACKENDS:
    IN_FLIGHT.labels(_m).set(0)
    LIMIT.labels(_m).set(_limits[_m].allowed if _limits[_m] else 0)


def _backend_up(url: str) -> float:
    health = url.split("/api/")[0] + "/health"
    try:
        with urllib.request.urlopen(health, timeout=2) as r:
            return 1.0 if r.status == 200 else 0.0
    except Exception:
        return 0.0


_health_pool = ThreadPoolExecutor(max_workers=len(BACKENDS))


class ScrapeTimeCollector:
    """Backend liveness and GPU memory, read fresh on every scrape."""

    def collect(self):
        # A busy backend answers /health slowly; check all of them in parallel so
        # one scrape takes at most ~2 s. A fully saturated backend may read 0.
        up = GaugeMetricFamily("decision_backend_up", "1 if the backend's /health answers within 2 s, else 0.", labels=["model"])
        for model, value in zip(BACKENDS, _health_pool.map(_backend_up, BACKENDS.values())):
            up.add_metric([model], value)
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


def _metrics_token():
    try:
        with open(METRICS_TOKEN_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


# /metrics gets its own thread, outside FastAPI's shared 40-thread pool, so a
# scrape is not stuck behind predict requests when the gateway is busy.
_metrics_limiter = CapacityLimiter(1)


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    token = _metrics_token()
    sent = request.headers.get("authorization", "")
    if not token or not hmac.compare_digest(sent, f"Bearer {token}"):
        raise HTTPException(status_code=404, detail="Not Found")
    body = await anyio.to_thread.run_sync(generate_latest, REGISTRY, limiter=_metrics_limiter)
    return Response(body, media_type=CONTENT_TYPE_LATEST)


# "/api/<model>/predict" -> model name, for the metrics middleware.
PREDICT_PATHS = {f"/api/{m}/predict": m for m in BACKENDS}


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    model = PREDICT_PATHS.get(request.url.path) if request.method == "POST" else None
    if model is None:
        return await call_next(request)
    limiter = _limits[model]
    if limiter is not None:
        limiter.adjust()
        LIMIT.labels(model).set(limiter.allowed)
    if limiter is not None and limiter.active >= limiter.allowed:
        # Kept out of LATENCY and IN_FLIGHT so instant rejections don't skew them.
        REQUESTS.labels(model, "429").inc()
        return JSONResponse(
            status_code=429,
            content={"detail": f"{model} is overloaded: {limiter.active} requests in progress, "
                               f"responses are slower than {LATENCY_TARGET_SECONDS:g} s. "
                               f"Retry after {RETRY_AFTER_SECONDS} second."},
            # This middleware sits outside CORSMiddleware, so add the header here.
            headers={"Retry-After": str(RETRY_AFTER_SECONDS), "Access-Control-Allow-Origin": "*"},
        )
    code = 500
    token = limiter.start() if limiter is not None else None
    IN_FLIGHT.labels(model).inc()
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - t0
        LATENCY.labels(model).observe(elapsed)
        REQUESTS.labels(model, str(code)).inc()
        IN_FLIGHT.labels(model).dec()
        if limiter is not None:
            limiter.finish(token, elapsed, ok=200 <= code < 300)
            LIMIT.labels(model).set(limiter.allowed)


def _record_input_shape(model: str, payload: Dict[str, Any], body: bytes):
    INPUT_BYTES.labels(model).observe(len(body))
    questions = payload.get("questions")
    if isinstance(questions, list):
        QUESTIONS.labels(model).observe(len(questions))
    else:
        props = (payload.get("schema") or {}).get("properties") if isinstance(payload.get("schema"), dict) else None
        if isinstance(props, dict):
            QUESTIONS.labels(model).observe(len(props))


def _proxy(model: str, payload: Dict[str, Any]):
    body = json.dumps(payload).encode()
    _record_input_shape(model, payload, body)
    req = urllib.request.Request(
        BACKENDS[model], data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
        if isinstance(result.get("elapsed_ms"), (int, float)):
            MODEL_TIME.labels(model).observe(result["elapsed_ms"] / 1000)
        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), (int, float)):
            INPUT_TOKENS.labels(model).observe(usage["input_tokens"])
        return result
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            detail = json.loads(body).get("detail", body)
        except ValueError:
            detail = body
        raise HTTPException(status_code=e.code, detail=detail)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=f"backend timed out: {e}")
    except urllib.error.URLError as e:
        code = 504 if isinstance(e.reason, TimeoutError) else 503
        raise HTTPException(status_code=code, detail=f"backend unreachable: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backends": list(BACKENDS.keys()),
        "latency_target_seconds": LATENCY_TARGET_SECONDS,
        "concurrency_limits": {m: (lim.allowed if lim else 0) for m, lim in _limits.items()},
        "in_progress": {m: (lim.active if lim else None) for m, lim in _limits.items()},
    }


@app.post("/api/nox4b/predict")
def nox4b_predict(req: Dict[str, Any]):
    return _proxy("nox4b", req)


@app.post("/api/lux9b/predict")
def lux9b_predict(req: Dict[str, Any]):
    return _proxy("lux9b", req)


@app.post("/api/lfm25rlcd/predict")
def lfm25rlcd_predict(req: Dict[str, Any]):
    return _proxy("lfm25rlcd", req)


@app.post("/api/laya/predict")
def laya_predict(req: Dict[str, Any]):
    return _proxy("laya", req)


@app.post("/api/decider2b/predict")
def decider2b_predict(req: Dict[str, Any]):
    return _proxy("decider2b", req)
