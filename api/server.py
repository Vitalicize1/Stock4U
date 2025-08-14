from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Response
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator

from langgraph_flow import run_prediction
from backtesting import run as bt
from utils.baseline import BaselineConfig, run_baseline
from utils.validation import InputValidator, ValidationResult
from utils.validation_middleware import create_validation_middleware
"""Paper trading endpoints removed per UI simplification."""


from utils.auth import auth_guard
from utils.logger import log_metric, increment
import time

app = FastAPI(title="Stock4U API", version="0.1.0")

# CORS (env-driven)
import os as _os
_origins_env = _os.getenv("ALLOWED_ORIGINS", "*")
_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] if _origins_env != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add validation middleware
enable_validation = _os.getenv("ENABLE_VALIDATION_MIDDLEWARE", "true").lower() == "true"
if enable_validation:
    create_validation_middleware(app, enable_validation=True)


# Request size limit middleware (simple Content-Length guard)
class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_bytes: int) -> None:
        super().__init__(app)
        self.max_body_bytes = int(max_body_bytes)

    async def dispatch(self, request, call_next):
        cl = request.headers.get("content-length")
        try:
            if cl is not None and int(cl) > self.max_body_bytes:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError:
            pass
        return await call_next(request)


_max_bytes = int(_os.getenv("MAX_BODY_BYTES", str(1024 * 1024)))  # 1MB default
app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=_max_bytes)
# --- Optional background scheduler for nightly learning
try:
    import os
    from apscheduler.schedulers.background import BackgroundScheduler
    from utils.agent_learn_schedule import run_learning_batch, BatchConfig

    _scheduler_enabled = os.getenv("LEARNING_SCHED_ENABLED", "1") == "1"
    if _scheduler_enabled:
        tickers_env = os.getenv("LEARNING_TICKERS", "AAPL,MSFT,GOOGL,NVDA,AMZN")
        timeframes_env = os.getenv("LEARNING_TIMEFRAMES", "1d,1w")
        period_env = os.getenv("LEARNING_PERIOD", "1y")
        iterations_env = int(os.getenv("LEARNING_ITERATIONS", "120"))
        lr_env = float(os.getenv("LEARNING_LR", "0.08"))
        use_ml = os.getenv("LEARNING_USE_ML", "0") == "1"
        online = os.getenv("LEARNING_ONLINE", "0") == "1"

        _scheduler = BackgroundScheduler(timezone="UTC")

        def _nightly_job():
            try:
                cfg = BatchConfig(
                    tickers=[t.strip().upper() for t in tickers_env.split(",") if t.strip()],
                    timeframes=[tf.strip() for tf in timeframes_env.split(",") if tf.strip()],
                    period=period_env,
                    iterations=iterations_env,
                    lr=lr_env,
                    use_ml_model=use_ml,
                    offline=(not online),
                )
                run_learning_batch(cfg)
            except Exception:
                # Swallow to keep API alive
                pass

        def _daily_picks_job():
            try:
                from utils.daily_picks import run_daily_picks_job
                run_daily_picks_job()
            except Exception:
                pass

        # Default: 02:30 UTC daily
        cron = os.getenv("LEARNING_CRON", "30 2 * * *")
        # APScheduler expects fields: minute hour day month day_of_week
        try:
            m, h, dom, mon, dow = cron.split()
            _scheduler.add_job(_nightly_job, "cron", minute=m, hour=h, day=dom, month=mon, day_of_week=dow, id="agent_learning")
            # Daily Top Picks at 14:00 UTC by default (after market open US)
            picks_cron = os.getenv("DAILY_PICKS_CRON", "0 14 * * 1-5")
            pm, ph, pdom, pmon, pdow = picks_cron.split()
            _scheduler.add_job(_daily_picks_job, "cron", minute=pm, hour=ph, day=pdom, month=pmon, day_of_week=pdow, id="daily_top_picks", replace_existing=True)
            _scheduler.start()
        except Exception:
            pass
except Exception:
    # Scheduler is optional; API still works without it
    pass



class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol")
    timeframe: str = Field("1d", description="Timeframe: 1d|1w|1m")
    low_api_mode: bool = False
    fast_ta_mode: bool = False
    use_ml_model: bool = False

    @validator("ticker")
    def _validate_ticker(cls, v: str) -> str:
        result = InputValidator.validate_ticker_symbol(v)
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value

    @validator("timeframe")
    def _validate_timeframe(cls, v: str) -> str:
        result = InputValidator.validate_timeframe(v)
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value


class BacktestRequest(BaseModel):
    ticker: Optional[str] = None
    tickers: Optional[List[str]] = None
    period: str = Field("6mo", description="History period (e.g., 6mo, 1y)")
    offline: bool = True
    policy: str = Field("agent", description="agent|rule|sma20 (single-asset only)")
    cash: float = 100_000.0
    fee_bps: float = 5.0
    slip_bps: float = 5.0

    @validator("ticker")
    def _validate_ticker(cls, v):
        if v is None:
            return v
        result = InputValidator.validate_ticker_symbol(v)
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value

    @validator("tickers", pre=True)
    def _validate_tickers(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = [t.strip() for t in v.split(",") if t.strip()]
        result = InputValidator.validate_ticker_list(v)
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value

    @validator("period")
    def _validate_period(cls, v: str) -> str:
        result = InputValidator.validate_period(v)
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value

    @validator("cash")
    def _validate_cash(cls, v: float) -> float:
        result = InputValidator.validate_numeric_range(v, 0, InputValidator.MAX_VALUES["max_cash_amount"], "cash")
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value

    @validator("fee_bps")
    def _validate_fee_bps(cls, v: float) -> float:
        result = InputValidator.validate_numeric_range(v, 0, InputValidator.MAX_VALUES["max_fee_bps"], "fee_bps")
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value

    @validator("slip_bps")
    def _validate_slip_bps(cls, v: float) -> float:
        result = InputValidator.validate_numeric_range(v, 0, InputValidator.MAX_VALUES["max_slip_bps"], "slip_bps")
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value
class BaselineRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL","MSFT","GOOGL","NVDA","AMZN"])
    period: str = "1y"
    policies: list[str] = Field(default_factory=lambda: ["agent","rule","sma20"])
    online: bool = False
    fee_bps: float = 5.0
    slip_bps: float = 5.0
    walk_forward: bool = False
    wf_splits: int = 3
    tune_thresholds: bool = False

    @validator("tickers")
    def _validate_tickers(cls, v: list[str]) -> list[str]:
        result = InputValidator.validate_ticker_list(v)
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value

    @validator("period")
    def _validate_period(cls, v: str) -> str:
        result = InputValidator.validate_period(v)
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value

    @validator("fee_bps")
    def _validate_fee_bps(cls, v: float) -> float:
        result = InputValidator.validate_numeric_range(v, 0, InputValidator.MAX_VALUES["max_fee_bps"], "fee_bps")
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value

    @validator("slip_bps")
    def _validate_slip_bps(cls, v: float) -> float:
        result = InputValidator.validate_numeric_range(v, 0, InputValidator.MAX_VALUES["max_slip_bps"], "slip_bps")
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value

    @validator("wf_splits")
    def _validate_wf_splits(cls, v: int) -> int:
        result = InputValidator.validate_numeric_range(v, 1, 10, "wf_splits")
        if not result.is_valid:
            raise ValueError(result.error_message)
        return int(result.sanitized_value)


class PaperTradeRequest(BaseModel):
    pass



@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest, _: None = Depends(auth_guard)) -> dict:
    t0 = time.perf_counter()
    try:
        result = run_prediction(
            req.ticker,
            req.timeframe,
            low_api_mode=req.low_api_mode,
            fast_ta_mode=req.fast_ta_mode,
            use_ml_model=req.use_ml_model,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        increment("api_request_total", 1, {"endpoint": "/predict", "status": "success"})
        log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/predict"})
        return {"status": "success", "result": jsonable_encoder(result)}
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        increment("api_request_total", 1, {"endpoint": "/predict", "status": "error"})
        log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/predict"})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
def backtest(req: BacktestRequest, _: None = Depends(auth_guard)) -> dict:
    t0 = time.perf_counter()
    try:
        broker = bt.BrokerConfig(starting_cash=req.cash, fee_bps=req.fee_bps, slip_bps=req.slip_bps)
        if req.tickers:
            out = bt._simulate_portfolio(
                symbols=[t.upper() for t in req.tickers],
                period=req.period,
                broker=broker,
                offline=req.offline,
                use_ml_model=False,
                outdir="cache/results",
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            increment("api_request_total", 1, {"endpoint": "/backtest", "status": "success"})
            log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/backtest"})
            return {"status": "success", "portfolio": jsonable_encoder(out)}
        elif req.ticker:
            out = bt.simulate(
                ticker=req.ticker.upper(),
                period=req.period,
                broker=broker,
                use_ml_model=False,
                offline=req.offline,
                policy=req.policy,
                outdir="cache/results",
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            increment("api_request_total", 1, {"endpoint": "/backtest", "status": "success"})
            log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/backtest"})
            return {"status": "success", "single": jsonable_encoder(out)}
        else:
            raise HTTPException(status_code=400, detail="Provide 'ticker' or 'tickers'")
    except HTTPException:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        increment("api_request_total", 1, {"endpoint": "/backtest", "status": "error"})
        log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/backtest"})
        raise
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        increment("api_request_total", 1, {"endpoint": "/backtest", "status": "error"})
        log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/backtest"})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/baseline")
def baseline(req: BaselineRequest, _: None = Depends(auth_guard)) -> dict:
    t0 = time.perf_counter()
    try:
        cfg = BaselineConfig(
            tickers=[t.upper() for t in req.tickers],
            period=req.period,
            policies=req.policies,
            offline=(not req.online),
            fee_bps=req.fee_bps,
            slip_bps=req.slip_bps,
            walk_forward=req.walk_forward,
            wf_splits=req.wf_splits,
            tune_thresholds=req.tune_thresholds,
        )
        out = run_baseline(cfg)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        increment("api_request_total", 1, {"endpoint": "/baseline", "status": "success"})
        log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/baseline"})
        return {"status": "success", "artifact": out}
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        increment("api_request_total", 1, {"endpoint": "/baseline", "status": "error"})
        log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/baseline"})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/baseline/latest")
def baseline_latest(period: str = "1y") -> dict:
    """Return the most recent baseline summary JSON for the period if available."""
    try:
        from pathlib import Path
        import json as _json
        base = Path("cache") / "metrics" / "accuracy_baseline"
        path = base / f"baseline_{period}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="No baseline summary found for the requested period")
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return {"status": "success", "summary": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/learn")
def agent_learn(tickers: str = "AAPL,MSFT,GOOGL", timeframes: str = "1d", period: str = "1y", iterations: int = 120, lr: float = 0.08, ml: bool = False, online: bool = False, _: None = Depends(auth_guard)) -> dict:
    """Manually trigger a learning batch now."""
    t0 = time.perf_counter()
    try:
        from utils.agent_learn_schedule import BatchConfig, run_learning_batch
        cfg = BatchConfig(
            tickers=[t.strip().upper() for t in (tickers or "").split(",") if t.strip()],
            timeframes=[tf.strip() for tf in (timeframes or "").split(",") if tf.strip()],
            period=period,
            iterations=int(iterations),
            lr=float(lr),
            use_ml_model=bool(ml),
            offline=(not bool(online)),
        )
        out = run_learning_batch(cfg)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        increment("api_request_total", 1, {"endpoint": "/agent/learn", "status": "success"})
        log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/agent/learn"})
        return {"status": "success", "artifact": out}
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        increment("api_request_total", 1, {"endpoint": "/agent/learn", "status": "error"})
        log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/agent/learn"})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tuning/manifest")
def get_tuning_manifest(ticker: str, timeframe: str = "1d", _: None = Depends(auth_guard)) -> dict:
    """Return the latest tuning manifest for a ticker/timeframe."""
    try:
        from pathlib import Path
        import json as _json
        path = Path("cache") / "metrics" / "ensemble_tuning" / f"best_{ticker.upper()}_{timeframe}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Manifest not found")
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return {"status": "success", "manifest": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/learn/last")
def get_last_learned(_: None = Depends(auth_guard)) -> dict:
    """Return the last learned weights per ticker/timeframe."""
    try:
        from pathlib import Path
        import json as _json
        path = Path("cache") / "metrics" / "agent_learning" / "last_learned.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="No learning summary available")
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return {"status": "success", "last_learned": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/learn/status")
def get_learning_status(_: None = Depends(auth_guard)) -> dict:
    """Lightweight status endpoint for last learning run (artifact path + timestamp)."""
    try:
        from pathlib import Path
        import json as _json
        path = Path("cache") / "metrics" / "agent_learning" / "last_status.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="No status available")
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return {"status": "success", "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Paper trading endpoint removed


@app.get("/validation/run")
def validation_run(tickers: str = "AAPL,MSFT,GOOGL", online: bool = False, ml: bool = False, _: None = Depends(auth_guard)) -> dict:
    t0 = time.perf_counter()
    try:
        from utils.validate_agent import ValidateConfig, validate_live
        cfg = ValidateConfig(
            tickers=[t.strip().upper() for t in (tickers or "").split(",") if t.strip()],
            offline=(not online),
            use_ml_model=bool(ml),
        )
        res = validate_live(cfg)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        increment("api_request_total", 1, {"endpoint": "/validation/run", "status": "success"})
        log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/validation/run"})
        return {"status": "success", "report": res}
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        increment("api_request_total", 1, {"endpoint": "/validation/run", "status": "error"})
        log_metric("api_request_ms", float(elapsed_ms), {"endpoint": "/validation/run"})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/errors")
def health_errors(limit: int = 50, _: None = Depends(auth_guard)) -> dict:
    """Return the most recent alert/error lines for ops visibility."""
    try:
        from pathlib import Path
        path = Path("cache") / "metrics" / "alerts.log"
        if not path.exists():
            return {"status": "success", "alerts": []}
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = lines[-max(1, int(limit)) :]
        return {"status": "success", "alerts": tail}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/verify")
def auth_verify(authorization: str | None = Header(None), _: None = Depends(auth_guard)) -> dict:
    """Verify auth and return a non-sensitive fingerprint of the token.

    The full token is never returned; only a short SHA-256 prefix for debugging.
    """
    import hashlib, os
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    fp = hashlib.sha256((token or "").encode("utf-8")).hexdigest()[:8] if token else None
    return {
        "status": "success",
        "authenticated": True,
        "fingerprint": fp,
        "rate_limit_per_min": int(os.getenv("RATE_LIMIT_PER_MIN", "60")),
    }


@app.get("/metrics")
def metrics(_: None = Depends(auth_guard)) -> Response:
    """Enhanced Prometheus-style metrics for Stock4U monitoring."""
    from pathlib import Path
    import json as _json
    import psutil
    import time
    lines = []
    
    try:
        # Basic health check
        lines.append("stock4u_up 1")
        
        # System metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        lines.append(f"stock4u_process_memory_bytes {memory_info.rss}")
        lines.append(f"stock4u_process_cpu_percent {process.cpu_percent()}")
        lines.append(f"stock4u_process_threads {process.num_threads()}")
        
        # Learning job metrics
        lp = Path("cache") / "metrics" / "agent_learning" / "last_status.json"
        if lp.exists():
            data = _json.loads(lp.read_text())
            lines.append(f"stock4u_learning_last_elapsed_seconds {float(data.get('elapsed_s', 0.0))}")
            lines.append(f"stock4u_learning_last_timestamp {int(time.time())}")
        
        # Cache metrics
        cache_dir = Path("cache")
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            lines.append(f"stock4u_cache_size_bytes {cache_size}")
        
        # Database connection metrics (if available)
        try:
            from utils.database import get_db_connection
            conn = get_db_connection()
            if conn:
                lines.append("stock4u_database_connected 1")
                conn.close()
            else:
                lines.append("stock4u_database_connected 0")
        except:
            lines.append("stock4u_database_connected 0")
            
    except Exception as e:
        lines.append("stock4u_up 0")
        lines.append(f"stock4u_error_info {str(e)}")
    
    return Response("\n".join(lines) + "\n", media_type="text/plain")

