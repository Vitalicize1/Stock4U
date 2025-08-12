from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator

from langgraph_flow import run_prediction
from backtesting import run as bt


app = FastAPI(title="Stock4U API", version="0.1.0")


class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol")
    timeframe: str = Field("1d", description="Timeframe: 1d|1w|1m")
    low_api_mode: bool = False
    fast_ta_mode: bool = False
    use_ml_model: bool = False

    @validator("ticker")
    def _upper(cls, v: str) -> str:
        return (v or "").upper().strip()


class BacktestRequest(BaseModel):
    ticker: Optional[str] = None
    tickers: Optional[List[str]] = None
    period: str = Field("6mo", description="History period (e.g., 6mo, 1y)")
    offline: bool = True
    policy: str = Field("agent", description="agent|rule|sma20 (single-asset only)")
    cash: float = 100_000.0
    fee_bps: float = 5.0
    slip_bps: float = 5.0

    @validator("tickers", pre=True)
    def _split(cls, v):
        if isinstance(v, str):
            return [t.strip().upper() for t in v.split(",") if t.strip()]
        return v


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        result = run_prediction(
            req.ticker,
            req.timeframe,
            low_api_mode=req.low_api_mode,
            fast_ta_mode=req.fast_ta_mode,
            use_ml_model=req.use_ml_model,
        )
        return {"status": "success", "result": jsonable_encoder(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
def backtest(req: BacktestRequest) -> dict:
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
            return {"status": "success", "single": jsonable_encoder(out)}
        else:
            raise HTTPException(status_code=400, detail="Provide 'ticker' or 'tickers'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


