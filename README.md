# Stock4U

A multi-agent stock analysis system built with LangGraph, Streamlit, and yfinance. It performs technical analysis, integrates sentiment when available, and produces a clear prediction with risk assessment. See detailed docs in `docs/README.md`.

## Features

- Multi-agent LangGraph workflow (orchestrator, data collection, technical analysis, prediction, evaluation, elicitation)
- Streamlit dashboard with predictions, chatbot assistant, and market data
- **MCP (Model Context Protocol) server** for AI integration and external tool access
## Production notes (auth, learning, monitoring)

- API auth: set `API_TOKEN` (any strong token) and call endpoints with header `Authorization: Bearer <token>`.
- Rate limiting: `RATE_LIMIT_PER_MIN` (default 60).
- Autonomous learning: enable with `LEARNING_SCHED_ENABLED=1` and configure `LEARNING_TICKERS`, `LEARNING_TIMEFRAMES`, `LEARNING_PERIOD`, `LEARNING_ITERATIONS`, `LEARNING_LR`, `LEARNING_CRON`.
- Key endpoints:
  - `POST /predict`, `POST /baseline`, `GET /baseline/latest`
  - `POST /agent/learn`, `GET /agent/learn/status`, `GET /agent/learn/last`
  - `GET /validation/run` (schema/range checks)
  - `GET /health/errors`, `GET /metrics`, `GET /auth/verify`

### Nightly QA (optional)

- Run locally:
```
python -m utils.nightly_qa
```
- Windows Task Scheduler helper script: `ops/nightly_qa.ps1`:
```
powershell -ExecutionPolicy Bypass -File ops/nightly_qa.ps1 -ApiUrl http://localhost:8000 -Token YOUR_TOKEN
```

### Prometheus monitoring (optional)

- Config at `ops/prometheus/prometheus.yml` (uses Bearer token from `token.txt`).
- See `ops/prometheus/README.md` for Docker/native instructions.

- Technical indicators, trend analysis, support/resistance, trading signals
- Risk assessment with visual breakdown
- Optional LLM integrations (OpenAI/Google) with quota awareness

## Architecture (High Level)

Orchestrator → Data Collector → Technical Analyzer → Sentiment Analyzer → Sentiment Integrator → Prediction Agent → Evaluator Optimizer → Elicitation

Core entry points:
- `langgraph_flow.py`: runs the end-to-end workflow
- `dashboard.py`: Streamlit UI
- `agents/`: agent and tool modules

## Setup

1) Create and activate a virtual environment

```bash
python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate
```

If PowerShell blocks scripts, run once (as current user):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) (Optional) Configure API keys in a `.env` file

```bash
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key
```

## Run

### Streamlit Dashboard

```bash
streamlit run dashboard.py
# Optional flags
# streamlit run dashboard.py --server.port 8501 --server.headless true
```

Key tabs:
- Predictions: run the workflow for a ticker and timeframe
- Chatbot: ask questions or trigger stock analysis via natural language
- Market Data: quick 5‑day candlestick and metrics

### CLI

```bash
python main.py
```

### Programmatic

```python
from langgraph_flow import run_prediction

result = run_prediction("AAPL", timeframe="1d", low_api_mode=False)
print(result)
```

### MCP Server (AI Integration)

```bash
# Start MCP server
python -m agents.mcp_server

# Test with MCP CLI
mcp dev agents.mcp_server --tool ping

# Run demo
python examples/mcp_demo.py
```

**📖 See [MCP Integration Guide](docs/MCP_INTEGRATION_GUIDE.md) for detailed examples and use cases.**

## Tests

```bash
python tests/run_all_tests.py
# or
pytest -q
```

## Docs

- Start here: `docs/README.md`
- Additional guides: chatbot, workflow visualization, tool integration

## Project Structure

```
Stock4U/
├── agents/                 # AI agent modules and tools
├── docs/                   # Documentation
├── llm/                    # LLM client wrappers
├── utils/                  # Helpers (logging, caching, fetchers)
├── dashboard.py            # Streamlit UI
├── langgraph_flow.py       # Main LangGraph workflow
├── main.py                 # CLI entry
└── tests/                  # Test suite
```

## Notes

- This project is for educational/informational purposes only; not financial advice.
- Some features (e.g., LLM calls) are optional and can be disabled via UI toggles.

## Repository

GitHub: https://github.com/Vitalicize1/Stock4U


