# Prometheus setup for Stock4U

1) Create a token file containing your API token (no newline required):

```powershell
# PowerShell
$env:TOKEN="YOUR_TOKEN"; Set-Content -NoNewline -Encoding ascii ops/prometheus/token.txt $env:TOKEN
```

2) Start Prometheus via Docker with this config mounted:

```bash
# From the project root
docker run --rm -p 9090:9090 \
  -v ${PWD}/ops/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
  -v ${PWD}/ops/prometheus/token.txt:/etc/prometheus/token.txt \
  prom/prometheus
```

3) Open Prometheus UI at http://localhost:9090 and query:
- stock4u_up
- stock4u_learning_last_elapsed_seconds

Note: ensure your API is running with API_TOKEN set, e.g.:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --env-file .env
```


