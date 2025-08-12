Param(
  [string]$ApiUrl = "http://localhost:8000",
  [string]$Token = $env:API_TOKEN,
  [string]$Tickers = "AAPL,MSFT,GOOGL",
  [string]$Period = "1y"
)

$env:API_URL = $ApiUrl
if ($Token) { $env:API_TOKEN = $Token }
$env:QA_TICKERS = $Tickers
$env:QA_BASELINE_PERIOD = $Period

python -m utils.nightly_qa
exit $LASTEXITCODE


