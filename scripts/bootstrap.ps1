<#
.SYNOPSIS
  bootstrap.ps1: from a fresh clone to a RUNNING, VERIFIED system with one command (Windows).
  Linux/macOS: scripts/bootstrap.sh does the same steps. Full guide: DEPLOY.md.

.EXAMPLE
  .\scripts\bootstrap.ps1                      # sync write mode (default), smoke test at the end
  .\scripts\bootstrap.ps1 -Mode async          # async write mode (ADR 0013)
  .\scripts\bootstrap.ps1 -MigrateCheck        # also prove a live blue/green model migration works
  .\scripts\bootstrap.ps1 -BuildIndex          # encode embeddings locally instead of downloading them
  .\scripts\bootstrap.ps1 -Down                # stop everything (data and caches are kept)

.NOTES
  Needs ONLY Docker Desktop (Compose v2). No Python on the host. Safe to re-run.
  If scripts are blocked: powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
  Steps: 0 prerequisites, 1 config, 2 image, 3 data, 4 embeddings, 5 start, 6 verify (see bootstrap.sh).
#>
param(
    [ValidateSet("sync", "async")] [string]$Mode = $(if ($env:APPLY_MODE) { $env:APPLY_MODE } else { "sync" }),
    [switch]$MigrateCheck,
    [switch]$BuildIndex,
    [switch]$Down
)
# "Continue", not "Stop": in Windows PowerShell 5.1, native commands that write progress to stderr
# (docker does) would otherwise abort the script. Failures are detected via $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)   # repo root, wherever the script is called from
$env:APPLY_MODE = $Mode
$ReleaseTag = "index-artifacts-v1"

function Step($t) { Write-Host "`n==> $t" -ForegroundColor Cyan }
function Ok($t) { Write-Host "    ok  $t" -ForegroundColor Green }
function Warn($t) { Write-Host "    warning: $t" -ForegroundColor Yellow }
function Die($t) { Write-Host "`nERROR: $t" -ForegroundColor Red; exit 1 }
# Run a native command; stop with a message if it fails (Windows PowerShell 5.1 doesn't throw on exit codes).
function Run([string]$what, [scriptblock]$cmd) { & $cmd; if ($LASTEXITCODE -ne 0) { Die "$what failed (exit $LASTEXITCODE)" } }
# One-off command inside the project image (compose profile "tools").
function Tools { docker compose --profile tools run --rm -T tools @args }

# --------------------------------------------------------------------------------------------------
Step "0/6 prerequisites"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $dd = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin"   # per-user Docker Desktop install
    if (Test-Path (Join-Path $dd "docker.exe")) { $env:PATH = "$dd;$env:PATH" }
    else { Die "Docker is not installed. Install Docker Desktop: https://docs.docker.com/desktop/setup/install/windows-install/" }
}
docker info *> $null; if ($LASTEXITCODE -ne 0) { Die "The Docker daemon is not running. Start Docker Desktop and wait until it says 'Engine running'." }
docker compose version *> $null; if ($LASTEXITCODE -ne 0) { Die "Docker Compose v2 is required ('docker compose'). Update Docker Desktop." }
Ok "$(docker --version); $(docker compose version)"

if ($Down) {
    Step "stopping the stack (data, embeddings and WALs are kept on disk)"
    Run "docker compose down" { docker compose --profile tools down }
    exit 0
}

$memGB = [math]::Floor([double](docker info --format "{{.MemTotal}}") / 1GB)
if ($memGB -lt 6) { Warn "Docker has only $memGB GB of memory. Give it >= 8 GB (Docker Desktop > Settings > Resources)." } else { Ok "Docker memory: $memGB GB" }
$freeGB = [math]::Floor((Get-PSDrive -Name (Get-Location).Drive.Name).Free / 1GB)
if ($freeGB -lt 12) { Warn "only $freeGB GB free disk; the image + data + caches need ~8-10 GB" } else { Ok "free disk: $freeGB GB" }

# Port 8000 must be free, unless it is OUR router from an earlier run.
$busy = $false
try { $c = New-Object Net.Sockets.TcpClient; $c.Connect("127.0.0.1", 8000); $busy = $true; $c.Close() } catch { }
if ($busy) {
    $running = docker compose ps --status running --services 2>$null
    if ($running -contains "router") { Ok "port 8000: our router is already running (will be updated in place)" }
    else { Die "port 8000 is already used by another program. Stop it, then re-run." }
}

# --------------------------------------------------------------------------------------------------
Step "1/6 configuration"
# .env must exist BEFORE any 'compose run tools' (it is bind-mounted; Docker would otherwise create a
# DIRECTORY named .env). It only holds optional LLM settings for Part C.
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Ok "created .env from .env.example (LLM keys optional)" } else { Ok ".env exists" }
foreach ($d in "data\raw", "artifacts", "results\runs") { New-Item -ItemType Directory -Force $d | Out-Null }
# PyPI mirror: PIP_INDEX_URL, else the host's pip config (corporate networks), else pypi.org.
if (-not $env:PIP_INDEX_URL) {
    foreach ($p in "pip", "pip3", "py") {
        if (Get-Command $p -ErrorAction SilentlyContinue) {
            $u = if ($p -eq "py") { & py -m pip config get global.index-url 2>$null } else { & $p config get global.index-url 2>$null }
            if ($LASTEXITCODE -eq 0 -and $u) { $env:PIP_INDEX_URL = "$u".Trim(); break }
        }
    }
}
if (-not $env:PIP_INDEX_URL) { $env:PIP_INDEX_URL = "https://pypi.org/simple" }
Ok "PyPI index for the image build: $env:PIP_INDEX_URL"
Ok "write mode: APPLY_MODE=$env:APPLY_MODE"

# --------------------------------------------------------------------------------------------------
Step "2/6 build the image (first time ~5-10 min; later runs use the cache)"
Run "docker compose build" { docker compose build }
Ok "image fiqa-retrieval:latest ready"

$slug = (git config --get remote.origin.url 2>$null) -replace '^.*github\.com[:/]', '' -replace '\.git$', '' -replace '/$', ''
# Download a release asset: 1) public URL, 2) GitHub API with a token (GITHUB_TOKEN / GH_TOKEN, or the
# gh CLI's login) for PRIVATE repos. Returns $true on success.
function Fetch-ReleaseAsset([string]$name, [string]$dest) {
    if (-not $slug) { return $false }
    $out = Join-Path $dest $name
    try {
        Invoke-WebRequest -UseBasicParsing -ErrorAction Stop -Uri "https://github.com/$slug/releases/download/$ReleaseTag/$name" -OutFile $out
        return $true
    } catch { Remove-Item $out -ErrorAction SilentlyContinue }
    $token = if ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } elseif ($env:GH_TOKEN) { $env:GH_TOKEN } else { $null }
    if (-not $token -and (Get-Command gh -ErrorAction SilentlyContinue)) { $token = (gh auth token 2>$null | Out-String).Trim() }
    if (-not $token) { return $false }
    try {
        $h = @{ Authorization = "Bearer $token" }
        $rel = Invoke-RestMethod -ErrorAction Stop -Headers $h -Uri "https://api.github.com/repos/$slug/releases/tags/$ReleaseTag"
        $asset = $rel.assets | Where-Object { $_.name -eq $name } | Select-Object -First 1
        if (-not $asset) { return $false }
        $h["Accept"] = "application/octet-stream"
        Invoke-WebRequest -UseBasicParsing -ErrorAction Stop -Headers $h -Uri $asset.url -OutFile $out
        return $true
    } catch { Remove-Item $out -ErrorAction SilentlyContinue; return $false }
}

# --------------------------------------------------------------------------------------------------
Step "3/6 data: BEIR FiQA (57,638 passages)"
if (Test-Path data\raw\fiqa\corpus.jsonl) { Ok "already present: data\raw\fiqa" }
else {
    Tools python scripts/download_data.py
    if ($LASTEXITCODE -eq 0) { Ok "downloaded from the BEIR mirror" }
    else {
        Warn "upstream download failed; trying the copy attached to the GitHub release"
        if (-not (Fetch-ReleaseAsset "fiqa.zip" "data\raw")) { Die "could not download FiQA. Check your internet connection / proxy and re-run." }
        Run "unzip fiqa.zip" { Tools python -c "import zipfile; zipfile.ZipFile('data/raw/fiqa.zip').extractall('data/raw')" }
        Ok "extracted data\raw\fiqa"
    }
}

# --------------------------------------------------------------------------------------------------
Step "4/6 embedding caches (MiniLM + bge-small for all passages)"
function Have-Caches { (Test-Path artifacts\emb_minilm.npy) -and (Test-Path artifacts\emb_minilm.meta.json) -and (Test-Path artifacts\emb_bge-small.npy) -and (Test-Path artifacts\emb_bge-small.meta.json) }
if (Have-Caches) { Ok "already present: artifacts\emb_*" }
else {
    if (-not $BuildIndex -and (Fetch-ReleaseAsset "fiqa_index_artifacts.zip" "artifacts")) {
        Run "unzip caches" { Tools python -c "import zipfile; zipfile.ZipFile('artifacts/fiqa_index_artifacts.zip').extractall('.')" }
        Remove-Item artifacts\fiqa_index_artifacts.zip
        Ok "downloaded pre-built caches from release $ReleaseTag"
    } else {
        if (-not $BuildIndex) { Warn "release download not possible (private repo without the gh CLI?)" }
        Warn "encoding 2 x 57,638 passages on the CPU: ~30-40 min, once"
        Run "build_index" { Tools python -m src.index.build_index --models minilm bge-small }
    }
    if (-not (Have-Caches)) { Die "embedding caches missing after step 4" }
    Ok "caches ready"
}

# --------------------------------------------------------------------------------------------------
Step "5/6 start: 3 shard containers + router (waits until every healthcheck passes, ~1-2 min)"
docker compose up -d --wait --wait-timeout 900
if ($LASTEXITCODE -ne 0) { docker compose ps; docker compose logs --tail 40; Die "the stack did not become healthy (logs above)" }
docker compose ps --format "table {{.Service}}`t{{.State}}`t{{.Health}}`t{{.Ports}}"

# --------------------------------------------------------------------------------------------------
Step "6/6 verify (inside a container, against http://router:8000)"
$smoke = @("python", "-m", "scripts.smoke_test", "--url", "http://router:8000")
if ($MigrateCheck) { $smoke += "--migrate" }
Tools @smoke
if ($LASTEXITCODE -ne 0) { Die "smoke test failed (see above). Logs: docker compose logs router shard0" }

Write-Host "`nSystem is up and verified." -ForegroundColor Green
Write-Host @"
  Router (the only entry point):  http://localhost:8000      API docs: http://localhost:8000/docs
  Try a search:
    Invoke-RestMethod -Method Post localhost:8000/search -ContentType application/json -Body '{"query":"what is short selling?"}'
  Status of every shard:          Invoke-RestMethod localhost:8000/admin/status
  Full dev evaluation (500 q):    docker compose --profile tools run --rm eval
  Logs:                           docker compose logs -f router shard0
  Stop (keeps data):              .\scripts\bootstrap.ps1 -Down
  More (scenarios, migration, load tests, troubleshooting): DEPLOY.md
"@
