<#
.SYNOPSIS
  docker_tour.ps1: run every Docker step of the manual tests (guide section 3) with one command, check each result,
  write a report with an animated replay per step, and put the stack back exactly as it was. macOS/Linux:
  scripts/docker_tour.sh does the same steps.

.EXAMPLE
  .\scripts\docker_tour.ps1                     # everything except the paid agent step (~2-3 min)
  .\scripts\docker_tour.ps1 -Azure              # also the agents (3e) on Azure; needs .env filled in
  .\scripts\docker_tour.ps1 -Skip eval,simple   # leave out some phases
  .\scripts\docker_tour.ps1 -Only writes,shard  # just these phases
  .\scripts\docker_tour.ps1 -NoOpen             # don't open the report at the end

.NOTES
  Needs ONLY Docker Desktop and a running stack (.\scripts\bootstrap.ps1). No Python on the host: the checks run in
  the project's "tools" container, with the repo's scenarios/, scripts/ and src/ mounted read-only.
  Phases, in order: search (3a), eval (3b), writes (3c), migration (3d), simple (3f), agents (3e, only with -Azure),
  shard (3g: stops and starts shard1).
  Not part of the tour: 3h (scripts/docker_check.py wipes artifacts\docker) and 3i (the async write mode): the
  tour never changes the write mode, and needs the stack in sync mode (the bootstrap default).
  Before anything changes it records the stack's shape; at the end it starts shard1 if needed, deletes the docs it
  wrote, restores the models, and verifies the shape is identical (also after a failure or Ctrl+C).
  Output: toy_runs\docker_tour\<stamp>\index.html (gitignored).
#>
param(
    [switch]$Azure,
    [switch]$NoOpen,
    [string[]]$Skip = @(),
    [string[]]$Only = @()
)
# "Continue", not "Stop": docker writes progress to stderr, which would abort the script in Windows PowerShell 5.1.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)   # repo root, wherever the script is called from
$Root = (Get-Location).Path
$Phases = @("search", "eval", "writes", "migration", "simple", "agents", "shard")
$Skip = @($Skip | ForEach-Object { $_ -split "," } | Where-Object { $_ })   # accept "-Skip eval,simple" either way
$Only = @($Only | ForEach-Object { $_ -split "," } | Where-Object { $_ })
foreach ($p in $Skip + $Only) { if ($Phases -notcontains $p) { Write-Host "unknown phase '$p' (phases: $($Phases -join ', '))" -ForegroundColor Red; exit 2 } }

function Step($t) { Write-Host "`n==> $t" -ForegroundColor Cyan }
function Want($p) { if ($Only.Count) { return $Only -contains $p } return -not ($Skip -contains $p) }

# ---------------------------------------------------------------------------------------------- prerequisites
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $dd = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin"   # per-user Docker Desktop install
    if (Test-Path (Join-Path $dd "docker.exe")) { $env:PATH = "$dd;$env:PATH" }
    else { Write-Host "Docker is not installed (or not on PATH)." -ForegroundColor Red; exit 2 }
}
docker info *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "The Docker daemon is not running. Start Docker Desktop first." -ForegroundColor Red; exit 2 }
if (-not (Test-Path .env)) { Copy-Item .env.example .env }   # the tools container mounts it

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RelDir = "toy_runs/docker_tour/$Stamp"
$Dir = Join-Path $Root "toy_runs\docker_tour\$Stamp"
New-Item -ItemType Directory $Dir -Force | Out-Null
$Mounts = @("-v", "$Root\scenarios:/app/scenarios:ro", "-v", "$Root\scripts:/app/scripts:ro",
            "-v", "$Root\src:/app/src:ro", "-v", "$Root\toy_runs:/app/toy_runs")
$AzureArg = @()
if ($Azure) {
    New-Item -ItemType Directory (Join-Path $Root ".llm_cache") -Force | Out-Null
    $Mounts += @("-v", "$Root\.llm_cache:/app/.llm_cache")   # reuse cached LLM answers: a rerun costs nothing
    $AzureArg = @("--azure")
}

# One command inside the project image, against http://router:8000 on the compose network.
function Tour { docker compose --profile tools run --rm -T @Mounts tools python -m scripts.docker_tour @args --dir $RelDir }
# The containers' state, read by the shape check (before and after).
function Save-Compose { ((docker compose ps --format json) -join "`n") | Set-Content -Encoding utf8 (Join-Path $Dir "compose_now.json") }
# A docker-level step (only the wrapper can do these) goes into the report too.
function Note($key, $section, $title, $what, $ok, $seconds, $got) {
    $rec = [ordered]@{ key = $key; section = $section; title = $title; what = $what; expected = "the stack is healthy again";
        status = $(if ($ok) { "pass" } else { "fail" }); seconds = [math]::Round($seconds, 1); got = $got;
        t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0 - $seconds }
    ($rec | ConvertTo-Json -Compress) | Add-Content -Encoding utf8 (Join-Path $Dir "wrapper.jsonl")
    $mark = $(if ($ok) { "PASS" } else { "FAIL" })
    Write-Host ("  {0,-4}  {1,3}  {2}  ({3:N1} s)" -f $mark, $section, $title, $seconds)
}
# Start shard1 again and wait until every healthcheck passes (it replays its WAL first). APPLY_MODE is pinned to
# sync so `up` never recreates the shards in another mode, whatever this shell has exported.
function Start-Shard1 {
    $env:APPLY_MODE = "sync"
    docker compose start shard1 2>&1 | Out-Host
    docker compose up -d --wait --wait-timeout 900 2>&1 | Out-Host
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------------------------------------- preflight
Step "preflight: the stack is up; its shape is recorded so it can be restored and verified at the end"
Save-Compose
Tour preflight
if ($LASTEXITCODE -ne 0) { Write-Host "`nNothing was changed." -ForegroundColor Yellow; exit 2 }
$Shard1Stopped = $false
$Completed = $false
$Rc = 1
$runNow = @("search", "eval", "writes", "migration", "simple", "agents" | Where-Object { Want $_ })
$skipped = @($Phases | Where-Object { -not (Want $_) } | ForEach-Object { "skip:$_" })

try {
    Step "3a-3f: search, the dev eval, writes, migration, the simple scenarios$(if ($Azure) { ', agents' })"
    Tour run @runNow @skipped @AzureArg

    if (Want "shard") {
        Step "3g: lose a shard"
        Tour run shard_before
        $t0 = Get-Date
        docker compose stop shard1
        $Shard1Stopped = $true
        Note "stop_shard1" "3g" "docker compose stop shard1" "stop the container; the router's calls to it now fail" ($LASTEXITCODE -eq 0) ((Get-Date) - $t0).TotalSeconds "shard1 stopped"
        Tour run shard_down
        $t0 = Get-Date
        $ok = Start-Shard1
        if ($ok) { $Shard1Stopped = $false }
        Note "start_shard1" "3g" "docker compose start shard1" "start it again; it replays its WAL before /health says healthy" $ok ((Get-Date) - $t0).TotalSeconds "shard1 healthy again: $ok"
        Tour run shard_back
    }
    $Completed = $true
}
finally {
    # Runs on success, on failure and on Ctrl+C.
    Step "restore: shard1, the tour's docs and the models; then verify the shape"
    if ($Shard1Stopped) {
        $t0 = Get-Date
        $ok = Start-Shard1
        Note "restart_shard1" "end" "Start shard1 again (the tour stopped early)" "docker compose start shard1, wait until healthy" $ok ((Get-Date) - $t0).TotalSeconds "shard1 healthy again: $ok"
    }
    Save-Compose
    Tour finish
    $Rc = $LASTEXITCODE
    if (-not $Completed -and $Rc -eq 0) { $Rc = 130 }
    $report = Join-Path $Dir "index.html"
    if ((Test-Path $report) -and -not $NoOpen) { Start-Process $report }
    Write-Host ""
    if (-not $Completed) { Write-Host "The tour stopped early; the stack was restored (the shape check is the last table in the report)." -ForegroundColor Yellow }
    elseif ($Rc -eq 0) { Write-Host "Every step passed, and the stack is back in its original shape." -ForegroundColor Green }
    else { Write-Host "Some steps failed (see the report). The shape check is the last table in it." -ForegroundColor Yellow }
    Write-Host "Report: $report"
}
exit $Rc