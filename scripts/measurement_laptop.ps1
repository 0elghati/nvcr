<# Uses the controlled-study runner. Check runs only a bounded smoke and one 100-frame timing pair. #>
[CmdletBinding()]
param(
    [ValidateSet('3050','5060')][string]$Gpu = '5060',
    [ValidateSet('Check','Run')][string]$Action = 'Check',
    [string]$Image = 'nvcr-measurement:1.0.7',
    [string]$EngineVolume,
    [string]$EngineDir,
    [string]$InputDir,
    [string]$ResultsDir,
    [string]$SmokeDir
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (!$EngineVolume) { $EngineVolume = "nvcr-portability-107-$Gpu" }
if ($EngineDir) { $EngineDir = (Resolve-Path -LiteralPath $EngineDir).Path }
if (!$InputDir) { $InputDir = Join-Path $repo 'datasets' }
$InputDir = (Resolve-Path -LiteralPath $InputDir).Path
if (!$ResultsDir) {
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $ResultsDir = Join-Path $repo "evidence/measurement-$Gpu-$stamp-$([guid]::NewGuid().ToString('N').Substring(0,8))"
}
if ($SmokeDir) { $SmokeDir = (Resolve-Path -LiteralPath $SmokeDir).Path }
if (Test-Path -LiteralPath $ResultsDir) {
    if (!(Test-Path -LiteralPath $ResultsDir -PathType Container) -or @(Get-ChildItem -LiteralPath $ResultsDir -Force).Count) {
        throw 'Result directory contains evidence; select a new directory. Nothing was overwritten.'
    }
} else { New-Item -ItemType Directory -Path $ResultsDir | Out-Null }
$ResultsDir = (Resolve-Path -LiteralPath $ResultsDir).Path

$inspectPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    $imageInfo = & docker image inspect $Image 2>$null
} finally { $ErrorActionPreference = $inspectPreference }
if ($LASTEXITCODE -ne 0) {
    & docker build --pull=false -t $Image -f (Join-Path $PSScriptRoot 'measurement_laptop.Dockerfile') $PSScriptRoot
    if ($LASTEXITCODE -ne 0) { throw 'Measurement image build failed.' }
    $imageInfo = & docker image inspect $Image
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect measurement image.' }
}
$info = @($imageInfo | ConvertFrom-Json)[0]
$imageId = $info.Id
$labels = $info.Config.Labels
if ($labels.'org.opencontainers.image.revision' -ne '6a66db7a98081f9210dedb49d768899bd4e9f2b7' -or $labels.'org.opencontainers.image.version' -ne '1.0.7') {
    throw 'Expected the recorded NVCR 1.0.7 release image.'
}
$probe = "import hashlib,json,shutil; from pathlib import Path; p=Path(shutil.which('nvcr')); print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
$binaryText = & docker run --rm --network none --read-only --entrypoint python3 $imageId -c $probe
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify measurement executable.' }
$binary = $binaryText | ConvertFrom-Json
if ($binary.sha256 -ne '8d12a772ede71839d9c3bfdbabd43da10e917d6bcdaa52199f3c45dc49dd69b0') {
    throw 'NVCR executable differs from the verified release binary.'
}

$manifest = Get-Content (Join-Path $repo 'docs/experiments/measurement-campaign-rtx4070.json') -Raw | ConvertFrom-Json
$manifest.implementations = @('nvcr')
$manifest.modes = @('throughput')
$manifest.target_profile = "/workspace/nvcr/configs/targets/rtx$Gpu-laptop-ubuntu2404.json"
$manifest.model_profile = '/workspace/nvcr/configs/models/dcvcrt-cvpr2025.json'
$manifest.nvcr = $binary.path
$manifest.build_dir = '/opt/nvcr/build-evidence-not-distributed'
$manifest.engine_root = '/opt/nvcr/engines'
$manifest.reference_root = '/workspace/nvcr/reference-not-selected'
$manifest.reference_python = '/usr/bin/python3'
$manifest.execution = [ordered]@{
    mode = 'container'; container_digest = $imageId
    image = [ordered]@{ digest = $imageId; id = $imageId; source_revision = $labels.'org.opencontainers.image.revision'; version = '1.0.7'; binary_sha256 = $binary.sha256 }
}
$manifest.provenance_note = 'NVCR-only laptop timing; identical controlled-study timing/warm-up contract. Image identity is separate from the mounted measurement harness.'
foreach ($sequence in $manifest.sequences) {
    $relative = $sequence.path -replace '^\.\./datasets/', ''
    if (!(Test-Path -LiteralPath (Join-Path $InputDir $relative))) { throw "Missing input: $relative" }
    $sequence.path = "/input/$relative"
}
if ($Action -eq 'Check') {
    $manifest.sequences = @($manifest.sequences[0]); $manifest.qps = @(0); $manifest.gops = @(30); $manifest.repetitions = 1
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $ResultsDir 'manifest.json'), ($manifest | ConvertTo-Json -Depth 32), $utf8)

function Invoke-Campaign {
    param([string]$Step, [string]$Output, [string]$PassedSmoke)
    $args = @('run','--rm','--network','none','--gpus','all',
        '--entrypoint','python3','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','GIT_CONFIG_COUNT=1','-e','GIT_CONFIG_KEY_0=safe.directory','-e','GIT_CONFIG_VALUE_0=/workspace/nvcr',
        '--mount',"type=bind,source=$repo,target=/workspace/nvcr,readonly",
        '--mount',"type=bind,source=$InputDir,target=/input,readonly",
        '--mount',"type=bind,source=$ResultsDir,target=/results")
    if ($EngineDir) { $args += @('--mount',"type=bind,source=$EngineDir,target=/opt/nvcr/engines,readonly") }
    else { $args += @('--mount',"type=volume,source=$EngineVolume,target=/opt/nvcr/engines,readonly") }
    if ($PassedSmoke) {
        # Validation regenerates derived summaries; preserve the original smoke.
        $smokeCopy = Join-Path $ResultsDir "$Output.smoke"
        Copy-Item -LiteralPath $PassedSmoke -Destination $smokeCopy -Recurse
        $args += @('--mount',"type=bind,source=$smokeCopy,target=/passed-smoke")
    }
    $args += @($imageId,'/workspace/nvcr/scripts/measurement_campaign.py',$Step,'--manifest','/results/manifest.json','--output',"/results/$Output")
    if ($PassedSmoke) { $args += @('--smoke-evidence','/passed-smoke') }
    $args | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $ResultsDir "$Output.launch.json")
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & docker @args 2>&1 | Tee-Object -FilePath (Join-Path $ResultsDir "$Output.console.log")
    } finally { $ErrorActionPreference = $previousPreference }
    if ($LASTEXITCODE -ne 0) { throw "$Step failed; original outputs and logs are retained." }
}

if (!$SmokeDir) {
    Invoke-Campaign 'smoke' 'smoke' ''
    $SmokeDir = Join-Path $ResultsDir 'smoke'
}
if ($Action -eq 'Check') {
    Invoke-Campaign 'run' 'check' $SmokeDir
    $observations = Get-Content (Join-Path $ResultsDir 'check/observations.jsonl') | ForEach-Object { $_ | ConvertFrom-Json }
    if (@($observations).Count -ne 2) { throw 'Expected one encode/decode timing pair.' }
    foreach ($row in $observations) {
        if ($row.status -ne 'passed' -or $row.process_seconds -le 0 -or $row.process_fps -le 0 -or $row.metrics.codec_seconds -le 0 -or $row.metrics.timed_frames -ne 100 -or $row.metrics.warmup_frames -ne 10) {
            throw 'The bounded check did not retain valid process and codec timings with the required frame counts.'
        }
    }
    Write-Host "CHECK PASSED: both timings saved. Smoke directory: $(Join-Path $ResultsDir 'smoke')"
} else {
    Invoke-Campaign 'run' 'campaign' $SmokeDir
    $analysis = Get-Content (Join-Path $ResultsDir 'campaign/analysis.json') -Raw | ConvertFrom-Json
    if ($analysis.status -ne 'complete' -or $analysis.observed_operations -ne 1440) { throw 'Full timing campaign is incomplete.' }
    Write-Host "COMPLETE: 1440 timing observations. Results: $ResultsDir"
}
