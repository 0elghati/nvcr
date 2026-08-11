<#
.SYNOPSIS
Runs the NVCR Docker resolution matrix from a Windows host checkout.

.EXAMPLE
& .\scripts\benchmark_docker.ps1 `
    -Image omarelghati/nvcr:0.19.1-amd64-cuda12.8-trt10.9 `
    -InputDir C:\research\nvcr\datasets `
    -EngineVolume C:\research\nvcr\build\engines\dcvcrt-cvpr2025 `
    -ResultsDir C:\research\nvcr\evidence\performance\rtx5060-docker `
    -Hardware rtx5060-docker `
    -MatrixArgs @('--resolutions', 'qcif 720p', '--frames', '300', '--qp', '32', '--gops', '1 299', '--repetitions', '3', '--warmup-frames', '10')

Use -SkipPull to use an image already present locally. Use -MatrixArgs to pass
options to scripts/benchmark_resolution_matrix.sh.

Before running, the launcher checks the engine volume for the profiles named
in -MatrixArgs' --resolutions (or the matrix default) and installs any that
are missing from the rolling asset catalog. Use -InstallProfiles to force a
specific list, or -NoInstall to skip this check entirely.
#>
[CmdletBinding()]
param(
    [string]$Image,
    [string]$EngineVolume = "nvcr-engines",
    [string]$InputDir,
    [string]$ResultsDir,
    [string]$OutputDir,
    [string]$Hardware = "",
    [string]$ContainerUser = "0:0",
    [string]$InstallProfiles = "",
    [switch]$NoInstall,
    [string]$HostRepoDir = "",
    [switch]$SkipPull,
    [string[]]$MatrixArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$version = (Get-Content (Join-Path $repoRoot "version.txt") -Raw).Trim()

if ([string]::IsNullOrWhiteSpace($Image)) {
    $Image = "omarelghati/nvcr:{0}-amd64-cuda12.8-trt10.9" -f $version
}
if ([string]::IsNullOrWhiteSpace($InputDir)) {
    $InputDir = Join-Path $repoRoot "datasets"
}
if ([string]::IsNullOrWhiteSpace($ResultsDir)) {
    $ResultsDir = Join-Path $repoRoot "evidence/performance/docker-$timestamp"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $ResultsDir "streams"
}

function Resolve-HostDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Create
    )

    $candidate = $Path
    if (-not [IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path (Get-Location).Path $candidate
    }
    if ($Create) {
        New-Item -ItemType Directory -Force -Path $candidate | Out-Null
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
        throw "Directory does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker is required on PATH"
}

$InputDir = Resolve-HostDirectory $InputDir
$ResultsDir = Resolve-HostDirectory $ResultsDir -Create
$OutputDir = Resolve-HostDirectory $OutputDir -Create

$repoMountSource = $repoRoot
if (-not [string]::IsNullOrWhiteSpace($HostRepoDir)) {
    $repoMountSource = Resolve-HostDirectory $HostRepoDir
}

$engineIsHostDirectory = Test-Path -LiteralPath $EngineVolume -PathType Container
$engineSource = $EngineVolume
if ($engineIsHostDirectory) {
    $engineSource = Resolve-HostDirectory $EngineVolume
}

$engineContainerRoot = "/opt/nvcr/engines"
if ($engineIsHostDirectory -and -not (Test-Path (Join-Path $engineSource "dcvcrt-qcif"))) {
    $nestedEngineDirs = @(Get-ChildItem -LiteralPath $engineSource -Directory)
    if ($nestedEngineDirs.Count -eq 1 -and (Test-Path (Join-Path $nestedEngineDirs[0].FullName "dcvcrt-qcif"))) {
        $engineContainerRoot = "/opt/nvcr/engines/{0}" -f $nestedEngineDirs[0].Name
    }
}

$engineRuntimeMountArgs = @()
if ($engineIsHostDirectory) {
    $engineRuntimeMountArgs += @("--mount", "type=bind,source=$engineSource,target=/opt/nvcr/engines,readonly")
} else {
    $engineRuntimeMountArgs += @("-v", "$EngineVolume`:/opt/nvcr/engines:ro")
}

$commit = (& git -C $repoRoot rev-parse HEAD 2>$null | Select-Object -First 1)
if ([string]::IsNullOrWhiteSpace($commit)) {
    $commit = "unknown"
}
$dirty = -not [string]::IsNullOrWhiteSpace((& git -C $repoRoot status --short 2>$null | Out-String))

if (-not $SkipPull) {
    Write-Host "Pulling Docker image: $Image"
    & docker pull $Image
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$imageDigest = (& docker image inspect --format '{{index .RepoDigests 0}}' $Image 2>$null | Select-Object -First 1)
if ([string]::IsNullOrWhiteSpace($imageDigest)) {
    $imageDigest = (& docker image inspect --format '{{.Id}}' $Image 2>$null | Select-Object -First 1)
}
if ([string]::IsNullOrWhiteSpace($imageDigest)) {
    throw "Cannot resolve Docker image identity: $Image"
}
$imageDigest = $imageDigest.Trim()

if ($MatrixArgs.Count -gt 0 -and $MatrixArgs[0] -eq "--") {
    $MatrixArgs = @($MatrixArgs | Select-Object -Skip 1)
}

function Test-EngineProfile {
    param([Parameter(Mandatory = $true)][string]$ProfileName)

    $testArgs = @("run", "--rm") + $engineRuntimeMountArgs + @(
        "--entrypoint", "/bin/bash",
        $Image,
        "-c",
        "test -f '$engineContainerRoot/dcvcrt-$ProfileName/engine_manifest.json' -o " +
            "-f '$engineContainerRoot/profiles/dcvcrt/$ProfileName/engine_manifest.json'"
    )
    & docker @testArgs | Out-Null
    return $LASTEXITCODE -eq 0
}

if (-not $NoInstall -and [string]::IsNullOrWhiteSpace($InstallProfiles)) {
    $matrixResolutions = "qcif cif 720p 1080p"
    for ($index = 0; $index -lt $MatrixArgs.Count; $index++) {
        if ($MatrixArgs[$index] -eq "--resolutions" -and ($index + 1) -lt $MatrixArgs.Count) {
            $matrixResolutions = $MatrixArgs[$index + 1]
        }
    }
    $missingProfiles = @($matrixResolutions -split "\s+" | Where-Object { $_ } | Where-Object { -not (Test-EngineProfile $_) })
    if ($missingProfiles.Count -gt 0) {
        Write-Host "Engine profiles not found in $EngineVolume, installing: $($missingProfiles -join ' ')"
        $InstallProfiles = $missingProfiles -join " "
    } else {
        Write-Host "Engine profiles already installed: $matrixResolutions"
    }
}

if (-not [string]::IsNullOrWhiteSpace($InstallProfiles)) {
    $installArgs = @("run", "--rm", "--gpus", "all")
    if ($engineIsHostDirectory) {
        $installArgs += @("--mount", "type=bind,source=$engineSource,target=/opt/nvcr/engines")
    } else {
        $installArgs += @("-v", "$EngineVolume`:/opt/nvcr/engines")
    }
    if ($env:GH_TOKEN) {
        $installArgs += @("-e", "GH_TOKEN=$($env:GH_TOKEN)")
    }
    $installArgs += @(
        "--entrypoint", "/opt/nvcr/bin/nvcr-artifacts",
        $Image,
        "install",
        "--engine-root",
        $engineContainerRoot
    )
    if ($InstallProfiles -eq "all") {
        $installArgs += "--all"
    } else {
        foreach ($profile in ($InstallProfiles -split "\s+" | Where-Object { $_ })) {
            $installArgs += @("--profile", $profile)
        }
    }
    Write-Host "Installing Docker engine profiles: $InstallProfiles"
    & docker @installArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$matrixScript = Join-Path $repoRoot "scripts/benchmark_resolution_matrix.sh"
$temporaryMatrixScript = Join-Path ([IO.Path]::GetTempPath()) "nvcr-benchmark-resolution-$PID.sh"
$scriptText = [IO.File]::ReadAllText($matrixScript) -replace "`r`n?", "`n"
[IO.File]::WriteAllText($temporaryMatrixScript, $scriptText, [Text.UTF8Encoding]::new($false))

$containerEnv = @(
    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    "NVIDIA_VISIBLE_DEVICES=all",
    "NVCR_BIN=/opt/nvcr/bin/nvcr",
    "NVCR_BENCH_DATA_ROOT=/input",
    "NVCR_BENCH_ENGINE_ROOT=$engineContainerRoot",
    "NVCR_ENGINE_ROOT=$engineContainerRoot",
    "NVCR_BENCH_OUTPUT_DIR=/output",
    "NVCR_BENCH_RESULTS_DIR=/results",
    "NVCR_BENCH_EXECUTION_MODE=docker",
    "NVCR_BENCH_CONTAINER_IMAGE=$Image",
    "NVCR_BENCH_CONTAINER_DIGEST=$imageDigest",
    "NVCR_BENCH_COMMIT=$commit",
    "NVCR_BENCH_DIRTY=$($dirty.ToString().ToLowerInvariant())",
    "NVCR_BENCH_HARDWARE=$Hardware",
    "NVCR_BENCH_MEMORY_PROFILER=/workspace/nvcr/scripts/profile_memory_command.py"
)

$dockerArgs = @("run", "--rm", "--gpus", "all", "--user", $ContainerUser, "--entrypoint", "/bin/bash")
foreach ($environment in $containerEnv) {
    $dockerArgs += @("-e", $environment)
}
if ($env:GH_TOKEN) {
    $dockerArgs += @("-e", "GH_TOKEN=$($env:GH_TOKEN)")
}

$dockerArgs += $engineRuntimeMountArgs
$dockerArgs += @(
    "--mount", "type=bind,source=$InputDir,target=/input,readonly",
    "--mount", "type=bind,source=$ResultsDir,target=/results",
    "--mount", "type=bind,source=$OutputDir,target=/output",
    "--mount", "type=bind,source=$repoMountSource,target=/workspace/nvcr,readonly",
    "--mount", "type=bind,source=$temporaryMatrixScript,target=/tmp/benchmark_resolution_matrix.sh,readonly",
    "-w", "/workspace/nvcr",
    $Image,
    "/tmp/benchmark_resolution_matrix.sh"
)
$dockerArgs += $MatrixArgs
$dockerArgs += @(
    "--output-dir", "/output",
    "--results-dir", "/results",
    "--jsonl", "/results/results.jsonl",
    "--csv", "/results/results.csv",
    "--report", "/results/summary.md"
)

Write-Host "Running Docker benchmark: image=$Image digest=$imageDigest"
Write-Host "Results: $ResultsDir"
Write-Host "Streams: $OutputDir"
try {
    & docker @dockerArgs
    $exitCode = $LASTEXITCODE
} finally {
    Remove-Item -LiteralPath $temporaryMatrixScript -Force -ErrorAction SilentlyContinue
}
exit $exitCode