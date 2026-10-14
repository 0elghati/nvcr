param([Parameter(Mandatory=$true)][string]$TestRoot)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$launcher=Join-Path $repo 'scripts/measurement_laptop.ps1'
New-Item -ItemType Directory -Path $TestRoot -Force | Out-Null
$inputs=Join-Path $TestRoot 'inputs'
$manifest=Get-Content (Join-Path $repo 'docs/experiments/measurement-campaign-rtx4070.json') -Raw | ConvertFrom-Json
foreach($s in $manifest.sequences) {
    $file=Join-Path $inputs ($s.path -replace '^\.\./datasets/','')
    New-Item -ItemType Directory -Path (Split-Path $file) -Force | Out-Null
    [IO.File]::WriteAllBytes($file,[byte[]]@(0))
}
$global:MockSteps=New-Object System.Collections.Generic.List[string]
$global:MockFailSmoke=$false
function global:docker {
    $a=@($args); $global:LASTEXITCODE=0
    if($a[0] -eq 'image') {
        @{Id=('sha256:'+'a'*64);Config=@{Labels=@{'org.opencontainers.image.revision'='6a66db7a98081f9210dedb49d768899bd4e9f2b7';'org.opencontainers.image.version'='1.0.7'}}} | ConvertTo-Json -Depth 5 -Compress
        return
    }
    if($a -contains '-c') {
        @{path='/opt/nvcr/bin/nvcr';sha256='8d12a772ede71839d9c3bfdbabd43da10e917d6bcdaa52199f3c45dc49dd69b0'} | ConvertTo-Json -Compress
        return
    }
    $i=[Array]::IndexOf($a,'/workspace/nvcr/scripts/measurement_campaign.py')
    $step=$a[$i+1]; $global:MockSteps.Add($step)
    if($step -eq 'smoke' -and $global:MockFailSmoke) { $global:LASTEXITCODE=1; return }
    $mount=@($a | Where-Object {$_ -match '^type=bind,source=.*target=/results$'})[0]
    $root=$mount -replace '^type=bind,source=','' -replace ',target=/results$',''
    $output=$a[[Array]::IndexOf($a,'--output')+1] -replace '^/results/',''
    $dir=Join-Path $root $output
    New-Item -ItemType Directory -Path $dir | Out-Null
    if($step -eq 'run') {
        if(!($a -contains '--smoke-evidence')) { throw 'Missing smoke binding' }
        @{status='complete';observed_operations=1440} | ConvertTo-Json | Set-Content (Join-Path $dir 'analysis.json')
    }
}
try {
    $empty=Join-Path $TestRoot 'empty-results'
    New-Item -ItemType Directory -Path $empty | Out-Null
    & $launcher -Gpu 3050 -Action Run -InputDir $inputs -ResultsDir $empty
    if(($global:MockSteps -join ',') -ne 'smoke,run') { throw 'Run did not generate and bind its fresh smoke' }
    $saved=Get-Content (Join-Path $empty 'manifest.json') -Raw | ConvertFrom-Json
    if($saved.repetitions -ne 10 -or $saved.sequences.Count -ne 6 -or $saved.qps.Count -ne 4 -or $saved.gops.Count -ne 3) { throw 'Full matrix changed' }
    $before=$global:MockSteps.Count
    $rejected=$false
    try { & $launcher -Gpu 3050 -Action Run -InputDir $inputs -ResultsDir $empty } catch { $rejected=$true }
    if(!$rejected -or $global:MockSteps.Count -ne $before) { throw 'Nonempty evidence was not protected' }
    $global:MockSteps.Clear(); $global:MockFailSmoke=$true
    $failed=$false
    try { & $launcher -Gpu 3050 -Action Run -InputDir $inputs -ResultsDir (Join-Path $TestRoot 'failed-smoke') } catch { $failed=$true }
    if(!$failed -or ($global:MockSteps -join ',') -ne 'smoke') { throw 'Campaign launched after a failed smoke' }
    Write-Host 'PASS: mocked launcher accepts empty folder, runs fresh smoke before campaign, protects existing evidence, stops on failed smoke, and preserves the 1440-operation matrix.'
} finally { Remove-Item Function:\docker }
