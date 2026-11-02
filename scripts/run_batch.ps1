param(
    [string]$ImageDir = "data\images"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$AbsoluteImageDir = Join-Path $ProjectRoot $ImageDir
$Images = Get-ChildItem -LiteralPath $AbsoluteImageDir -File |
    Where-Object {
        $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg", ".png")
    } |
    Sort-Object Name

if ($Images.Count -eq 0) {
    throw "No images found in: $AbsoluteImageDir"
}

Write-Host "Found $($Images.Count) images." -ForegroundColor Cyan

foreach ($Image in $Images) {
    $Stem = $Image.BaseName

    Write-Host ""
    Write-Host "Processing $($Image.Name)" -ForegroundColor Green

    $YoloStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    python -m src.yolo_demo $Image.FullName
    $YoloStopwatch.Stop()

    if ($LASTEXITCODE -ne 0) {
        throw "YOLO failed for $($Image.Name)"
    }

    $YoloJson = "outputs\yolo\${Stem}_yolo.json"

    $SamStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    python -m src.sam_segment `
        $Image.FullName `
        --detections $YoloJson
    $SamStopwatch.Stop()

    if ($LASTEXITCODE -ne 0) {
        throw "SAM failed for $($Image.Name)"
    }

    $SamImage = "outputs\sam\${Stem}_sam.jpg"
    $SamJson = "outputs\sam\${Stem}_sam.json"

    $VlmStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    python -m src.vlm_describe $Image.FullName
    $VlmStopwatch.Stop()

    if ($LASTEXITCODE -ne 0) {
        throw "VLM failed for $($Image.Name)"
    }

    $VlmJson = "outputs\vlm\${Stem}_vlm.json"

    python -m src.compare_results `
        $Image.FullName `
        --sam-image $SamImage `
        --yolo-json $YoloJson `
        --sam-json $SamJson `
        --vlm-json $VlmJson

    if ($LASTEXITCODE -ne 0) {
        throw "Comparison failed for $($Image.Name)"
    }

    $TimingDir = Join-Path $ProjectRoot "outputs\timing"
    New-Item -ItemType Directory -Force -Path $TimingDir | Out-Null

    $Timing = [ordered]@{
        image = $Image.Name
        yolo_seconds = [math]::Round($YoloStopwatch.Elapsed.TotalSeconds, 3)
        sam_seconds = [math]::Round($SamStopwatch.Elapsed.TotalSeconds, 3)
        vlm_seconds = [math]::Round($VlmStopwatch.Elapsed.TotalSeconds, 3)
        traditional_total_seconds = [math]::Round(
            $YoloStopwatch.Elapsed.TotalSeconds +
            $SamStopwatch.Elapsed.TotalSeconds,
            3
        )
    }

    $TimingPath = Join-Path $TimingDir "${Stem}_timing.json"
    $Timing |
        ConvertTo-Json |
        Set-Content -LiteralPath $TimingPath -Encoding UTF8

    Start-Sleep -Seconds 3
}

Write-Host ""
Write-Host "Batch processing complete." -ForegroundColor Green