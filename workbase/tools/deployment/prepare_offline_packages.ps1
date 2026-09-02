# 在联网环境中准备离线依赖包。

param(
    [string]$OutputPath = "offline_packages",
    [string]$RequirementsFile = "docs/requirements.txt",
    [switch]$IncludeProject
)

$manifestPath = Join-Path $OutputPath "offline_manifest.json"

Write-Host "=== Project1 offline package preparation ===" -ForegroundColor Green
Write-Host ""

if (-not (Test-Path $RequirementsFile)) {
    Write-Error "Requirements file not found: $RequirementsFile"
    exit 1
}

Write-Host "1. Creating output directory: $OutputPath" -ForegroundColor Yellow
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath | Out-Null
}

Write-Host "2. Downloading dependency wheels..." -ForegroundColor Yellow
python -m pip download -r $RequirementsFile -d $OutputPath
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download dependency packages."
    exit 1
}

Write-Host "3. Downloading bootstrap packaging tools..." -ForegroundColor Yellow
python -m pip download pip setuptools wheel -d $OutputPath
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download pip/setuptools/wheel for offline bootstrap."
    exit 1
}

if ($IncludeProject) {
    Write-Host "4. Downloading the project package..." -ForegroundColor Yellow
    python -m pip download -e . -d $OutputPath
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Project package download failed, but dependency packages are ready."
    }
}

Write-Host "5. Writing offline manifest..." -ForegroundColor Yellow
$pythonVersion = python -c "import platform; print(platform.python_version())"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to detect Python version."
    exit 1
}
$requirements = Get-Content $RequirementsFile | ForEach-Object { $_.Trim() } | Where-Object {
    $_ -and -not $_.StartsWith('#')
}
$manifest = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    python_version = $pythonVersion.Trim()
    requirements_file = $RequirementsFile
    requirements = $requirements
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "6. Verifying downloaded files..." -ForegroundColor Yellow
$packageCount = (Get-ChildItem $OutputPath -File).Count
Write-Host "Downloaded $packageCount package files." -ForegroundColor Green

Write-Host ""
Write-Host "Downloaded packages:" -ForegroundColor Cyan
Get-ChildItem $OutputPath -Name | ForEach-Object { Write-Host "  - $_" }

Write-Host ""
Write-Host "=== Offline package preparation complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Copy the whole project directory to the offline machine."
Write-Host "2. Copy the $OutputPath directory to the offline machine."
Write-Host "3. In PyCharm, later point the project interpreter to .venv\\Scripts\\python.exe"
Write-Host "4. Run: powershell -ExecutionPolicy Bypass -File 'workbase/tools/deployment/deploy_offline.ps1'"
Write-Host ""
Write-Host "Make sure the offline Python version matches the environment used for the downloads." -ForegroundColor Yellow
