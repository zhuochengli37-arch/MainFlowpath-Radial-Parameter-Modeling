# 使用本地 wheel 包在离线环境中安装 Project1。

param(
    [string]$OfflinePackagesPath = "offline_packages",
    [string]$RequirementsFile = "docs/requirements.txt",
    [string]$VenvPath = ".venv",
    [string]$PythonExe = "python",
    [switch]$InstallEditable
)

$manifestPath = Join-Path $OfflinePackagesPath "offline_manifest.json"
$pythonExeWasExplicit = $PSBoundParameters.ContainsKey("PythonExe")
$projectVenvHintPath = ".project_venv_python"

Write-Host "=== Project1 offline deployment ===" -ForegroundColor Green
Write-Host ""

if (-not (Test-Path $RequirementsFile)) {
    Write-Error "Requirements file not found: $RequirementsFile"
    exit 1
}

if (-not (Test-Path $OfflinePackagesPath)) {
    Write-Error "Offline package directory not found: $OfflinePackagesPath"
    exit 1
}

if (-not (Test-Path $manifestPath)) {
    Write-Error "Offline manifest not found: $manifestPath"
    exit 1
}

function Get-PythonVersion {
    param(
        [string]$PythonCommand
    )

    try {
        $versionOutput = & $PythonCommand -c "import platform; print(platform.python_version())" 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        return $versionOutput.Trim()
    } catch {
        return $null
    }
}

function Resolve-PreferredPython {
    param(
        [string]$RequestedPython,
        [string]$ExpectedVersion,
        [bool]$WasExplicit
    )

    $requestedVersion = Get-PythonVersion -PythonCommand $RequestedPython
    if ($requestedVersion -eq $ExpectedVersion) {
        return $RequestedPython
    }

    if ($WasExplicit) {
        if (-not $requestedVersion) {
            Write-Error "Failed to detect Python version from explicitly provided interpreter: $RequestedPython"
        } else {
            Write-Error "Python version mismatch. Offline packages were prepared with $ExpectedVersion, current interpreter is $requestedVersion."
        }
        exit 1
    }

    Write-Warning "Default interpreter '$RequestedPython' did not match offline package version $ExpectedVersion. Attempting to auto-detect a compatible Python interpreter."

    $candidates = New-Object System.Collections.Generic.List[string]

    foreach ($candidate in @(
        ".\python.exe",
        "python",
        "D:\python310\python.exe",
        "C:\Python310\python.exe",
        "C:\Program Files\Python310\python.exe",
        "C:\Program Files\Python Software Foundation\Python 3.10\python.exe",
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe")
    )) {
        if ($candidate -and -not $candidates.Contains($candidate)) {
            $candidates.Add($candidate)
        }
    }

    foreach ($candidate in $candidates) {
        $candidateVersion = $null
        $candidateVersion = Get-PythonVersion -PythonCommand $candidate
        if ($candidateVersion -eq $ExpectedVersion) {
            return $candidate
        }
    }

    Write-Error "Python version mismatch. Offline packages were prepared with $ExpectedVersion, and no compatible Python interpreter was auto-detected. Please install Python $ExpectedVersion and re-run deploy_offline.ps1 with -PythonExe '<path-to-python.exe>'."
    exit 1
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$expectedPythonVersion = [string]$manifest.python_version
$PythonExe = Resolve-PreferredPython -RequestedPython $PythonExe -ExpectedVersion $expectedPythonVersion -WasExplicit $pythonExeWasExplicit
$currentPythonVersion = Get-PythonVersion -PythonCommand $PythonExe
$currentPythonVersion = $currentPythonVersion.Trim()
Write-Host "Detected Python interpreter: $PythonExe" -ForegroundColor Cyan
Write-Host "Detected Python version: $currentPythonVersion" -ForegroundColor Cyan
$currentPythonVersion = $currentPythonVersion.Trim()
if ($currentPythonVersion -ne $expectedPythonVersion) {
    Write-Error "Python version mismatch. Offline packages were prepared with $expectedPythonVersion, current interpreter is $currentPythonVersion."
    exit 1
}

function Ensure-VenvPip {
    param(
        [string]$PythonPath,
        [string]$PackagesPath
    )

    & $PythonPath -m pip --version | Out-Null
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Warning "pip is not available in the target virtual environment. Attempting offline bootstrap."
    $pipWheel = Get-ChildItem -Path $PackagesPath -Filter "pip-*.whl" | Sort-Object Name -Descending | Select-Object -First 1
    $setuptoolsWheel = Get-ChildItem -Path $PackagesPath -Filter "setuptools-*.whl" | Sort-Object Name -Descending | Select-Object -First 1
    $wheelPackage = Get-ChildItem -Path $PackagesPath -Filter "wheel-*.whl" | Sort-Object Name -Descending | Select-Object -First 1

    if (-not $pipWheel -or -not $setuptoolsWheel) {
        Write-Error "Offline pip bootstrap requires pip and setuptools wheels in $PackagesPath. Re-run prepare_offline_packages.ps1."
        exit 1
    }

    $bootstrapArgs = @(
        "install",
        "--no-index",
        "--find-links", $PackagesPath,
        "pip",
        "setuptools"
    )
    if ($wheelPackage) {
        $bootstrapArgs += "wheel"
    }
    $bootstrapArgsLiteral = ($bootstrapArgs | ForEach-Object {
        "'" + (($_ -replace "\\", "\\\\") -replace "'", "\\'") + "'"
    }) -join ", "
    $bootstrapScript = @"
import json
import runpy
import sys
sys.path.insert(0, r'$($setuptoolsWheel.FullName)')
sys.path.insert(0, r'$($pipWheel.FullName)')
sys.argv = ['pip', $bootstrapArgsLiteral]
runpy.run_module('pip', run_name='__main__', alter_sys=True)
"@
    & $PythonPath -c $bootstrapScript
    if ($LASTEXITCODE -ne 0) {
        & $PythonPath -m pip --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to bootstrap pip from offline wheels."
            exit 1
        }
        Write-Warning "Offline pip bootstrap returned a non-zero exit code, but pip is available. Continuing."
    }

    & $PythonPath -m pip --version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip is still unavailable after offline bootstrap."
        exit 1
    }
}

Write-Host "1. Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path $VenvPath)) {
    & $PythonExe -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        $venvPythonAfterFailure = Join-Path $VenvPath "Scripts\\python.exe"
        if (-not (Test-Path $venvPythonAfterFailure)) {
            Write-Error "Failed to create virtual environment."
            exit 1
        }
        Write-Warning "Virtual environment creation returned a non-zero exit code, but the interpreter was created. Continuing with the existing environment."
    }
}

Write-Host "2. Resolving virtual environment interpreter..." -ForegroundColor Yellow
$venvPython = Join-Path $VenvPath "Scripts\\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment interpreter not found: $venvPython"
    exit 1
}
Ensure-VenvPip -PythonPath $venvPython -PackagesPath $OfflinePackagesPath

Write-Host "2.1 Saving project interpreter hint..." -ForegroundColor Yellow
$resolvedVenvPython = [System.IO.Path]::GetFullPath($venvPython)
Set-Content -Path $projectVenvHintPath -Value $resolvedVenvPython -Encoding UTF8

Write-Host "3. Upgrading pip from local wheels..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip --no-index --find-links=$OfflinePackagesPath
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Pip upgrade failed, continuing with existing pip."
}

Write-Host "4. Installing dependencies..." -ForegroundColor Yellow
& $venvPython -m pip install --no-index --find-links=$OfflinePackagesPath -r $RequirementsFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dependency installation failed."
    exit 1
}

if ($InstallEditable) {
    Write-Host "5. Installing the project in editable mode..." -ForegroundColor Yellow
    & $venvPython -m pip install --no-index --find-links=$OfflinePackagesPath -e .
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Editable install failed, but dependencies are installed."
        Write-Warning "For direct script execution from the project root, this warning usually does not block usage."
    }
}

Write-Host "6. Verifying installation..." -ForegroundColor Yellow
try {
    & $venvPython -c "import numpy, scipy, sklearn, yaml, tqdm, xgboost, lightgbm; print('Core dependencies imported successfully.')"
    & $venvPython -c "import json, platform; from pathlib import Path; manifest = json.loads(Path(r'$manifestPath').read_text(encoding='utf-8-sig')); current = platform.python_version(); assert current == manifest['python_version'], (current, manifest['python_version']); print('Python version matches offline manifest.')"
    & $venvPython -c "import sys; from pathlib import Path; root = Path.cwd(); sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'workbase' / 'src')); import project1; import workbase.common.config_loader; print('Project modules imported successfully.')"
    & $venvPython -c "import sys; from pathlib import Path; root = Path.cwd(); sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'workbase' / 'src')); import project1.cli; import project1.experiments.benchmark_1d_runner; import workbase.common.generic_tabular; print('Entrypoint modules imported successfully.')"
    & $venvPython workbase/generic_scripts/generic_predict_1d.py --help | Out-Null
    Write-Host "Entrypoint check passed." -ForegroundColor Green
} catch {
    Write-Error "Installation verification failed: $_"
    exit 1
}

Write-Host ""
Write-Host "=== Offline deployment complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Recommended next steps:" -ForegroundColor Cyan
Write-Host "0. If you prefer direct script execution, keep using this project's root directory as the working directory."
Write-Host "1. Activate the virtual environment: $VenvPath\Scripts\Activate.ps1"
Write-Host "2. In PyCharm, set the project interpreter to $resolvedVenvPython"
Write-Host "3. Run a 1D benchmark script: python workbase/current_scripts/benchmark_1d.py"
Write-Host "4. Run a 1D prediction script: python workbase/current_scripts/predict_1d.py"
Write-Host "5. Unified CLI is also available: python -m project1 --help"
Write-Host ""
Write-Host "If something fails, inspect the offline_packages directory first." -ForegroundColor Yellow
