# GPT-OSS Agent - one-click installer for Windows
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/Luzhen-0502/gptoss-client/main/install.ps1 | iex

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Repo       = 'Luzhen-0502/gptoss-client'
$InstallDir = Join-Path $env:USERPROFILE 'GPTOSS-Agent'
$ExeUrl     = "https://github.com/$Repo/releases/latest/download/GPTOSS-Agent.exe"
$BatUrl     = "https://raw.githubusercontent.com/$Repo/main/%E4%B8%80%E9%94%AE%E4%B8%8B%E8%BD%BD%E6%A8%A1%E5%9E%8B.bat"

function Info($m) { Write-Host $m -ForegroundColor Cyan }
function Ok($m)   { Write-Host $m -ForegroundColor Green }
function Warn($m) { Write-Host $m -ForegroundColor Yellow }

Info "=================================================="
Info "  GPT-OSS Agent installer"
Info "  install dir: $InstallDir"
Info "=================================================="

if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}
Set-Location $InstallDir

# ---------------------------------------------------------------- 1. exe
Warn "[1/4] downloading GPTOSS-Agent.exe (about 305 MB)..."
$exePath = Join-Path $InstallDir 'GPTOSS-Agent.exe'
try {
    Invoke-WebRequest -Uri $ExeUrl -OutFile $exePath -UseBasicParsing
} catch {
    Write-Host "download failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "check https://github.com/$Repo/releases and download manually." -ForegroundColor Red
    return
}
$sizeMb = [math]::Round((Get-Item $exePath).Length / 1MB, 1)
Ok "      OK  $sizeMb MB"

# ------------------------------------------------------- 2. model script
Warn "[2/4] downloading model downloader script..."
$batPath = Join-Path $InstallDir 'download-models.bat'
try {
    Invoke-WebRequest -Uri $BatUrl -OutFile $batPath -UseBasicParsing
    Ok "      OK  saved as download-models.bat"
} catch {
    Warn "      skip (you can download models manually, see models/README.md)"
}

# ------------------------------------------------- 3. desktop shortcut
Warn "[3/4] creating desktop shortcut..."
try {
    $desktop = [Environment]::GetFolderPath('Desktop')
    $lnkPath = Join-Path $desktop 'GPT-OSS Agent.lnk'
    $wsh = New-Object -ComObject WScript.Shell
    $lnk = $wsh.CreateShortcut($lnkPath)
    $lnk.TargetPath = $exePath
    $lnk.WorkingDirectory = $InstallDir
    $lnk.IconLocation = "$exePath,0"
    $lnk.Description = 'GPT-OSS Agent'
    $lnk.Save()
    Ok "      OK  desktop: GPT-OSS Agent"
} catch {
    Warn "      shortcut failed: $($_.Exception.Message)"
}

# --------------------------------------------------------- 4. models
Warn "[4/4] models are about 19 GB and NOT included in the exe."
Write-Host "      (1) Download 3 AI models now  -> about 19 GB, takes a while"
Write-Host "      (2) Skip -> run download-models.bat later"
$ans = Read-Host "      Download models now? [y/N]"
if ($ans -match '^[Yy]') {
    if (Test-Path $batPath) {
        Info "starting model downloader (close the window when it finishes)..."
        Start-Process -FilePath $batPath -WorkingDirectory $InstallDir -Wait
    } else {
        Warn "download-models.bat not found, please download models manually."
    }
} else {
    Info "skipped. double-click download-models.bat when you are ready."
}

Write-Host ""
Ok "=================================================="
Ok "  installed!"
Ok "  app: $exePath"
Ok "  models: $InstallDir\models   (needed before first run)"
Ok "  desktop shortcut: GPT-OSS Agent"
Ok "=================================================="
Info "First launch takes 1-2 minutes (unpacking). Be patient."
