# download_glad.ps1
# Script to download and setup GLAD for VisualizedMD4

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GLAD Download Helper for VisualizedMD4" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if glad directory exists
$gladDir = Join-Path $PSScriptRoot "glad"
$gladInclude = Join-Path $gladDir "include\glad"
$gladKHR = Join-Path $gladDir "include\KHR"
$gladSrc = Join-Path $gladDir "src"

# Create directories if they don't exist
New-Item -ItemType Directory -Force -Path $gladInclude | Out-Null
New-Item -ItemType Directory -Force -Path $gladKHR | Out-Null
New-Item -ItemType Directory -Force -Path $gladSrc | Out-Null

Write-Host "This script will guide you to download GLAD manually." -ForegroundColor Yellow
Write-Host ""
Write-Host "Please follow these steps:" -ForegroundColor Green
Write-Host ""
Write-Host "1. Open your browser and go to:" -ForegroundColor White
Write-Host "   https://glad.dav1d.de/" -ForegroundColor Blue
Write-Host ""
Write-Host "2. Use these settings:" -ForegroundColor White
Write-Host "   - Language: C/C++" -ForegroundColor Gray
Write-Host "   - Specification: OpenGL" -ForegroundColor Gray
Write-Host "   - Profile: Core" -ForegroundColor Gray
Write-Host "   - gl (API): Version 3.3" -ForegroundColor Gray
Write-Host "   - Generate a loader: CHECKED" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Click 'GENERATE' button" -ForegroundColor White
Write-Host ""
Write-Host "4. Download the ZIP file (glad.zip)" -ForegroundColor White
Write-Host ""
Write-Host "5. Extract and copy files to:" -ForegroundColor White
Write-Host "   - glad.h -> $gladInclude\glad.h" -ForegroundColor Gray
Write-Host "   - khrplatform.h -> $gladKHR\khrplatform.h" -ForegroundColor Gray
Write-Host "   - glad.c -> $gladSrc\glad.c" -ForegroundColor Gray
Write-Host ""

# Try to open browser
$openBrowser = Read-Host "Would you like to open the GLAD website now? (y/n)"
if ($openBrowser -eq "y" -or $openBrowser -eq "Y") {
    Start-Process "https://glad.dav1d.de/"
    Write-Host ""
    Write-Host "Browser opened. Please follow the instructions above." -ForegroundColor Green
}

Write-Host ""
Write-Host "After downloading and extracting, press any key to verify..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Verify installation
$gladH = Join-Path $gladInclude "glad.h"
$khrH = Join-Path $gladKHR "khrplatform.h"
$gladC = Join-Path $gladSrc "glad.c"

$allFound = $true

if (Test-Path $gladH) {
    Write-Host "[OK] glad.h found" -ForegroundColor Green
} else {
    Write-Host "[MISSING] glad.h not found at $gladH" -ForegroundColor Red
    $allFound = $false
}

if (Test-Path $khrH) {
    Write-Host "[OK] khrplatform.h found" -ForegroundColor Green
} else {
    Write-Host "[MISSING] khrplatform.h not found at $khrH" -ForegroundColor Red
    $allFound = $false
}

if (Test-Path $gladC) {
    Write-Host "[OK] glad.c found" -ForegroundColor Green
} else {
    Write-Host "[MISSING] glad.c not found at $gladC" -ForegroundColor Red
    $allFound = $false
}

Write-Host ""
if ($allFound) {
    Write-Host "SUCCESS! GLAD is properly installed." -ForegroundColor Green
    Write-Host "You can now build the project with CMake." -ForegroundColor Green
} else {
    Write-Host "GLAD installation incomplete. Please check the missing files." -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
