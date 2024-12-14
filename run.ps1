# YouTube Video Summarizer PowerShell Runner
# This script activates the virtual environment and runs the summarizer

# Stop on first error
$ErrorActionPreference = "Stop"

try {
    Write-Host "YouTube Video Summarizer" -ForegroundColor Green
    Write-Host "========================" -ForegroundColor Green
    
    # Check if venv exists
    if (-not (Test-Path ".\venv")) {
        Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
        if (-not $?) { throw "Failed to create virtual environment" }
    }
    
    # Activate virtual environment
    Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
    .\venv\Scripts\Activate
    if (-not $?) { throw "Failed to activate virtual environment" }
    
    # Check if requirements are installed
    if (-not (Test-Path ".\venv\Lib\site-packages\yt_dlp")) {
        Write-Host "`nInstalling requirements..." -ForegroundColor Yellow
        pip install -r requirements.txt
        if (-not $?) { throw "Failed to install requirements" }
    }
    
    # Run the summarizer
    Write-Host "`nStarting YouTube Video Summarizer..." -ForegroundColor Yellow
    python main.py
    
} catch {
    Write-Host "`nError: $_" -ForegroundColor Red
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Make sure Python 3.12+ is installed and in your PATH"
    Write-Host "2. Make sure FFmpeg is installed and in your PATH"
    Write-Host "3. If using GPU acceleration, ensure CUDA 12.1 and cuDNN 9.x are installed"
    Write-Host "4. Check that all required environment variables are set in .env file"
} finally {
    # Deactivate virtual environment if it's active
    if (Test-Path Function:\deactivate) {
        deactivate
    }
    
    Write-Host "`nPress any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
