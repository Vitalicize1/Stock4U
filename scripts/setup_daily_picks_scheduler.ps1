# PowerShell script to set up Windows Task Scheduler for Daily Picks
# Run this as Administrator

param(
    [string]$Time = "14:00",  # 2 PM UTC (default)
    [string]$ScriptPath = "D:\Stock4U\scripts\auto_update_daily_picks.bat"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Stock4U Daily Picks Scheduler Setup  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ Error: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Verify script exists
if (-not (Test-Path $ScriptPath)) {
    Write-Host "❌ Error: Script not found at $ScriptPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

try {
    # Create the scheduled task
    $TaskName = "Stock4U-DailyPicks"
    $TaskDescription = "Automatically generates and publishes Stock4U daily picks"
    
    # Create task action
    $Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$ScriptPath`""
    
    # Create task trigger (daily at specified time)
    $Trigger = New-ScheduledTaskTrigger -Daily -At $Time
    
    # Create task settings
    $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    
    # Create task principal (run whether user is logged on or not)
    $Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
    
    # Register the task
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description $TaskDescription -Force
    
    Write-Host "✅ SUCCESS: Scheduled task created!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Task Details:" -ForegroundColor Cyan
    Write-Host "   Name: $TaskName"
    Write-Host "   Schedule: Daily at $Time"
    Write-Host "   Script: $ScriptPath"
    Write-Host ""
    Write-Host "🎯 Your daily picks will now update automatically every day at $Time!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 To verify the task:" -ForegroundColor Yellow
    Write-Host "   1. Open Task Scheduler (taskschd.msc)"
    Write-Host "   2. Look for 'Stock4U-DailyPicks' in the task list"
    Write-Host "   3. Right-click → Run to test immediately"
    Write-Host ""
    Write-Host "📝 Logs will be saved to: D:\Stock4U\logs\daily_picks_auto.log" -ForegroundColor Cyan
    
} catch {
    Write-Host "❌ Error creating scheduled task: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Read-Host "Press Enter to exit"
