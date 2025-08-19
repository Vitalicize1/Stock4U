# Automatic Daily Picks Scheduler Setup

This guide sets up automatic daily updates for your Stock4U daily picks using Windows Task Scheduler.

## Quick Setup (Recommended)

### Step 1: Run the Setup Script
1. **Right-click PowerShell** and select **"Run as Administrator"**
2. Navigate to your Stock4U directory:
   ```powershell
   cd "D:\Stock4U"
   ```
3. Run the setup script:
   ```powershell
   .\scripts\setup_daily_picks_scheduler.ps1
   ```

### Step 2: Verify the Task
1. Open **Task Scheduler** (press `Win+R`, type `taskschd.msc`)
2. Look for **"Stock4U-DailyPicks"** in the task list
3. Right-click → **"Run"** to test it immediately

## Configuration Options

### Change the Update Time
By default, picks update at **2:00 PM UTC** (good for US market hours). To change:

```powershell
# Update at 9:30 AM UTC (market open)
.\scripts\setup_daily_picks_scheduler.ps1 -Time "09:30"

# Update at 6:00 PM UTC (after market close)
.\scripts\setup_daily_picks_scheduler.ps1 -Time "18:00"
```

### Time Zone Notes
- **UTC 14:00** = **9:00 AM EST** / **6:00 AM PST**
- **UTC 18:00** = **1:00 PM EST** / **10:00 AM PST**
- **UTC 09:30** = **4:30 AM EST** / **1:30 AM PST**

## What Happens Automatically

1. **Daily Execution**: Runs every day at the scheduled time
2. **Generates Fresh Picks**: Analyzes current market data
3. **Publishes to Gist**: Updates the public URL automatically
4. **Logs Results**: Saves to `logs/daily_picks_auto.log`
5. **No User Interaction**: Runs even when you're not logged in

## Monitoring

### Check Logs
View the automatic update log:
```cmd
type "D:\Stock4U\logs\daily_picks_auto.log"
```

### Task Scheduler Status
1. Open Task Scheduler (`taskschd.msc`)
2. Find "Stock4U-DailyPicks"
3. Check "Last Run Result" and "Next Run Time"

### Test Run
Force an immediate update:
1. Open Task Scheduler
2. Right-click "Stock4U-DailyPicks"
3. Select "Run"

## Troubleshooting

### Task Not Running
- **Check if PC is on**: Task only runs when computer is powered on
- **Check user permissions**: Task runs as SYSTEM account
- **Check script path**: Ensure `D:\Stock4U\scripts\auto_update_daily_picks.bat` exists

### Picks Not Updating
1. Check the log file for errors
2. Verify GitHub token is still valid
3. Test manual run: `python utils/publish_daily_picks.py`

### GitHub Token Expired
If your token expires, update the script:
1. Edit `scripts/auto_update_daily_picks.bat`
2. Replace the `GITHUB_TOKEN` value with a new token
3. No need to recreate the scheduled task

## Manual Management

### Remove the Scheduled Task
```powershell
# Run as Administrator
Unregister-ScheduledTask -TaskName "Stock4U-DailyPicks" -Confirm:$false
```

### Modify the Schedule
```powershell
# Change to run twice daily (9 AM and 6 PM UTC)
$Trigger1 = New-ScheduledTaskTrigger -Daily -At "09:00"
$Trigger2 = New-ScheduledTaskTrigger -Daily -At "18:00"
Set-ScheduledTask -TaskName "Stock4U-DailyPicks" -Trigger @($Trigger1, $Trigger2)
```

### Pause/Resume
```powershell
# Pause the task
Disable-ScheduledTask -TaskName "Stock4U-DailyPicks"

# Resume the task
Enable-ScheduledTask -TaskName "Stock4U-DailyPicks"
```

## Alternative: Manual Cron-style Setup

If you prefer manual setup:

1. Open Task Scheduler (`taskschd.msc`)
2. Click "Create Basic Task"
3. Name: "Stock4U Daily Picks"
4. Trigger: "Daily"
5. Time: "2:00 PM" (or your preferred time)
6. Action: "Start a program"
7. Program: `cmd.exe`
8. Arguments: `/c "D:\Stock4U\scripts\auto_update_daily_picks.bat"`
9. Finish

## Production Notes

- **Reliability**: Task runs as SYSTEM account, so it works even when logged out
- **Error Handling**: Script continues on errors and logs results
- **Resource Usage**: Minimal CPU/memory usage, runs for ~1-2 minutes
- **Network**: Requires internet connection to fetch data and publish to GitHub

## Support

If you encounter issues:
1. Check `logs/daily_picks_auto.log` for error messages
2. Test the batch script manually
3. Verify GitHub token permissions
4. Ensure Python environment is properly set up

The scheduler is designed to be "set and forget" - once configured, it will keep your daily picks fresh automatically!
