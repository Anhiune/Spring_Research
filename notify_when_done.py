"""
Watches bluesky_data/ and sends a Windows toast notification
when all monthly CSVs from START through END are present.

Run this in a separate terminal:
    python notify_when_done.py
It will check every 60 seconds and exit after notifying.
"""

import time
import subprocess
from pathlib import Path
from datetime import date

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "bluesky_data"
START_YEAR, START_MONTH = 2023, 1
END_YEAR,   END_MONTH   = 2026, 2
CHECK_EVERY = 60  # seconds between checks


# ── Month list ────────────────────────────────────────────────────────────────
def all_months(sy, sm, ey, em):
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


TARGET_MONTHS = all_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
TOTAL = len(TARGET_MONTHS)


# ── Windows Toast (uses PowerShell BurntToast or fallback msg.exe) ────────────
def toast(title: str, message: str):
    # Try BurntToast (if installed); fall back to a simple balloon via PowerShell
    ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$notify = New-Object System.Windows.Forms.NotifyIcon
$notify.Icon = [System.Drawing.SystemIcons]::Information
$notify.Visible = $true
$notify.ShowBalloonTip(10000, "{title}", "{message}", [System.Windows.Forms.ToolTipIcon]::Info)
Start-Sleep -Seconds 10
$notify.Dispose()
"""
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_script],
            creationflags=0x00000008  # DETACHED_PROCESS
        )
    except Exception as e:
        print(f"[Notification failed: {e}]")


# ── Main loop ─────────────────────────────────────────────────────────────────
def check_progress():
    present = sorted(
        f.stem.replace("bluesky_", "")
        for f in DATA_DIR.glob("bluesky_*.csv")
    )
    done    = [m for m in TARGET_MONTHS if m in present]
    missing = [m for m in TARGET_MONTHS if m not in present]
    return done, missing


print(f"Watching {DATA_DIR}")
print(f"Target: {TOTAL} months ({TARGET_MONTHS[0]} to {TARGET_MONTHS[-1]})")
print(f"Checking every {CHECK_EVERY}s. Press Ctrl+C to stop.\n")

last_count = -1
while True:
    done, missing = check_progress()
    count = len(done)

    if count != last_count:
        pct = round(100 * count / TOTAL)
        print(f"[{time.strftime('%H:%M:%S')}] {count}/{TOTAL} months done ({pct}%)  |  Next needed: {missing[0] if missing else 'None'}")
        last_count = count

    if not missing:
        msg = f"All {TOTAL} months scraped ({TARGET_MONTHS[0]} to {TARGET_MONTHS[-1]}). Check bluesky_data/."
        print(f"\n*** DONE! {msg} ***")
        toast("Bluesky Scraper Complete", msg)
        break

    time.sleep(CHECK_EVERY)
