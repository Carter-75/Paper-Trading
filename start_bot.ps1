$ErrorActionPreference = "Continue"
Set-Location 'C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading'
while ($true) {
  python -u runner.py -t2 -m1000 2>&1 | Tee-Object -FilePath '.\bot.log' -Append
  Start-Sleep -Seconds 10
}
