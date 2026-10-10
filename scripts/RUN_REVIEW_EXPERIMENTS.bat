@echo off
rem The review experiments of 1 Oct 2026 that need no model training:
rem scene bootstrap (W5), CVVDP display check (W4), HDRTV1K standard test (W1), H7 exposure control (W6).
rem Resumable. See scripts\review_experiments_2026_10_01.ps1 for paths and switches.
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\review_experiments_2026_10_01.ps1 %*
pause
