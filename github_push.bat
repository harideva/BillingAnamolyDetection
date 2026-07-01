@echo off
echo ============================================
echo  GitHub Push Script
echo ============================================
echo.

REM Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo Git is not installed or not in PATH.
    pause
    exit /b 1
)

echo Git found!

REM Initialize if needed
if not exist .git (
    echo Initializing git repository...
    git init
)

echo Adding all files...
git add .

echo Committing files...
git commit -m "MVP: AI-Powered Billing Anomaly Detection System"

echo Creating main branch...
git branch -M main

echo Setting remote...
git remote add origin https://github.com/harideva/BillingAnamolyDetection.git 2>nul

echo Pushing to GitHub...
git push -u origin main

echo.
echo Complete!
pause