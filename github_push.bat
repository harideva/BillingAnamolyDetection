@echo off
echo ============================================
echo  GitHub Push Script   
REM TOKEN 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
REM https://docs.github.com/BillingAnamolyDetection
echo ============================================
echo.

REM Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Git is not installed or not in PATH.
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo ✅ Git found!

REM Step 1: Initialize git
echo.
echo 📂 Initializing git repository...
git init

REM Step 2: Add all files
echo.
echo 📤 Adding all files...
git add .

REM Step 3: Commit
echo.
echo 📝 Committing files...
git commit -m "MVP: AI-Powered Billing Anomaly Detection System"

REM Step 4: Create main branch
echo.
echo 🌿 Creating main branch...
git branch -M main

REM Step 5: Ask for GitHub repo URL
echo.
echo ============================================
echo  Enter your GitHub repository URL
echo ============================================
echo Example: https://github.com/harideva/billing-anomaly-detection.git
echo.
set /p REPO_URL="GitHub Repo URL: "

if "%REPO_URL%"=="" (
    echo ❌ No URL provided. Skipping push.
    pause
    exit /b 1
)

REM Step 6: Add remote
echo.
echo 🔗 Adding remote origin...
git remote add origin %REPO_URL%

REM Step 7: Push
echo.
echo 🚀 Pushing to GitHub...
git push -u origin main

echo.
echo ============================================
echo ✅ GitHub Push Complete!
echo ============================================
pause