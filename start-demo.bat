@echo off
echo 🚀 Starting Brain Tumor Detection Demo...

REM Start minikube if not running
minikube status >nul 2>&1
if errorlevel 1 (
    echo 📦 Starting minikube...
    minikube start
)

REM Set docker environment (for CMD, use 'call' instead of eval)
FOR /F "tokens=*" %%i IN ('minikube docker-env --shell cmd') DO call %%i

REM Deploy if not already deployed
echo 🔧 Deploying application...
kubectl apply -k k8s/

REM Wait for pods to be ready
echo ⏳ Waiting for pods to be ready...
kubectl wait --for=condition=ready pod -l app=brain-api --timeout=300s

REM Kill any existing port forwards
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8080"') do taskkill /PID %%p /F >nul 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8081"') do taskkill /PID %%p /F >nul 2>&1

REM Start port forwarding in background
echo 🌐 Setting up access...
start "" cmd /c "kubectl port-forward svc/brain-api 8080:80 >nul 2>&1"
start "" cmd /c "kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8081:80 >nul 2>&1"

REM Give it a moment to start
timeout /t 3 >nul

REM Test if everything is working
echo 🧪 Testing API...
curl -s http://localhost:8080/health | findstr "model_loaded.*true" >nul
if %errorlevel%==0 (
    echo ✅ SUCCESS! Demo is ready!
    echo.
    echo 🧠 Brain Tumor Detection API: http://localhost:8080
    echo 🔬 Kubeflow Pipelines UI:     http://localhost:8081
    echo.
    echo 📊 Available endpoints:
    echo   • GET  /health          - Check status
    echo   • POST /predict         - Detect tumors
    echo   • POST /feedback        - Submit feedback
    echo   • POST /trigger-retrain - Start retraining
    echo.
    echo 🎉 Ready for your presentation!
) else (
    echo ❌ Something went wrong. Check the logs:
    echo kubectl logs -l app=brain-api
)
