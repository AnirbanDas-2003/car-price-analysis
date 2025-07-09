@echo off
echo Starting Car Price Prediction System...
echo.
echo 1. Checking if models are trained...
if not exist "car_price_models.pkl" (
    echo Models not found. Training models first...
    python train_model.py
    echo.
)

echo 2. Starting Streamlit app...
echo.
echo The app will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.
streamlit run streamlit_app.py
