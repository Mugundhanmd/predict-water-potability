@echo off
echo Setting up the application...

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt
pip install xgboost

echo.
echo Starting the Flask application...
echo Please leave this window open. To stop the server, press Ctrl+C.
echo Open your browser and go to http://localhost:8080/
echo.

python app.py
pause
