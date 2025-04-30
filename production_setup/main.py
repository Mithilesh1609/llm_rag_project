from app import app

# Import is sufficient - Gunicorn will look for the 'app' variable
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)