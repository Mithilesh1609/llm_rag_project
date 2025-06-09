# from app import app

# # Import is sufficient - Gunicorn will look for the 'app' variable
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import os
from mangum import Mangum
from app import app

# Mangum is a wrapper that allows FastAPI to work with AWS Lambda
handler = Mangum(app)

# For local development (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))