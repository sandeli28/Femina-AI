import uvicorn
import os
import sys

# Ensure the current directory is in the path
sys.path.append(os.getcwd())

if __name__ == "__main__":
    print("Starting System...")
    print("Importing app.main...")
    from app.main import app
    print("Import successful. Starting uvicorn...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
