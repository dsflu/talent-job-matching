"""
This script starts the api server. Check http://0.0.0.0:8000/docs for swagger UI.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=True)