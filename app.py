from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uvicorn

# load environment variable
load_dotenv()

# initialize the app
app = FastAPI()
# add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok"} 
# uvicorn app:app --host 0.0.0.0 --port 5001 --reload
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    uvicorn.run(
        "app:app",   # filename:app
        # host="0.0.0.0",
        port=port,
        reload=True
    )