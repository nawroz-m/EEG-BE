from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uvicorn
import joblib
import os
from pydantic import BaseModel
from typing import Optional
from utils.utils import predict_sample, validate_input_signal

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

dataset_compositon=[
    {"Set": "A",
     "Label": 1,
     "Anatomical/Clinical Context": "Healthy volunteers, eyes open",
     "Electrode Type": "Scalp (10–20)",
     "State": "Baseline, eyes open"},
    {"Set": "B",
     "Label": 2,
     "Anatomical/Clinical Context": "Healthy volunteers, eyes closed",
     "Electrode Type": "Scalp (10–20)",
     "State": "Baseline, eyes closed"},
    {"Set": "C",
     "Label": 3,
     "Anatomical/Clinical Context": "Epilepsy patients, interictal, outside focus",
     "Electrode Type": "Depth (hippocampal)",
     "State": "Non-seizure, outside focus"},
    {"Set": "D",
     "Label": 4,
     "Anatomical/Clinical Context": "Epilepsy patients, interictal, inside focus",
     "Electrode Type": "Depth (hippocampal)",
     "State": "Non-seizure, within focus"},
    {"Set": "E",
     "Label": 5,
     "Anatomical/Clinical Context": "Epilepsy patients, ictal (seizure)",
     "Electrode Type": "Depth (epileptogenic)",
     "State": "During seizure"},
]
grid_model = joblib.load('./models/search_grid_cv.pkl')
norm_classifier_model = joblib.load('./models/classifier.pkl')


class PredictRequest(BaseModel):
    signal: list[float]
    model: Optional[str] = "norm"


@app.post('/pred')
async def pred(data: PredictRequest):
    """
    Predict a signal using provided model type
        `nomr: normal classifer model`
        'grid: grid search classifier`    
    """
    try:

        if data.model == "grid":
            model = grid_model
        else:
            model = norm_classifier_model
        
        # validate the input signal & scale the signal using z-transform
        sample = validate_input_signal(signal=data.signal, model=model) 
        # predict the signal
        result = predict_sample(model=model['models']['svm_lin'], X=sample) 
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    # return the final result
    return {
        'result': result,
    }

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