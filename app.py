from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uvicorn
import joblib
from io import BytesIO
from pydantic import BaseModel
from typing import Optional
from utils.utils import predict_sample, validate_input_signal
import pandas as pd

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
async def pred(
    # request: Optional[PredictRequest] = None,
    file: Optional[UploadFile] = File(None),
    model_name:str= Form('norm')):
    """
    Predict a signal using provided model type
        `nomr: normal classifer model`
        'grid: grid search classifier`    
    """
    try:

        if model_name == "grid":
            model = grid_model
        else:
            model = norm_classifier_model
        
        print(file.filename)
        if file is not None:
            content = await file.read()
            signal_df = pd.read_excel(BytesIO(content)) 
            # drop the empty rows and columns
            signal_df = signal_df.dropna(axis=1, how='all')
        else:
            raise ValueError("No file uploaded")
        
        # validate the input signal & scale the signal using z-transform
        sample = validate_input_signal(signal=signal_df, model=model) 
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