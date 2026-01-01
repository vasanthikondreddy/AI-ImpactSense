from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


df = pd.read_csv("earthquake_alert_balanced_dataset.csv")

model = joblib.load("model.pkl")


class InputData(BaseModel):
    magnitude: float
    depth: float
    cdi: float
    mmi: float
    sig: float

@app.post("/predict")
async def predict(data: InputData):
    input_vec = [[data.magnitude, data.depth, data.cdi, data.mmi, data.sig]]
    prediction = model.predict(input_vec)[0]
    return {"alert": prediction}

@app.get("/dataset")
async def get_dataset_preview(limit: int = 50):
    
    preview = df.head(limit).to_dict(orient="records")
    return {"rows": preview}
