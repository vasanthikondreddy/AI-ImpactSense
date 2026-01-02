from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib   # for loading the saved model

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset for preview
df = pd.read_csv("earthquake_alert_balanced_dataset.csv")

# Load pre-trained Decision Tree model
model = joblib.load("model.pkl")

# Input schema
class InputData(BaseModel):
    magnitude: float
    depth: float
    cdi: float
    mmi: float
    sig: float

@app.post("/predict")
def predict(data: InputData):
    input_vec = [[data.magnitude, data.depth, data.cdi, data.mmi, data.sig]]
    prediction = model.predict(input_vec)[0]
    return {"alert": prediction}

@app.get("/dataset")
def get_dataset_preview():
    return {"rows": df.to_dict(orient="records")}
