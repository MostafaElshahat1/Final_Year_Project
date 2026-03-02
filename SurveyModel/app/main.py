from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .inference import ModelInference
from .utils import preprocess_input

app = FastAPI(title="Mental Health Risk API")
engine = ModelInference()

# Define the expected input schema from the backend
class SurveyInput(BaseModel):
    # Everything is now a string to please the backend
    Age: str
    Parents_Home: str
    Parents_Dead: str
    Fathers_Education: str
    Mothers_Education: str
    Co_Curricular: str
    Percieved_Academic_Abilities: str
    Gender: str
    Form: str
    Religion: str
    Boarding_day: str
    School_type: str
    School_Demographics: str
    School_County: str
    Sports: str

    class Config:
        extra = "allow" # This allows BL_1, ACES_1 etc. to come in as strings too

@app.post("/predict")
async def predict_student_risk(data: SurveyInput):
    try:
        # 1. Convert Pydantic object to dict
        raw_dict = data.dict()
        
        # 2. Apply Preprocessing (Calculating Trauma_x_Bullying, etc.)
        processed_df = preprocess_input(raw_dict)
        
        # 3. Run Inference
        prediction, probability = engine.predict(processed_df)
        
        # 4. Return Output to Backend
        return {
            "status": "success",
            "prediction": {
                "risk_id": int(prediction),
                "risk_label": "Support Recommended" if prediction == 1 else "Low Concern",
                "probability_score": f"{round(probability * 100, 2)}%",
                "intervention_needed": bool(prediction)
            },
            "recommendation": "Please arrange a meeting with the school counselor." if prediction == 1 else "Maintain regular check-ins."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))