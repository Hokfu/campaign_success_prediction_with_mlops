import pickle
from fastapi import FastAPI
from pydantic import BaseModel

with open('./models/rf_clf.bin', 'rb') as f:
    rf_clf = pickle.load(f)

with open('./models/dv.bin', 'rb') as f:
    dv = pickle.load(f)

app = FastAPI()

class CampaignData(BaseModel):
    goalamount: float
    raisedamount: float
    durationdays: int
    numbackers: int
    category: str
    launchmonth: str
    country: str
    currency: str
    ownerexperience: int
    videoincluded: str
    socialmediapresence: int
    numupdates: int

@app.post("/predict")
def predict(campaign_data: CampaignData):
    data = campaign_data.dict()
    X = dv.transform([data])
    y_pred = rf_clf.predict(X)
    result = {
        "success": bool(y_pred),
    }
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
