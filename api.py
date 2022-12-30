from fastapi import FastAPI
import uvicorn
import numpy as np

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


# Define predict function
@app.post('/predict_bmi')


def predict_bmi(weight:float, height:float):
    """ You can calculate your BMI by dividing your weight in kilograms by your height in meters squared"""

    bmi = weight
    return {'predicted bmi': list(bmi)}         

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)  