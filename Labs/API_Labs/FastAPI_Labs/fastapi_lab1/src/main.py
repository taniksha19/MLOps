from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data, predict_rf


app = FastAPI()

iris_classes = {0: "setosa", 1: "versicolor", 2: "virginica"}

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    response:str

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]

        prediction = predict_data(features)
        class_name = iris_classes[int(prediction[0])]
        return IrisResponse(response =class_name)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_rf", response_model=IrisResponse)
async def predict_iris_rf(iris_features: IrisData):
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]

        prediction = predict_rf(features)
        class_name = iris_classes[int(prediction[0])]
        return IrisResponse(response =class_name)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    
