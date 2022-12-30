from fastapi import FastAPI
import uvicorn
import numpy as np
import pandas as pd
import pickle

##############################################
# load model artifacts
##############################################

with open('./artifacts/numeric_imputer.pickle', 'rb') as filename: # trained model to impute missing numeric data
    numeric_imputer = pickle.load(filename)

with open('./artifacts/categorical_imputer.pickle', 'rb') as filename: # trained model to impute missing categorical data
    categorical_imputer = pickle.load(filename) 

with open('./artifacts/rare_encoder.pickle', 'rb') as filename: # trained model to encode rare labels
    rare_encoder = pickle.load(filename)

with open('./artifacts/capper.pickle', 'rb') as filename: # trained model to cap outliers
    capper = pickle.load(filename)   

with open('./artifacts/enc.pickle', 'rb') as filename: # trained one hot encoder
    enc = pickle.load(filename)

with open('./artifacts/model.pickle', 'rb') as filename: # trained random forest classifier
    model = pickle.load(filename)





app = FastAPI()

@app.get("/")
async def root():
    return {"message": "This is Your Customer Segmentation API!"}

# Define bmi-test function
@app.post('/predict_bmi')
def predict_bmi(weight:float, height:float):
    """ For testing purposes, calculate the BMI by dividing weight in kilograms by height in meters squared"""
    bmi = [weight/height**2]
    return {'predicted bmi': list(bmi)}


# Define customer segmentation prediction function
@app.post('/predict')
def predict(
    credit_score: float,
    country: str,
    gender: str,
    age: int,
    tenure: int,
    balance: float,
    products_number: int,
    credit_card: str,
    active_member: str,
    estimated_salary: float):
    
    """ Predict customer segments given """
    
    # transform data to dataframe
    data = [
        {'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary
        }
    ]
    df_row = pd.DataFrame(data)

    # get numeric and categorical columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = df_row.select_dtypes(include=numerics).columns.to_list()
    categorical_columns = df_row.select_dtypes(exclude=numerics).columns.to_list()

    # impute mising numeric features
    df_numeric = pd.DataFrame(
        numeric_imputer.transform(df_row[numeric_columns]), 
        columns=numeric_columns, 
        index=df_row.index)

    # impute mising categorical features
    df_categorical = pd.DataFrame(
        categorical_imputer.transform(df_row[categorical_columns]), 
        columns=categorical_columns, 
        index=df_row.index)

    # concate numeric and categorical features
    df_row = pd.concat([df_numeric, df_categorical], axis=1)

    # remove rare labels
    df_row[categorical_columns] = rare_encoder.transform(df_row[categorical_columns])

    # remove outliers
    df_row[numeric_columns] = capper.transform(df_row[numeric_columns])

    # one hot encoding categorical features
    df_cat_hotenc = pd.DataFrame(
        enc.transform(df_row[categorical_columns]), 
        columns=enc.get_feature_names_out(),
        index=df_row.index) 

    # concate numeric and hot-encoded categorical features
    df_hotenc = pd.concat([df_row[numeric_columns], df_cat_hotenc], axis=1)

    # predict cluster
    labels = model.predict(df_hotenc)

    return {'predicted cluster label ': labels.tolist()}             







if __name__ == '__main__':
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000)  