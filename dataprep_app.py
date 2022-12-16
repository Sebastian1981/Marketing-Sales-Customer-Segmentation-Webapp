import streamlit as st
import pandas as pd
import pickle

from feature_engine.encoding import RareLabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# load model artifacts
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

with open('./artifacts/model.pickle', 'rb') as filename: # trained random forrest classifier
    model = pickle.load(filename)



def run_dataprep_app():

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Relabel taget colum since there is no target in unsupervised learning
        df.rename(columns={"Target": "Income"}, inplace=True)
        # Correct the Income column typo
        df['Income'] = df['Income'].apply(lambda x: x.replace('.', ''))

        # get numeric and categorical columns
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_columns = df.select_dtypes(include=numerics).columns.to_list()
        categorical_columns = df.select_dtypes(exclude=numerics).columns.to_list()

        # impute mising numeric features
        df_numeric = pd.DataFrame(
            numeric_imputer.transform(df[numeric_columns]), 
            columns=numeric_columns, 
            index=df.index)
        
        # impute mising categorical features
        df_categorical = pd.DataFrame(
            categorical_imputer.transform(df[categorical_columns]), 
            columns=categorical_columns, 
            index=df.index)
        
        # concate numeric and categorical features
        df = pd.concat([df_numeric, df_categorical], axis=1)

        ## remove rare labels
        #df[categorical_columns] = rare_encoder.transform(df[categorical_columns])

        

        st.write(df)

