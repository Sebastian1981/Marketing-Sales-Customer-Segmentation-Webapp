import streamlit as st
import pandas as pd
import pickle

#@st.cache # cache the conversion to prevent computation on every rerun
def prepare_data(df:pd.DataFrame, numeric_columns, categorical_columns)->pd.DataFrame:
    """ prepare dataset for segmentation"""
    
    ############################################
    # Load Model Artifacts
    ############################################
    with open('./artifacts/numeric_imputer.pickle', 'rb') as filename: # trained model to impute missing numeric data
        numeric_imputer = pickle.load(filename)

    with open('./artifacts/categorical_imputer.pickle', 'rb') as filename: # trained model to impute missing categorical data
        categorical_imputer = pickle.load(filename) 

    with open('./artifacts/rare_encoder.pickle', 'rb') as filename: # trained model to encode rare labels
        rare_encoder = pickle.load(filename)

    with open('./artifacts/capper.pickle', 'rb') as filename: # trained model to cap outliers
        capper = pickle.load(filename)   
    

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

    # remove rare labels
    df[categorical_columns] = rare_encoder.transform(df[categorical_columns])

    # remove outliers
    df[numeric_columns] = capper.transform(df[numeric_columns])

    return df


#@st.cache 
def predict_cluster(df:pd.DataFrame, numeric_columns, categorical_columns)->pd.DataFrame:
    """predict labels for preprocessed data frame"""

    # load one-hot encoder
    with open('./artifacts/enc.pickle', 'rb') as filename: # trained one hot encoder
        enc = pickle.load(filename)

    # load trained cluster model
    with open('./artifacts/model.pickle', 'rb') as filename: # trained random forest classifier
        model = pickle.load(filename)

    # one hot encoding categorical features
    df_cat_hotenc = pd.DataFrame(
        enc.transform(df[categorical_columns]), 
        columns=enc.get_feature_names_out(),
        index=df.index)

    # concate numeric and hot-encoded categorical features
    df_hotenc = pd.concat([df[numeric_columns], df_cat_hotenc], axis=1)
        
    # predict cluster
    labels = model.predict(df_hotenc)

    # add cluster label to df
    df['cluster'] = labels

    return df


@st.cache
def convert_df(df:pd.DataFrame):
    """convert dataframe to csv format"""
    return df.to_csv().encode('utf-8')