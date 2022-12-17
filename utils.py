import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


################################################
# Data Preparation
################################################

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

@st.cache
def convert_df(df:pd.DataFrame):
    """convert dataframe to csv format"""
    return df.to_csv().encode('utf-8')



################################################
# Cluster Prediction
################################################

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


################################################
# Cluster Visualization
################################################

def plot_cluster_distribution(df:pd.DataFrame):
    """plot cluster distribution"""
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(
        data=df, 
        x='cluster',
        bins=15)
    plt.title('Cluster Distribution')
    plt.grid('on')
    st.pyplot(fig)

def plot_numeric_features(df:pd.DataFrame, numeric_columns):
    """plot feature distributon for each cluster"""        
    options = st.multiselect(
        'Select Numeric Features',
        numeric_columns)

    for feat in options:

        fig, ax = plt.subplots(
            figsize=(25,5), 
            nrows=1, ncols=df['cluster'].nunique(), 
            sharex=True)
        
        for c in np.sort(df['cluster'].unique()):

            sns.histplot(
                data=df[df['cluster']==c], 
                x=feat,
                bins=15,
                kde=True,
                ax=ax[c])
            ax[c].set_title(feat + ' Distribution in Cluster '+ str(c))
            ax[c].grid()
        st.pyplot(fig)

# visualize categorical features for each cluster
def plot_categorical_features(df, categorical_columns):
    """visualize categorical feature distributions for each cluster"""

    options = st.multiselect(
        'Select Categorical Features',
        categorical_columns)

    for feat in options:

        fig, ax = plt.subplots(
            figsize=(25,5), 
            nrows=1, ncols=df['cluster'].nunique(), 
            sharex=True)
        
        for c in np.sort(df['cluster'].unique()):

            sns.histplot(
                data=df[df['cluster']==c], 
                x=feat,
                ax=ax[c])
            ax[c].set_title(feat + ' Distribution in Cluster '+ str(c))
            ax[c].set_xticks(ax[c].get_xticks(), ax[c].get_xticklabels(), rotation=90)
            ax[c].grid()
        st.pyplot(fig)