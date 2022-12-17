import streamlit as st
import numpy as np
import pandas as pd
from utils import prepare_data, predict_cluster, convert_df, plot_cluster_distribution, plot_numeric_features, plot_categorical_features
import matplotlib.pyplot as plt
import seaborn as sns

def run_segment_app():

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        ############################################
        # Read Raw CSV File
        ############################################
        df = pd.read_csv(uploaded_file)
        # Relabel taget colum since there is no target in unsupervised learning
        df.rename(columns={"Target": "Income"}, inplace=True)
        # Correct the Income column typo
        df['Income'] = df['Income'].apply(lambda x: x.replace('.', ''))

        ############################################
        # Get Column Types
        ############################################
        # get numeric and categorical columns
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_columns = df.select_dtypes(include=numerics).columns.to_list()
        categorical_columns = df.select_dtypes(exclude=numerics).columns.to_list()
        
        ############################################
        # Predict Cluster
        ############################################
        # prepare data and predict clusters
        df = prepare_data(df, numeric_columns, categorical_columns)

        # prepare data and predict clusters
        df = predict_cluster(df, numeric_columns, categorical_columns)

        # show labeled data
        st.write(df)

        # download labeled dataset
        st.download_button(
            label="Download Data as CSV",
            data=convert_df(df),
            file_name='df_clustered.csv',
            mime='text/csv',
        )

        ############################################
        # Visualize Cluster
        ############################################
        # visualize cluster distribution
        plot_cluster_distribution(df)

        # visualize numeric features for each cluster
        plot_numeric_features(df, numeric_columns)

        # visualize categorical features for each cluster
        plot_categorical_features(df, categorical_columns)