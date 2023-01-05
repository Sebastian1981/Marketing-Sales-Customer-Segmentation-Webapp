import streamlit as st
import pandas as pd
from utils import prepare_data, predict_cluster, convert_df, plot_cluster_distribution, plot_numeric_features, plot_categorical_features, pie_plot_cluster_distribution


def run_segment_app():

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        ############################################
        # Read Raw CSV File
        ############################################
        df = pd.read_csv(uploaded_file, index_col='customer_id')
        # drop churn column for segmentation
        df.drop('churn', axis=1, inplace=True)
        # change type of categorical columns "credit_card" and "active_member"
        df['credit_card'] = df['credit_card'].apply(lambda x: 'yes' if x == 1 else 'no')
        df['active_member'] = df['active_member'].apply(lambda x: 'yes' if x == 1 else 'no')

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
        st.subheader('Plot Cluster Distributions')
        pie_plot_cluster_distribution(df)
        plot_cluster_distribution(df)

        # visualize numeric features for each cluster
        st.subheader('Plot Univariate Distributions')
        plot_numeric_features(df, numeric_columns)

        # visualize categorical features for each cluster
        plot_categorical_features(df, categorical_columns)

        
         
        
        