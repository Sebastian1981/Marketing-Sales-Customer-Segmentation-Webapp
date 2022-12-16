import streamlit as st
import pandas as pd


def run_cluster_app():
    print('Hello World!')

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)