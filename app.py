import streamlit as st
from path import Path


#from utils import convert_df
from dataprep_app import run_dataprep_app
from cluster_app import run_cluster_app


def main():
    st.title("Let´s Segment Your Customers!")

    menu = ["About", "Data Preparation", "Segmentation"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "About":
        st.subheader("About")
        st.markdown(Path('About.md').read_text())
    
    elif choice == "Data Preparation":
        st.subheader('Prepare Dataset')
        run_dataprep_app()

    elif choice == "Segmentation":
        st.subheader('Customer Segmentation')
        run_cluster_app()
    
if __name__ == "__main__":
    main()



