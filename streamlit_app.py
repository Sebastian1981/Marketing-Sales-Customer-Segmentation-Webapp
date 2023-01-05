import streamlit as st
from path import Path


#from utils import convert_df
from segmentation_app import run_segment_app

def main():
    st.title("Your Marketing App for Customer Segmentation!")

    menu = ["About", "Segmentation"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "About":
        st.subheader("About")
        st.markdown(Path('About.md').read_text())
    
    elif choice == "Segmentation":
        st.subheader('Segmenting Your Customers')
        run_segment_app()
    
if __name__ == "__main__":
    main()



