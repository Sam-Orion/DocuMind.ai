import streamlit as st
import requests

st.set_page_config(page_title="DocuMind AI", layout="wide")

st.title("DocuMind AI")
st.markdown("Intelligent Document Processing System")

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg', 'pdf'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("Processing...")
    # TODO: Call backend API
