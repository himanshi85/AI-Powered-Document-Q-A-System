import streamlit as st
from transformers import pipeline

pipe=pipeline('sentiment-analysis')
text=st.text_area('Enter Text Here!!!')

if text:
    out=pipe(text)
    st.json(out)