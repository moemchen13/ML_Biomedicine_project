import streamlit as st 
import backend_ML as be

st.set_page_config(page_title="ML Tabular data site", layout="wide")
st.title("Welcome to the your online Machine learning site")

data_uploader,conf_uploader = st.columns([0.9,0.1])
with data_uploader:
    f = st.file_uploader("Please upload the dataset you want to analyse with Machine learning models")
with conf_uploader:
    conf_uploader.wr
    conf_button = st.button(label="Upload Config JSON")
    #conf = st.file_uploader(label="Config",label_visibility="collapsed")
if f:
    df = be.return_df(f)
    st.success("File uploaded")
    df = st.data_editor(df)




popover =  st.popover("Select task")

reg,clas = st.columns(2)
with reg:
        st.markdown("Classification")
with clas:
        red = popover.toggle("Use Regression")

if red:
    st.write(":red[We now use regresseion]")
else:
    st.write("We use classification")

