import streamlit as st 
import backend_ML as be

st.set_page_config(page_title="ML Tabular data site", layout="wide")
st.title("Welcome to the your online Machine learning side")


data_uploader, conf_uploader = st.columns([0.8,0.2],vertical_alignment="center")
with data_uploader:
    f = st.file_uploader("Please upload the dataset you want to analyse with Machine learning models")
    if f:
        df = be.return_df(f)
        st.success("File uploaded")
        df = st.data_editor(df)

with conf_uploader:
    st.write("")
    with st.popover("Upload \n config"):
        conf = st.file_uploader("config json")
        if conf:
            config = be.return_df(conf)
            st.success("Uploaded config file")
            



st.markdown("Select task")
clas,tog,reg = st.columns(3)

with tog:
        task_is_regression = st.toggle("")
with clas:
    if task_is_regression:
        st.markdown("Classification")
    else:
        st.markdown(":blue[Classification]")
with reg:
    if task_is_regression:
        st.markdown(":red[Regression]")
    else:
        st.markdown("Regression")


