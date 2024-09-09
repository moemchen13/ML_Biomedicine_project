import streamlit as st 
import backend_ML as be
import time

st.set_page_config(page_title="ML Tabular data site", layout="wide")
st.title("Welcome to the your online Machine learning side")


data_uploader, conf_uploader = st.columns([0.8,0.2],vertical_alignment="top")
with data_uploader:
    f = st.file_uploader("Please upload the dataset you want to analyse with Machine learning models")
    if f:
        df = be.return_df(f)
        st.success("File uploaded")
        df = st.data_editor(df)

with conf_uploader:
    st.write("")
    st.write("")
    st.write("")
    with st.popover("Upload \n config"):
        conf = st.file_uploader("config json")
        if conf:
            config = be.return_df(conf)
            st.success("Uploaded config file")
            

#summary stats

st.divider()
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


if f:
    targ,seed,train_test = st.columns(3)
    with targ:
        target = st.selectbox("Select target variable",df.columns,index=len(df.columns)-1)
        possible_features = df.columns.drop(target)
    with seed:
        seed = st.number_input("Set seed",value = 42,step=1,min_value=1,max_value=None)
    with train_test:
        train_test_split = st.number_input("relative size of test set",value = 0.2,step=0.01,min_value=0.01,max_value=0.99)
    
    imput,scale = st.columns(2)
    with imput:
        data_imputation = st.selectbox("Replace missing data with",options=["mean","median","random"],index=0)
    with scale: 
        data_scaling = st.selectbox("Scaling",options=["No scaling","min_max scaling","standardising","0-1-standardizing"],index=0)


models = be.models(task_is_regression)
    st.tabs(len(classification_models))

#st.toast()