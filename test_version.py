import streamlit as st
from streamlit_timeline import timeline
import json
from datetime import datetime, timedelta
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def return_df(file):
    name = file.name
    extension = name.split(".")[-1]
    if extension=="csv":
        df = pd.read_csv(name)
    elif extension=="tsv":
        df = pd.read_csv(name,sep="\t")
    elif extension=="csv":
        df = pd.read_json(name)
    elif extension=="xlsx":
        df = pd.read_excel(name)
    elif extension=="xml":
        df = pd.read_xml(name)
    return df

def start_timeline():

    base_time = datetime.now()
    if "events_json" not in st.session_state:
        st.session_state.events_json = {"events":[]}


def update_events(header,text=""):
    event_time = datetime.now()
    event = {"start_date": {
                    "year": event_time.year,
                    "month": event_time.month,
                    "day": event_time.day,
                    "hour": event_time.hour,
                    "minute": event_time.minute,
                    "second": event_time.second
        },
        "text": {
            "headline": header,
            "text": text
        }}
    st.session_state.events_json["events"].append(event)

def check_options_for_validity(task,reg_models,clas_models):
    is_valid=True
    submit_message = "OPtions are well choosen"
    if task=="Regression​" and reg_models==[]:
        is_valid = False
        submit_message = "Select Regression model(s)"
    if task=="Classification" and clas_models==[]:
        is_valid = False
        submit_message = "Select Classification model(s)"

    return is_valid,submit_message

def check_data_submission_for_task(ML_task,df,feature_selection,target):
    is_valid = True
    submit_message = "Submission was successfull!"
    is_numeric = pd.api.types.is_numeric_dtype(df.loc[target])
    if ML_Task=="Regression​" and not is_numeric:
        submit_message = "Regression target is no number"
        is_valid=False
    return is_valid, submit_message


def select_model(model):
    if model=="Logistic Regression":
        metric = accuracy_score
        ML_model = LogisticRegression()
    elif model == "Decision Tree":
        metric = accuracy_score
        ML_model = tree.DecisionTreeClassifier()
    elif model == "Random Forest":
        metric = accuracy_score
        ML_model = RandomForestClassifier()
    elif model == "Linear Regression":
        metric = mean_squared_error
        ML_model = LinearRegression()
    elif model == "Regression Tree":
        metric = mean_squared_error
        ML_model = tree.DecisionTreeRegressor()
    elif model == "Ridge Regression":
        metric = mean_squared_error
        ML_model = Ridge()

    return model,metric

def train_model(model,X_train,X_test,y_train,y_test):
    update_events("Model training has started",f"{model} is trained and evaluated")
    ML_model,metric = select_model(model)
    ML_model.fit(X_train,y_train)
    y_pred_train = ML_model.predict(X_train)
    y_pred_test = ML_model.predict(X_test)
    train_metric = metric(y_pred_train,y_train)
    test_metric = metric(y_pred_test,y_test)
    
    return model, train_metric, test_metric

def start_ML_Pipeline(ML_task,df,feature,target,reg_models,clas_models):
    X = df.loc[feature]
    y = df.loc[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=True,test_split=0.2)

    models = reg_models
    if ML_task == "Classification":
        models = clas_models
    
    stats = np.zeros(size=(2,len(models)))
    hyperparams = []
    models = []

    for i,model in enumerate(models):
        train_metric,test_metric,model = train_model(model,X_train,X_test,y_train,y_test)
        stats[0,i] = train_metric
        stats[1,i] = test_metric
        models.append(model)
    return stats, models,hyperparams

def get_best_model(ML_task,metric,models,reg_models,clas_models):
    if ML_task == "Classification":
        index = np.argmax(metric[1,:])
        best_value = np.max(metric[1,:])
        model= clas_models[index]
    else:
        index = np.argmin(metric[1,:])
        best_value = np.min(metric[1,:])
        model= clas_models[index]

    return model,best_value,index

#### Start side ####
st.set_page_config(page_title="M learning experiment", layout="wide")

#create timeline
start_timeline()
if st.session_state.events_json["events"]==[]:
    update_events("Website visit","User is on website")


#Task selection
st.subheader('Task Choice')
ML_task = st.radio(
    "ML task that needs solving",
    ["Regression​", "Classification"],
    captions=[
        "Predict values 〽️",
        "Predict classes 🐔"
    ],
)

col1, col2= st.columns(2, gap="small")


with col1:
    st.markdown("<h4>Regression models:</h4>",unsafe_allow_html=True)
    regression_models = ["Linear Regression","Regression Tree","Ridge Regression"]
    reg_is_disabled = ML_task =="Classification"
    reg_models = st.multiselect("Select one or more of the following models:", regression_models,disabled=reg_is_disabled)

with col2:
    st.markdown("<h4>Classification models:</h4>",unsafe_allow_html=True)
    classification_models = ["Logistic Regression","Decision Tree","Random Forest"]
    clas_is_disabled = ML_task == "Regression​"
    clas_models = st.multiselect("Select one or more of the following models:", classification_models,disabled=clas_is_disabled)



#create Upload
f = st.file_uploader("Please upload a dataset of your choosing", type=["csv","tsv","xlsx","xml","json"])
if f:
    df = return_df(f)
    st.success("File uploaded")
    update_events("File upload","User uploaded a file")
    st.dataframe(df)



    tab1,tab2 = st.tabs(["ML model","EDA"])
    #ML models
    with tab1:
        col_target,col_fea = st.columns(2)

        with col_target:
            target = st.multiselect("Please select the target",df.columns,index=len(df.columns)-1)
            
        with col_fea:
            features = st.multiselect("Select one or more of the following models:", df.columns)
    #EDA
    with tab2:
        #pr = ProfileReport(df)
        #st_profile_report(pr)
        st.markdown("CCC")
        pass

    print(target)
    print(features)

if f:
    if st.button("Submit", key="submit_button", help=""):
        is_valid_option,option_message = check_options_for_validity(ML_task,reg_models,clas_models)
        st.write(option_message)
        is_valid_submission,submit_message = check_data_submission_for_task(ML_task,df,features,target)
        st.write(submit_message)
        if not is_valid_option:
            update_events("Failed Submission", "Options are not sufficient choosen.")
        elif not is_valid_submission:
            update_events("Failed Submission", "Data selected is unfit.")
        else:
            update_events("Started ML Pipeline", "Successfull submission starting to train models")
            stats, models,hyperparams = start_ML_Pipeline(ML_task,df,feature_selection,target_selection,reg_models,clas_models) 
            model,best_value,index = get_best_model(ML_task,metric,reg_models,clas_models) 
            if ML_task=="Classification":
                st.markdown(f"<p>The best model has an Accuracy of {best_value}</p>",unsafe_allow_html=True)
                st.markdown(f"<p>It is {clas_models[index]}</p>",unsafe_allow_html=True)
            else:
                st.markdown(f"<p>The best model has an MSE of {best_value}</p>",unsafe_allow_html=True)
                st.markdown(f"<p>It is {reg_models[index]}</p>",unsafe_allow_html=True)
            st.markdown(f"<p>You can now download it</p>",unsafe_allow_html=True)
        update_events("Evaluation finished", "Models were trained and evaluated")


        #add download and add metric visualisation
#place timeline at the bottom
timeline(st.session_state.events_json,height=300)