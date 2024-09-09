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