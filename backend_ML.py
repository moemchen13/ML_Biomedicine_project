import streamlit as st
from streamlit_timeline import timeline
import json
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# read in files as pandas dataframe
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

def models(ML_task):
    if ML_task == "Classification":
        return ["Logistic Regression", "Decision Tree", "Random Forest"]
    elif ML_task == "Regression":
        return ["Linear Regression", "Regression Tree", "Ridge Regression"]



# select machine lernning model
def select_model(model):
    if model=="Logistic Regression":
        ML_model = LogisticRegression()
    elif model == "Decision Tree":
        ML_model = tree.DecisionTreeClassifier()
    elif model == "Random Forest Classificator":
        ML_model = RandomForestClassifier()
    elif model == "Linear Regression":
        ML_model = LinearRegression()
    elif model == "Regression Tree":
        ML_model = tree.DecisionTreeRegressor()
    elif model == "Ridge Regression":
        ML_model = Ridge()
    elif model == "Random Forest Regressor":
        ML_model = RandomForestRegressor()
    return ML_model
        

["Linear Regression", "Regression Tree", "Ridge Regression","Random Forest Regressor"]


# checks if the chosen model is suited for the desired task
def check_options_for_validity(task,reg_models,clas_models):
    is_valid=True
    submit_message = "Options are well choosen"
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
        submit_message = "Regression target is not a number"
        is_valid=False
    return is_valid, submit_message

def train_model(model, train_test_split, metric):
    update_events("Model training has started",f"{model} is trained and evaluated")
    
    X_train, X_test, y_train, y_test = train_test_split
    folds = 5
    metric = "accuracy"
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9],  # Number of neighbors
        'knn__weights': ['uniform', 'distance'],  # Weight function used in prediction
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
    }


    # Create a GridSearchCV object with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=folds, n_jobs=-1, scoring=metric, refit=True, return_train_score=True)

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    
    # Evaluate the model on the test set
    test_score = grid_search.score(X_test, y_test)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, test_metric


def ML_Pipeline(ML_task,df,features,target,train_test_split, models):
    X = df.loc[features]
    y = df.loc[target]      

    stats = np.zeros(size=(2,len(models)))
    hyperparams = []
    trained_models = []

    for i,model in enumerate(models):
        train_metric,test_metric,model = train_model(model, train_test_split)
        stats[0,i] = train_metric
        stats[1,i] = test_metric
        trained_models.append(model)
    return stats, trained_models,hyperparams

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