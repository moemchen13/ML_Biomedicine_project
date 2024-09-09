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
from sklearn.ensemble import RandomForestClassifier


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

def train_model(model,X_train,X_test,y_train,y_test, ):
    update_events("Model training has started",f"{model} is trained and evaluated")
    
    # Create a pipeline with classifier
    pipeline = Pipeline([
        ('model', model)  # KNN classifier
    ])

    # Define the parameter grid for grid search
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9],  # Number of neighbors
        'knn__weights': ['uniform', 'distance'],  # Weight function used in prediction
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
    }

    # Create a GridSearchCV object with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    ML_model,metric = select_model(model)
    ML_model.fit(X_train,y_train)
    y_pred_train = ML_model.predict(X_train)
    y_pred_test = ML_model.predict(X_test)
    train_metric = metric(y_pred_train,y_train)
    test_metric = metric(y_pred_test,y_test)
    
    return model, train_metric, test_metric


def ML_Pipeline(ML_task,df,features,target,test_split,seed):
    X = df.loc[features]
    y = df.loc[target]
    

    X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=True,test_split=test_split, random_state=seed)
    
    # imputation and scaling 
    
    
    if ML_task == "Classification":
        models = models("Classification")
    else: 
        models = models("Regression")
    
    stats = np.zeros(size=(2,len(models)))
    hyperparams = []
    trained_models = []

    for i,model in enumerate(models):
        train_metric,test_metric,model = train_model(model,X_train,X_test,y_train,y_test)
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