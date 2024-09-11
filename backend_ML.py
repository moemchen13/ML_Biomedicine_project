import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import ast

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

def save_model(model, model_name, folder_name='saved_models'):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Define the file path
    file_path = os.path.join(folder_name, model_name + '.pkl')

    # Save the model using joblib
    joblib.dump(model, file_path)
    print(f"Model saved at: {file_path}")
    

# Custom imputation using random values between min and max for each column
def random_impute(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        missing = np.isnan(col)
        col_min, col_max = np.nanmin(col), np.nanmax(col)
        col[missing] = np.random.uniform(col_min, col_max, size=missing.sum())
    return X
        
def clean_data(X, imputation_strategy='most_frequent', scaling_method='minmax'):
    # Separate numerical and non-numerical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns

    # Handle missing values based on the chosen imputation strategy
    if imputation_strategy in ['mean', 'median','most_frequent']:
        imputer = SimpleImputer(strategy=imputation_strategy)
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    elif imputation_strategy == 'random':        
        X_imputed = random_impute(X.copy())  # Apply random imputation before scaling
    else:
        raise ValueError("Invalid imputation strategy. Choose 'mean', 'median', 'most_frequent' or 'random'.")
    
    # Encode non-numerical columns with integer encoding
    X_imputed[categorical_cols] = X_imputed[categorical_cols].apply(LabelEncoder().fit_transform)


    # Apply scaling based on the chosen scaling method
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaling method. Choose 'minmax' or 'standard'.")
    
    # Fit and transform the data
    X_scaled = scaler.fit_transform(X_imputed)
    X_cleaned = pd.DataFrame.from_records(data=X_scaled, columns=X.columns)
    return X_cleaned

def train_model(model, train_test_split, train_config):
    
    X_train, X_test, y_train, y_test = train_test_split
    param_grid = model["params"]
    method = select_model(model["model_type"])
    metric = train_config["metric"]
    folds = train_config["CV"]
    if folds == False:
        folds = None 
        
    if "n_iter" in model.keys():
        # Create a RandomizedSearch object with cross-validation    
        searchCV = RandomizedSearchCV(method, param_grid, n_iter=model["n_iter"], cv=folds, n_jobs=-1, scoring=metric, random_state=train_config["seed"], return_train_score=True)
    else:
        for p in param_grid:
            if not isinstance(param_grid[p], list):
                param_grid[p] = [param_grid[p]]
        searchCV = GridSearchCV(method, param_grid, cv=folds, n_jobs=-1, scoring=metric, return_train_score=True)
    
    # Fit the model on the training data
    searchCV.fit(X_train, y_train)
    
    # Test model on test dataset
    test_score =  searchCV.score(X_test, y_test)
    
    return searchCV.best_estimator_, searchCV.best_params_, searchCV.best_score_, searchCV.cv_results_, test_score


def ML_Pipeline(data, json_config):
    config = json.loads(json_config)
    
    models = {**config["Runs"], **config["Models"]}
    train_config = config["Training"]

    target = train_config["target"]
    
    df = clean_data(data)
    X = df.drop(target, axis=1)
    y = df[target]
    
    data_split = train_test_split(X, y, test_size = train_config['train_test_split'], shuffle=True, random_state=train_config['seed'])
    
    stats = []
    hyperparams = []
    trained_models = []     

    for model in enumerate(models):
        trained_model, best_params, val_score, cv_results, test_score= train_model(models[model[1]], data_split, train_config)
        
        s = {"validation_score": val_score, "cv_summary": pd.DataFrame(cv_results)}
        stats.append(s)
        trained_models.append(trained_model)
        hyperparams.append(best_params)
    return stats, trained_models, hyperparams

def get_best_model(stats, trained_models):
    index = np.argmin(stats[1,:])
    best_value = np.min(stats[1,:])
    model= trained_models[index]
    return model,best_value,index