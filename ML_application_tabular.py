import streamlit as st 
import backend_ML as be
import time
import json
import base64

st.set_page_config(page_title="ML Tabular data site", layout="wide")
st.title("Welcome to the your online Machine learning side")

if "runs" not in st.session_state:
    st.session_state["runs"] = []
if "models" not in st.session_state:
    st.session_state["models"] = []
if "task" not in st.session_state:
    st.session_state["task"] = True
if "use_cv" not in st.session_state:
    st.session_state["use_cv"] = False
if "k_fold" not in st.session_state:
    st.session_state["k_fold"] = 5
if "metric" not in st.session_state:
    st.session_state["metric"] = None
if "json_valid" not in st.session_state:
    st.session_state["json_valid"] = False
if "json_checked" not in st.session_state:
    st.session_state["json_checked"] = False
if "target" not in st.session_state:
    st.session_state["Target"] = ""
if "target" not in st.session_state:
    st.session_state["Seed"] = None
if "train_test_split" not in st.session_state:
    st.session_state["train_test_split"] = 0.2


metric = None
use_cv = False
k_fold = 5

data_uploader, conf_uploader = st.columns([0.6,0.2],vertical_alignment="top")
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
            process_JSON()
            

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
st.session_state.task = task_is_regression

if f:
    targ,seed,train_test = st.columns(3)
    with targ:
        target = st.selectbox("Select target variable",df.columns,index=len(df.columns)-1)
        #possible_features = df.columns.drop(target)
        st.session_state.target = target
    with seed:
        seed = st.number_input("Set seed",value = 42,step=1,min_value=1,max_value=None)
        st.session_state.seed = seed
    with train_test:
        train_test_split = st.number_input("relative size of test set",value = 0.2,step=0.01,min_value=0.01,max_value=0.99)
        st.session_state.train_test_split = train_test_split
    
    imput,scale = st.columns(2)
    with imput:
        data_imputation = st.selectbox("Replace missing data with",options=["mean","median","random"],index=0)
    with scale: 
        data_scaling = st.selectbox("Scaling",options=["No scaling","min_max scaling","standardising","0-1-standardizing"],index=0)


# Modelselection methods
param_grid_decision_tree_regression = {
    'max_depth': [int, [1,20,1]],  # Controls the maximum depth of the tree
    'min_samples_split': [int,[2, 25,1]],  # Minimum samples required to split an internal node
    'min_samples_leaf': [int,[1, 50,1]],  # Minimum samples required at each leaf node
    'max_features': [str,['auto', 'sqrt', 'log2']],  # Number of features to consider for the best split
    'criterion': [str,['squared_error', 'friedman_mse', 'absolute_error', 'poisson']],  # Split quality function
    'splitter': [str,['best', 'random']],  # Strategy to split at each node
}


param_grid_ridge = {
    'alpha': [int, [0.01, 100.0,0.01]],
    'solver': [str,['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']],
    'max_iter': [int, [1000, 10000,100]],
}

param_grid_logistic = {
    'penalty': [str,['l1', 'l2', 'elasticnet', 'none']],
    'C': [int, [0.01, 100.0,0.01]],
    'solver': [str,['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']],
    'max_iter': [int, [100, 1000,10]],
    'l1_ratio': [int,[0.1,0.9,0.1]]  # Only used if penalty is 'elasticnet'
}
 
param_grid_random_forest_regressor = {
    'n_estimators': [int,[100, 1000,10]],
    'max_depth': [int,[5, 10, 15, 50]],
    'min_samples_split': [int,[2, 20,1]],
    'min_samples_leaf': [int,[1, 10,1]],
    'max_features': [str,['auto', 'sqrt', 'log2']],
    'bootstrap': [bool,[True, False]],
    'criterion': [str,['squared_error', 'absolute_error', 'poisson']]
}

param_grid_decision_tree_classifier = {
    'max_depth': [int, [1,20, 5]],  # [min, max, step_size]
    'min_samples_split': [int, [2, 15, 3]],
    'min_samples_leaf': [int, [1, 10, 2]],
    'max_features': [str, [ 'auto', 'sqrt', 'log2']],
    'max_leaf_nodes': [int, [1, 100, 1]],
    'min_weight_fraction_leaf': [int, [0.0, 0.2, 0.1]],
    'criterion': [str, ['gini', 'entropy', 'log_loss']],
    'splitter': [str, ['best', 'random']],
}

param_grid_random_forest_classifier = {
    'n_estimators': [int, [100, 500, 100]],
    'max_depth': [int, [1, 20, 1]],
    'min_samples_split': [int, [2, 10, 3]],
    'min_samples_leaf': [int, [1, 4, 1]],
    'max_features': [str, ['auto', 'sqrt', 'log2']],
    'bootstrap': [bool, [True, False]],
    'criterion': [str, ['gini', 'entropy', 'log_loss']],
}


def audio_autoplay(file):
    audio_file = open(file, 'rb')
    audio_bytes = audio_file.read()

    # Convert the audio bytes into a base64 string to embed it in HTML
    audio_base64 = base64.b64encode(audio_bytes).decode()

    # Define HTML for autoplay audio
    audio_html = f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
        </audio>
        """

    # Embed the HTML into Streamlit
    st.components.v1.html(audio_html, height=100)



def run(model_type,param_dict):
    run = {}
    run["params"] = dict()
    run["model_type"] = model_type
    run["n_iter"] = st.number_input(label="Number of models to train",key="nModels"+model_type,min_value=1,max_value=10,value=5,step=1)
    #grid search limit sleection
    for param_name, options in param_dict.items():
        #use_param = st.markdown(f"Limit {param_name} in Grid search")
        grid_search_options = select_grid_search_range(model_type,param_name,options)
        run["params"][param_name] = grid_search_options

    name_side, add_side,delete_side = st.columns(3,vertical_alignment="bottom")
    with name_side:
        run["name"] = st.text_input(label="run name",value="",key="name"+model_type,placeholder="Name your run here")
    with add_side:
        #might need a st.form_submit_button instead
        if st.button(label= f"Add {run['name']} search",key="add_run"+model_type,disabled= run["name"]==""):
            st.session_state.runs.append(run)
            st.toast("Added run 🐕")
    with delete_side:
        if st.button(label= f"Delete run: {run['name']}",key="delete_run"+model_type,disabled= run["name"]==""):
            for i,run_name in enumerate(st.session_state.runs):
                if run_name["name"] == run["name"]:
                    st.toast(f"Run {run['name']} deleted")
                    st.session_state.runs.remove(i)
                    break


def model(model_type,param_dict):
    model = {}
    model["model_type"] = model_type
    model["params"]= dict()
    for param_name, options in param_dict.items():
                #st.markdown(f"{param_name}")
                value = select_hyperparam_values(model_type,param_name,options)
                model["params"][param_name] = value


    name_side, add_side,delete_side = st.columns(3,vertical_alignment="bottom")
    with name_side:
        model["name"] = st.text_input(label="model name",key="name"+model_type,value="",placeholder="Name your model here")

    with add_side:
        #might need a st.form_submit_button instead
        if st.button(label= f"Add {model['name']}",key="add_model"+model_type,disabled = model["name"]==""):
            st.session_state.models.append(model)
            st.toast("Added model 🥳")

    with delete_side:
        if st.button(label = f"Delete model {model['name']}",key="delete"+model_type,disabled = model["name"]==""):
            for i,run_name in enumerate(st.session_state.model_names):
                if model_name["name"] == model["name"]:
                    st.toast(f"Model {model['name']} deleted")
                    st.session_state.models.remove(i)
                    break


def selection_wrapper(name,param_dict):
    use_random_search=False
    is_random_search_disabled = False
    if param_dict == {}:
        st.write("Due to a deterministic model random search is disabled")
        st.wirte("Multiple models are not beneficial")
        is_random_search_disabled = True
    use_random_search = st.checkbox(label="Use random search for hyperparameters",key="rs"+name,value=False,disabled=is_random_search_disabled)

    if use_random_search:
        run(name,param_dict)
    else:
        model(name,param_dict)


def select_hyperparam_values(model_type,name,hyperparam):
    dtype = hyperparam[0]
    if dtype == bool:
        value = st.checkbox(label=f"Use {name}",key="bool"+model_type+name)  
    elif dtype==str:
        value = st.selectbox(label=f"Set parameter {name}",key="str"+model_type+name,options=hyperparam[1],index=0)
    else:
        min_v = hyperparam[1][0]
        max_v = hyperparam[1][1]
        step_size = hyperparam[1][2]
        value = st.slider(label=f"Set parameter {name}",key="number"+model_type+name,min_value=min_v, max_value=max_v, value=min_v, step=step_size)
    return value



def select_grid_search_range(model_type,name,hyperparam):
    dtype = hyperparam[0]
    if dtype==bool:
        options = st.multiselect(label=f"Do you want to restrict {name}",key="bool_select"+model_type+name, options=[True,False],default=[True,False])        
    elif dtype==str:
        options = st.multiselect(label=f"Unselect values for {name}",key="options_select"+model_type+name,options=hyperparam[1],default=hyperparam[1])
    else:
        min_v = hyperparam[1][0]
        max_v = hyperparam[1][1]
        step_size = hyperparam[1][2]
        options = st.slider(label=f"Select range for {name}",key="range_select"+model_type+name,min_value=max_v, max_value=min_v, value=[min_v,max_v], step=step_size)
    return options



if task_is_regression:
    models = ["Linear Regression", "Regression Tree", "Ridge Regression","Random Forest Regressor"]
    lin_reg, reg_tree, rid_reg,rf_reg = st.tabs(models)
    
    with lin_reg:
        selection_wrapper(models[0],{})
    with reg_tree:
        selection_wrapper(models[1],param_grid_decision_tree_regression)
    with rid_reg:
        selection_wrapper(models[2],param_grid_ridge)
    with rf_reg:
        selection_wrapper(models[3],param_grid_random_forest_regressor)
    
else:
    models = ["Logistic Regression", "Decision Tree", "Random Forest Classificator"]
    log_reg, dec_tree, ran_for = st.tabs(models)
    
    with log_reg:
        selection_wrapper(models[0],param_grid_logistic)
    with dec_tree:
        selection_wrapper(models[1],param_grid_decision_tree_classifier )
    with ran_for:
        selection_wrapper(models[2],param_grid_random_forest_classifier)

cv_side, k_fold_side= st.columns(2,vertical_alignment="center")
with cv_side:
    use_cv = st.checkbox("Use crossfold validation",value=False)
with k_fold_side:
    if use_cv:
        k = st.slider(label="Fold Number (k)",min_value=2, max_value=25,step=1,value=5)
#metrics
if f:
    if task_is_regression:
        metric_options = ["Accuracy","Precision","F1-Score","Recall","Matthew Correlation Coefficient"]
        metric_descriptions = ["a","b","c","d","e"]
        metric = st.selectbox(label="performance metric",options=metric_options,index=0)
        

    else:
        metric_options = ["mean squared error","mean absolute error","root mean squared error","explained variance score","max error"]
        metric_descriptions = ["","","","","",""]
        metric = st.selectbox(label="performance metric",options=metric_options,index=0)


def check_JSON_validity():
    #TODO
    if conf is None:
        create_JSON()
    st.session_state.json_valid=True

def create_JSON():
    global conf
    configuration = dict()
    configuration["Runs"] = dict()
    configuration["Models"] = dict()
    configuration["Training"] = dict()
    for model in st.session_state.models:
        model_name = model["name"]
        configuration["Models"][model_name] = dict()
        configuration["Models"][model_name]["model_type"] = model["model_type"]
        configuration["Models"][model_name]["params"] = model["params"]

    for run in st.session_state.runs:
        run_name = run["name"]
        configuration["Runs"][run_name] = dict()
        configuration["Runs"][run_name]["model_type"] = run["model_type"]
        configuration["Runs"][run_name]["params"] = run["params"]

    configuration["Training"]["is_regression"] = st.session_state.task
    configuration["Training"]["CV"] = st.session_state.use_cv
    if st.session_state.use_cv:
        configuration["Training"]["k_fold"] = st.session_state.k_fold
    configuration["Training"]["metric"] = st.session_state.metric
    configuration["Training"]["target"] = st.session_state.target
    configuration["Training"]["seed"] = st.session_state.seed
    configuration["Training"]["train_test_split"] = st.session_state.train_test_split
    
    conf = json.dumps(configuration)


def process_JSON():

    json_file = open('json1')
    json_str = json1_file.read()
    json_data = json.loads(json1_str)[0]
    st.session_state.metric = json_data["Training"]["metric"] 
    st.session_state.use_cv = json_data["Training"]["CV"] 
    st.session_state.k_fold = json_data["Training"]["k_fold"]
    st.session_state.target = json_data["Training"]["target"] 
    st.session_state.seed = json_data["Training"]["seed"]
    st.session_state.train_test_split =json_data["Training"]["train_test_split"]

    for model_name,model in json_data["Models"].items():
        new_dict_model = model
        new_dict_run["name"] = model_name
        st.session_state.models.append(new_dict_run)
        
    for run_name,run in json_data["Runs"].items():
        new_dict_run = run
        new_dict_run["name"] = run_name
        st.session_state.runs.append(new_dict_run)


def check_JSON():
    check_JSON_validity()
    st.session_state.json_checked=True


if f:
    if st.button(label="Check Configuration for validity"):
        create_JSON()
        check_JSON()
        if not st.session_state.json_valid and st.session_state.json_checked:
            st.error("Your Configuration file is errorneous")
        elif not st.session_state.json_checked:
            st.warning("Your Configuration file is unchecked.")
        submit_side,preprocess_side,download_conf_side = st.columns(3)
        with submit_side:
            if st.button("Start ML Pipeline"):
                stats,trained_models, params_trained_models = be.start_ML_Pipeline(conf)
        with preprocess_side:
            #TODO
            st.write("Wait for preprocess to uncomment")
            #st.download_button(label="Download configuration",data=be.preprocess_data(df,conf).to_csv(),file_name="processed_data.csv")
        with download_conf_side:
            st.download_button(label="Download configuration",data=conf,file_name="configuration.json")
            
#Add datavisalisation
stats = False

##Iced Matcha LAtte audio
#audio_autoplay("./music/Bauch_Beine_Po.mp3")

audio_file = open('./music/Bauch_Beine_Po.mp3', 'rb')
audio_bytes = audio_file.read()

audio_base64 = base64.b64encode(audio_bytes).decode()

# Define HTML for autoplay audio
audio_html = f"""
    <audio autoplay loop>
    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    Your browser does not support the audio element.
    </audio>
    """

# Embed the HTML into Streamlit
st.components.v1.html(audio_html, height=100)





if stats:
    pass