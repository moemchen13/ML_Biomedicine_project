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


# Modelselection methods
example_param_dict= {
    "Interest": [int,[0,10,1]],
    "is_usable": [bool,[]],
    "difficulty":[str,["easy","medium","hard"]],
}


def model(name,param_dict):
    use_random_search=False
    grid_search_dict = dict()
    values = dict()
    if not param_dict == dict():
        use_random_search = st.checkbox("Use random Grid search")
        
        if use_random_search:
            models_to_train = st.number_input(label="Number of models to train",min_value=1,max_value=10,value=5,step=1)
            #grid search limit sleection
            for param_name, options in param_dict.items():
                use_param = st.markdown(f"Limit {param_name} in Grid search")
                grid_search_options = select_grid_search_range(options)
                grid_search_dict[param_name] = grid_search_options
        else:
            #value selection
            for param_name, options in param_dict.items():
                st.markdown(f"{param_name}")
                value = select_hyperparam_values(options)
                values[param_name] = value

    else:
        st.write("Model is deterministic")
    return use_random_search, grid_search_dict, values   



def select_hyperparam_values(hyperparam):
    dtype = hyperparam[0]
    if dtype == bool:
        value = st.checkbox(label="")  
    elif dtype==str:
        value = st.selectbox(label="",options=hyperparam[1],index=0)
    else:
        min_v = hyperparam[1][0]
        max_v = hyperparam[1][1]
        step_size = hyperparam[1][2]
        value = st.slider(label="",min_value=min_v, max_value=max_v, value=[min_v], step=step_size)
    return value



def select_grid_search_range(hyperparam):
    dtype = hyperparam[0]
    if dtype==bool:
        options = st.multiselect(label="", options=["True","False"],default=["True","False"])        
    elif dtype==str:
        options = st.multiselect(label="",options=hyperparam[1],default=hyperparam[1])
    else:
        min_v = hyperparam[1][0]
        max_v = hyperparam[1][1]
        step_size = hyperparam[1][2]
        options = st.slider(label="",min_value=max_v, max_value=min_v, value=[min_v,max_v], step=step_size)
    return options



def add_model(model_name):
    #TODO
    pass
    st.toast("Added model 🥳")

def add_run(run_name):
    #TODO
    pass
    st.toast("Added run 🐕")

def name_add_run_model(use_random_search):
    name_side, add_side = st.columns(2,vertical_alignment="bottom")
    with name_side:
        if use_random_search:
            run_name = st.text_input(label="run name",value="",placeholder="Name your run here")
        else:
            model_name = st.text_input(label="model name",value="",placeholder="Name your model here")
    with add_side:
        if use_random_search:
            st.button(label= f"Add {models[0]} search",on_click=add_run(run_name))
        else:
            st.button(label= f"Add {models[0]}",on_click=add_model(model_name))
    
#model selection

use_random_search = False
if task_is_regression:
    models = ["Linear Regression", "Regression Tree", "Ridge Regression"]
    lin_reg, reg_tree, rid_reg = st.tabs(models)
    
    with lin_reg:
        use_random_search, grid_search_dict, values  = model("Example",example_param_dict)
        name_add_run_model(use_random_search)
    with reg_tree:
        st.markdown("fff")
    with rid_reg:
        st.markdown("www")
    
else:
    models = ["Logistic Regression", "Decision Tree", "Random Forest"]
    log_reg, dec_tree, ran_for = st.tabs(models)
    
    name=""
    with log_reg:
        use_grid_search, grid_search_dict, values  = model("Example",example_param_dict)
        name_add_run_model(use_random_search)
    with dec_tree:
        st.markdown("dd")
    with ran_for:
        st.markdown("fefe")

#
def recommended_metric(df,target,is_reg):
    #TODO
    pass
#

#metrics
if f:
    index = recommended_metric(df,target,task_is_regression)
    if task_is_regression:
        metric_options = ["Accuracy","Precision","F1-Score","Recall","Balanced Accuracy","Matthew Correlation Coefficient"]
        metric_descriptions = ["a","b","c","d","e","f"]
        metric = st.selectbox(label="performance metric",options=metric_options,index=0)
        for i,metric_option in enumerate(metric_options):
            if metric_option == metric:
                metric_description = metric_descriptions[i]
                st.write(metric_description)

    else:
        metric_options = ["mean squared error","mean absolute error","root mean squared error","explained variance score","max error"]
        metric_descriptions = ["","","","","",""]
        metric = st.selectbox(label="performance metric",options=metric_options,index=0)
        for i,metric_option in enumerate(metric_options):
            if metric_option == metric:
                metric_description = metric_descriptions[i]
                st.write(metric_description)

def start_ML_Pipeline():
    #TODOs
    pass

st.form_submit_button(label="Start ML Pipeline",on_click=start_ML_Pipeline())

#Add datavisalisation
