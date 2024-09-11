import pandas as pd
import streamlit as st
from streamlit_image_comparison import image_comparison
from streamlit_extras.vertical_slider import vertical_slider
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import base64


st.set_page_config(page_title="ML image data site", layout="wide")

def salt_pepper_noise(img,p=0.4):
    pixels_without_noise = np.random.rand(*img.shape[:2])>p
    random_noise = np.random.choice([-255,255],size=pixels_without_noise.shape)
    random_noise[pixels_without_noise] = 0
    img = img + random_noise[:,:,np.newaxis]
    img[img>255]=255
    img[img<0]=0
    return img

def gaussian_noise(img,std=0.1,all_channels=False):
    if all_channels:
        noise = np.random.normal(0,std,img.shape[:2]).astype(np.uint8)
        img = img + noise[:,:,np.newaxis] 
    else:
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = img + noise 
    img[img>255]=255
    img[img<0]=0
    return img


def sharpening(img,strong,excessive):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    if strong:
        kernel = np.array([[1, -2, 1],[-2, 5, -2],[1, -2, 1]])
    if excessive:
        kernel = np.array([[-1, -1, -1],[-1,  9, -1],[-1, -1, -1]])

    augmented_image = cv2.filter2D(img, -1, kernel)
    return augmented_image


def flip(img,horizontal,vertical):
    if horizontal:
        img = cv2.flip(img,1)
    if vertical:
        img = cv2.flip(img,0)
    return img


def rotation(img,rotate): #90degree
    rotated_image = cv2.transpose(img)
    rotated_image = cv2.flip(rotated_image, flipCode=0)
    return rotated_image


#example_options = {"param_name":, "type": , "min_value":,"max_value":,"left_setting": "right_setting":}
#1 numerical
blur_options = {"params": {"blur":{ "type":int , "min_value":2,"max_value":30,"setting":10}}}
median_blur_options = {"params": {"k_size":{ "type":int , "min_value":2,"max_value":30,"setting":10}}}
sp_noise_options = {"params": {"p":{ "type":float , "min_value":0.01,"max_value":0.99,"setting":0.3}}}
#1 numerical + 1 bool
gaussian_noise_options = {"params":{"std": {"type":int , "min_value":1,"max_value":100,"setting":25},"all_channels":{"type": bool,"setting":True}}}
#2 numerical
canny_options = {"params": {"lower_bound":{"type":int , "min_value":10,"max_value":250,"setting":80},"higher_bound":{"type":int , "min_value":10,"max_value":255,"setting":120}}}
gaussian_blur_options = {"params": {"blur":{"type":int , "min_value":2,"max_value":50,"setting":10},"std":{"type":int , "min_value":1,"max_value":100,"setting":15}}}
contrast_brightness_options = {"params": {"contrast":{"type":float , "min_value":0.1,"max_value":3.0,"setting":1.5},"brightness":{"type":int , "min_value":-30,"max_value":30,"setting":15}}}
#rotation has not options always 90 degree
flip_options = {"params":{"horizontal_flip": {"type": bool, "setting":True},"vertical_flip":{"type": bool,"setting":False}}}
sharpening_options = {"params":{"strong": {"type": bool, "setting":True},"excessive":{"type": bool,"setting":False}}}
rotation_options = {"params":{"rotations": {"type": int,"setting": 1}}}


def use_filter(pic,filtering,settings):
    if filtering == cv2.blur:
        settings = [(settings[0],settings[0])]
    elif filtering == cv2.GaussianBlur:
        settings = [(settings[0],settings[0]),settings[1]]
    elif filtering ==cv2.Canny:
        settings = [settings[0],settings[1]]
    elif filtering == cv2.medianBlur:
        settings = [settings[0]]
    elif filtering == salt_pepper_noise:
        settings = [settings[0]]
    elif filtering == gaussian_noise:
        settings = [settings[0],settings[1]]
    elif filtering == sharpening:
        settings = [settings[0],settings[1]]
    elif filtering == rotation:
        settings = [settings[0]]
    elif filtering == flip: #1,0
        settings = [settings[0],settings[1]]
    else:# filtering == cv2.convertScaleAbs: #Contrast adjustment
        settings = [settings[0],settings[1]]

    return filtering(pic,*settings)


def load_pic(file_name):
    extension = file_name.split(".")[-1]
    if extension=="npz":
        pic = np.load(file_name)["train_images"][0,:,:,:]
    else:
        pic = cv2.imread(file_name)
    return pic

#anpassen 
@st.dialog("Select your filter here",width="large")
def filter_one_numerical(filter_name,pic,filtering,options):
    st.write(f"Select for filter settings for {filter_name}")
    param_name = next(iter(options["params"]))
    param = options["params"][param_name]

    pic_left_label = "Original"
    pic_right_label = filter_name
    augmented_pic = use_filter(pic,filtering,[param["setting"]])
    image_comp_side, right_pic_side = st.columns([0.9,0.1],gap="small",vertical_alignment="center")
    with right_pic_side:
        param["setting"] = vertical_slider(label=param_name, min_value=param["min_value"], max_value=param["max_value"], default_value=param["setting"])

    render_side, right_button_side = st.columns(2,vertical_alignment="center")

    with render_side:
        if st.button("Render"):
            augmented_pic = use_filter(pic,filtering,[param["setting"]])

    with image_comp_side:
        image_comparison(
            img1=pic,
            img2=augmented_pic,
            label1=pic_left_label,
            label2=pic_right_label,
            width=600
        )

    with right_button_side:
        if st.button("Take setting"):
            st.session_state[filter_name] = param["setting"]
            st.rerun()


@st.dialog("Select your filter here",width="large")
def filter_two_numerical(filter_name,pic,filtering,options):
    st.write(f"Select for filter settings for {filter_name}")
    param1_name,param2_name = options["params"].keys()
    param1 = options["params"][param1_name]
    param2 = options["params"][param2_name]

    pic_left_label = "Original"
    pic_right_label = filter_name
    augmented_pic = use_filter(pic,filtering,[param1["setting"],param2["setting"]])
    image_comp_side, right_pic_side,right_right_pic_side = st.columns([0.84,0.08,0.08],gap="small",vertical_alignment="center")
    with right_pic_side:
        param1["setting"] = vertical_slider(label=param1_name, min_value=param1["min_value"], max_value=param1["max_value"], default_value=param1["setting"])
    with right_right_pic_side:
        param2["setting"] = vertical_slider(label=param2_name, min_value=param2["min_value"], max_value=param2["max_value"], default_value=param2["setting"])
        
        if cv2.Canny==filtering:
            if param2["setting"]<param1["setting"]:
                param2["setting"]= param1[setting]+1

    render_side, right_button_side = st.columns(2,vertical_alignment="center")
    with render_side:
        if st.button("Render"):
            augmented_pic = use_filter(pic,filtering,[param1["setting"],param2["setting"]])

    with image_comp_side:
        image_comparison(
            img1=pic,
            img2=augmented_pic,
            label1=pic_left_label,
            label2=pic_right_label,
            width=580
        )

    with right_button_side:
        if st.button("Take setting"):
            st.session_state[filter_name] = [param1["setting"],param2["setting"]]
            st.rerun()


@st.dialog("Select your filter here",width="large")
def filter_one_numerical_one_bool(filter_name,pic,filtering,options):
    st.write(f"Select for filter settings for {filter_name}")
    param1_name, param2_name = options["params"].keys()
    param1 = options["params"][param1_name]
    param2 = options["params"][param2_name]

    pic_left_label = "Original"
    pic_right_label = filter_name
    augmented_pic = use_filter(pic,filtering,[param1["setting"],param2["setting"]])
    image_comp_side, right_pic_side = st.columns([0.9,0.1],gap="small",vertical_alignment="center")
    with right_pic_side:
        param1["setting"] = vertical_slider(label=param1_name, min_value=param1["min_value"], max_value=param1["max_value"], default_value=param1["setting"])

    param2["setting"] = st.checkbox(label=param2_name,value=param2["setting"])
    render_side, right_button_side = st.columns(2,vertical_alignment="center")

    with render_side:
        if st.button("Render"):
            augmented_pic = use_filter(pic,filtering,[param1["setting"],param2["setting"]])

    with image_comp_side:
        image_comparison(
            img1=pic,
            img2=augmented_pic,
            label1=pic_left_label,
            label2=pic_right_label,
            width=600
        )

    with right_button_side:
        if st.button("Take setting"):
            st.session_state[filter_name] = [param1["setting"],param2["setting"]]
            st.rerun()


@st.dialog("Select your filter here",width="large")
def rotate_dialog(filter_name,pic,filtering,options):
    st.write(f"Select for filter settings for {filter_name}")
    param_name = next(iter(options["params"]))
    param = options["params"][param_name]

    augmented_pic = use_filter(pic,filtering,[param["setting"]])


    right_side,left_side = st.columns(2)
    with left_side:
        if st.button("Rotate"):
            param["setting"] = param["setting"]+1
            augmented_pic = use_filter(augmented_pic,filtering,[param["setting"]])

    st.image(augmented_pic,width=650)

    with right_side:
        if st.button("Take setting"):
            if filter_name not in st.session_state:
                st.session_state[filter_name] = param["setting"]
            else:
                if param["setting"]//4 != 0 and param["setting"]//4 not in st.session_state[filter_name]:
                    st.session_state[filter_name].append(param["setting"]//4)
            st.rerun()


@st.dialog("Select your filter here",width="large")
def two_bool_dialog(filter_name,pic,filtering,options):

    st.write(f"Select for filter settings for {filter_name}")
    param1_name, param2_name = options["params"].keys()
    param1 = options["params"][param1_name]
    param2 = options["params"][param2_name]

    is_h_flip = param1["setting"]
    is_v_flip = param2["setting"]

    left_side,right_side = st.columns(2)
    with left_side:
        if st.checkbox("horizontal flip",value=is_h_flip):
            is_h_flip = not is_h_flip
            augmented_pic = use_filter(pic,filtering,[is_h_flip,is_v_flip])
    
    with right_side:
        if st.checkbox("vertical flip",value=is_v_flip):
            is_v_flip = not is_v_flip
            augmented_pic = use_filter(pic,filtering,[is_h_flip,is_v_flip])


    augmented_pic = use_filter(pic,filtering,[is_h_flip,is_v_flip])
    st.image(augmented_pic,width=700)

    if st.button("Take setting"):
        st.session_state[filter_name] = [is_h_flip,is_v_flip]
        st.rerun()

def img_button(img,text):
    image = Image.fromarray(image_np)

    # Convert the PIL image to a byte stream (PNG format)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_data = buffer.getvalue()

    # Encode the image in base64 to embed it in HTML
    encoded_image = base64.b64encode(img_data).decode()

    # Create HTML with the image as a button
    button_html = f"""
        <style>
        .image-button {{
            background-color: transparent;
            border: none;
            padding: 2;
            cursor: pointer;
        }}
        </style>
        <form action="">
            <button class="image-button" type="submit">
                <img src="data:image/png;base64,{encoded_image}" alt="Button Image" width="300">
                <h4>{text}
                </h4>
            </button>
        </form>
    """

    # Render the button in Streamlit
    st.markdown(button_html, unsafe_allow_html=True)

def train_model(data,augmented_data,use_augmented_data=False):
    #TODO implement
    pass


def augment_uploaded_data():
    #TODO implement
    pass


st.title("Welcome to the your Machine learning Image augmentation side")

#f = st.file_uploader("upload_file")
#if f:
pic = load_pic("bloodmnist.npz")
st.success("Picture uploaded")
#TODO add file loader
#TODO add picture preview
#https://discuss.streamlit.io/t/display-images-one-by-one-with-a-next-button/21976/2
##
#TODO Select one pic for preview

#Make order in chaos here
st.header("Data Augmentation")
first_row_left,first_row_middle,first_row_right = st.columns(3)

second_row_left,second_row_middle,second_row_right = st.columns(3)

third_row_left,third_row_middle,third_row_right = st.columns(3)

with first_row_left:
    if img_button(blur_img,"Use the Blur filter"):
        filter_one_numerical("Blur",pic,cv2.blur,blur_options)
with first_row_middle:
    if img_button(median_blur_img,"Use the Median Blur filter"):
        filter_one_numerical("Median_blur",pic,cv2.medianBlur,median_options)
with first_row_right:
    if img_button(gaussian_blur_img,"Use the Gaussian Blur filter"):
        filter_two_numerical("Gaussian_Blur",pic,cv2.GaussianBlur,gaussian_blur_options)


with second_row_left: 
    if img_button(sharpened_img,"Use the sharpening"):
        two_bool_dialog("sharpening",pic,sharpening,sharpening_options)
with second_row_middle: 
    if img_button(blur_img,"Use the Gaussian noise filter"):
        filter_one_numerical_one_bool("Gaussian_noise",pic,gaussian_noise,gaussian_noise_options)
with second_row_right:
    if img_button(blur_img,"Use the Salt and Pepper noise"):
        filter_one_numerical("Salt_pepper_noise",pic,salt_pepper_noise,sp_noise_options)


with third_row_left: 
    if img_button(canny_img,"Use the Edge filter (Canny)"):
        filter_two_numerical("Canny",pic,cv2.Canny,canny_options)

with third_row_middle:
    if img_button(blur_img,"Rotate Images"):
        rotate_dialog("rotate",pic,rotation,rotation_options)

with third_row_right:
    if img_button(blur_img,"Flip Images"):
        two_bool_dialog("flip",pic,flip,flip_options)


#Use on all pictures
if st.button("Augment Data"):
    #TODO add progress bar
    augmented_data = augment_uploaded_data(data)
    st.download_button("Download augmented data",data=augmented_data,file_name = "augmented_data")
#Train models and return comparison augmented vs not augmented


st.header("The Impact of Data augmentation on Deep Learning")
st.write("in the following we train two models. One on the original dataset and one on the newly created one. Then those models are compared")
if st.button("Train models"):
    with st.spinner(text="In progress first model...")
        model_without_augmentation,stats = train_model(data,augmented_data)
    with st.spinner(text="In progress second model...")
        model_with_augmentation,stats_a = train_model(data,augmented_data,use_augmented_data=True)

    model1_acc_data = stats[0]
    model2_acc_data = stats_a[0]
    model1_acc_aug_data = stats[1]
    model2_acc_aug_data = stats_a[1]
    model1_f1_data = stats[2]
    model2_f1_data = stats_a[2]
    model1_f1_aug_data = stats[2]
    model2_f1_aug_data = stats_a[2]

    st.header("Model Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", model2_acc_data, model2_acc_data-model1_acc_data)
    col2.metric("Accuracy on augmented", model2_acc_aug_data, model2_acc_aug_data - model1_acc_aug_data)
    col3.metric("F1 Score", model2_f1_data,model2_f1_data-model1_f1_data)

    st.download_button("Download model trained on augmented data", data=model_with_augmentation,file="model_augmented")
