import pandas as pd
import streamlit as st
from streamlit_image_comparison import image_comparison
from streamlit_image_select import image_select
from streamlit_extras.vertical_slider import vertical_slider
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from stqdm import stqdm
import tensorflow as tf 
from tensorflow.keras import callbacks
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import time

st.set_page_config(page_title="ML image data site", layout="wide")


#example_options = {"param_name":, "type": , "min_value":,"max_value":,"left_setting": "right_setting":}
#1 numerical
blur_options = {"params": {"blur":{ "type":int , "min_value":2,"max_value":30,"setting":5}}}
sp_noise_options = {"params": {"p":{ "type":float , "min_value":0.01,"max_value":0.99,"setting":0.3}}}
median_blur_options = {"params": {"k_size":{ "type":int , "min_value":3,"max_value":31,"setting":5}}}
#1 numerical + 1 bool
gaussian_noise_options = {"params":{"std": {"type":int , "min_value":1,"max_value":50,"setting":25},"all_channels":{"type": bool,"setting":True}}}
#2 numerical
canny_options = {"params": {"lower_bound":{"type":int , "min_value":10,"max_value":250,"setting":80},"higher_bound":{"type":int , "min_value":10,"max_value":255,"setting":120}}}
gaussian_blur_options = {"params": {"blur":{"type":int , "min_value":2,"max_value":50,"setting":2},"std":{"type":int , "min_value":1,"max_value":100,"setting":15}}}
contrast_brightness_options = {"params": {"alpha":{"type":int , "min_value":-30,"max_value":30,"setting":10},"beta":{"type":float, "min_value":0.1,"max_value":3.0,"setting":1.1}}}
#rotation has not options always 90 degree
flip_options = {"params":{"horizontal_flip": {"type": bool, "setting":True},"vertical_flip":{"type": bool,"setting":False}}}
sharpening_options = {"params":{"strong": {"type": bool, "setting":True},"excessive":{"type": bool,"setting":False}}}
rotation_options = {"params":{"rotations": {"type": int,"setting": 1}}}


def salt_pepper_noise(img,p=0.4):
    pixels_without_noise = np.random.rand(*img.shape[:2])>p
    random_noise = np.random.choice([-255,255],size=pixels_without_noise.shape)
    random_noise[pixels_without_noise] = 0
    img = img + random_noise[:,:,np.newaxis]
    img[img>255]=255
    img[img<0]=0
    img = img.astype(np.uint8)
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

def gaussianBlur(pic,blur,std):
    if (blur[0]%2) ==0:
        if blur[0]<2:
            blur = (blur[0]+1,blur[0]+1)
        else:
            blur = (blur[0]-1,blur[0]+-1)
    return cv2.GaussianBlur(pic,blur,std)
        

def medianBlur(pic,settings):
    if (settings%2) ==0:
        if settings<2:
            settings +=1
        else:
            settings -= 1
    return cv2.medianBlur(pic,settings)


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
    for i in range(rotate):
        rotated_image = cv2.transpose(img)
        rotated_image = cv2.flip(rotated_image, flipCode=0)
    return rotated_image


def use_filter(pic,filtering,settings:list):
    if filtering == cv2.blur:
        settings = [(settings[0],settings[0])]
    elif filtering == gaussianBlur:
        settings = [(settings[0],settings[0]),settings[1]]
    elif filtering ==cv2.Canny:
        settings = [settings[0],settings[1]]
    elif filtering == medianBlur:
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


def load_data(file_name):
    extension = file_name.split(".")[-1]
    return np.load(file_name)


def get_data_from_uploaded_list(files):
    data = []
    for file in files:
        data_from_file = load_data(file.name)
        X_train = np.concatenate((data_from_file["train_images"],data_from_file["val_images"]),axis=0)
        y_train = np.concatenate((data_from_file["train_labels"],data_from_file["val_labels"]),axis=0)
        X_test = data_from_file["test_images"]
        y_test = data_from_file["test_labels"]
        data.append((X_train,X_test,y_train,y_test))  
    return data

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
        if param["type"]==float:
            param["setting"] = vertical_slider(label=param_name, min_value=param["min_value"], max_value=param["max_value"],step=0.01, default_value=param["setting"])
        else:
            param["setting"] = vertical_slider(label=param_name, min_value=param["min_value"], max_value=param["max_value"],step=1, default_value=param["setting"])
            
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
    rotate=1

    augmented_pic = use_filter(pic,filtering,[param["setting"]])

    right_side,left_side = st.columns(2)
    with left_side:
        if st.button("Rotate"):
            rotate += 1
            if rotate % 4 == 0:
                rotate = 1
            augmented_pic = use_filter(augmented_pic,filtering,[rotate])

    st.image(augmented_pic,width=700)

    with right_side:
        if st.button("Take setting"):
            st.session_state[filter_name] = [rotate]
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
        

    # Render the button in Streamlit
    st.markdown(button_html, unsafe_allow_html=True)
    return True


def add_augmented_sample(sample):
    augmented_samples = []
    
    if "Blur" in st.session_state:
        augmented_samples.append(use_filter(sample,cv2.blur,st.session_state.Blur))
    if "Median_blur" in st.session_state:
        augmented_samples.append(use_filter(sample,medianBlur,st.session_state.Median_blur))
    if "Gaussian_blur" in st.session_state:
        augmented_samples.append(use_filter(sample,gaussianBlur,st.session_state.Gaussian_blur))
    if "sharpening" in st.session_state:
        augmented_samples.append(use_filter(sample,sharpening,st.session_state.sharpening))
    if "Gaussian_noise" in st.session_state:
        augmented_samples.append(use_filter(sample,gaussian_noise,st.session_state.Gaussian_noise))
    if "Salt_pepper_noise" in st.session_state:
        augmented_samples.append(use_filter(sample,salt_pepper_noise,st.session_state.Salt_pepper_noise))
    if "Canny" in st.session_state:
        augmented_samples.append(np.repeat(use_filter(sample,cv2.Canny,st.session_state.Canny)[:, :, np.newaxis], 3, axis=2))
    if "rotate" in st.session_state:
        print(st.session_state.rotate)
        augmented_samples.append(use_filter(sample,rotation,st.session_state.rotate))
    if "flip" in st.session_state:
        augmented_samples.append(use_filter(sample,flip,st.session_state.flip))
    if "CSA_adjustment" in st.session_state:
        augmented_samples.append(use_filter(sample,cv2.convertScaleAbs,st.session_state.CSA_adjustment))
    augmented_samples = np.stack(augmented_samples)
    return augmented_samples


def augment_uploaded_data(X_train,X_test,y_train,y_test):
    aX_train,aX_test,ay_train,ay_test = [None]*4

    n_images = X_train.shape[0]+X_test.shape[0]

    for i in stqdm(range(n_images),desc="Create augmented pictures"):
        #in test

        if i>=X_train.shape[0]:
            index_test_set = i-X_train.shape[0]
            #print(f"i:{i} index:{index_test_set}")
            
            if aX_test is None:
                aX_test = X_test[np.newaxis,index_test_set,:,:,:]
                ay_test = y_test[np.newaxis,index_test_set]
            else:
                aX_test = np.concatenate((aX_test,X_test[np.newaxis,index_test_set,:,:,:]),axis=0)
                ay_test = np.concatenate((ay_test,y_test[np.newaxis,index_test_set]),axis=0)
            aX_test_sample = add_augmented_sample(X_test[index_test_set,:,:,:])
            aX_test= np.concatenate((aX_test,aX_test_sample),axis=0)
            ay_test = np.concatenate((ay_test,ay_test[np.newaxis,index_test_set]),axis=0)

        else:

            if aX_train is None:
                aX_train = X_train[np.newaxis,i,:,:,:]
                ay_train =y_train[np.newaxis,i]
            else:
                aX_train = np.concatenate((aX_train,X_train[np.newaxis,i,:,:,:]),axis=0)
                ay_train = np.concatenate((ay_train,y_train[np.newaxis,i]),axis=0)
            aX_train_sample = add_augmented_sample(X_train[i,:,:,:])
            aX_train = np.concatenate((aX_train,aX_train_sample),axis=0)
            ay_train = np.concatenate((ay_train,y_train[np.newaxis,i]),axis=0)
    return aX_train,aX_test,ay_train,ay_test


def select_picture(index,X_train,X_test):
    if index>X_train.shape[0]:
        pic = X_test[index-X_train.shape[0]]
    else:
        pic = X_train[index]
    return pic


def show_pic(index,X_train,X_test):
    pic = select_picture(index,X_train,X_test)
    
    image = Image.fromarray(pic)
    st.image(image,width=700)
    st.markdown("This is your selected_picture")

    st.session_state.selected_pic += 1
    st.session_state.selected_pic = st.session_state.selected_pic % (X_train.shape[0]+X_test.shape[0])


def generate_all_augmented_pics(pic):
    pics = []
    pics.append(use_filter(pic,cv2.blur,[blur_options["params"]["blur"]["setting"]]))
    pics.append(use_filter(pic,medianBlur,[median_blur_options["params"]["k_size"]["setting"]]))
    pics.append(use_filter(pic,gaussianBlur,[gaussian_blur_options["params"]["blur"]["setting"],gaussian_blur_options["params"]["std"]["setting"]]))
    pics.append(use_filter(pic,cv2.Canny,[canny_options["params"]["lower_bound"]["setting"],canny_options["params"]["higher_bound"]["setting"]]))
    pics.append(use_filter(pic,cv2.convertScaleAbs,[contrast_brightness_options["params"]["alpha"]["setting"],contrast_brightness_options["params"]["beta"]["setting"]]))
    pics.append(use_filter(pic,gaussian_noise,[gaussian_noise_options["params"]["std"]["setting"],gaussian_noise_options["params"]["all_channels"]["setting"]]))
    pics.append(use_filter(pic,sharpening,[sharpening_options["params"]["strong"]["setting"],sharpening_options["params"]["excessive"]["setting"]]))
    pics.append(use_filter(pic,rotation,[rotation_options["params"]["rotations"]["setting"]]))
    pics.append(use_filter(pic,salt_pepper_noise,[sp_noise_options["params"]["p"]["setting"]]))
    pics.append(use_filter(pic,flip,[flip_options["params"]["horizontal_flip"]["setting"],flip_options["params"]["vertical_flip"]["setting"]]))
    return pics


def combine_data(data,labels_not_same):
    X_train,X_test,y_train,y_test,highest_label = [None]*5
    for data_set in data:
        X_train_set,X_test_set,y_train_set,y_test_set = data_set
        if X_train is None:
            X_train = X_train_set
            X_test = X_test_set
            y_train = y_train_set
            y_test = y_test_set
            highest_label=y_train_set.max()
        else:
            X_train = np.concatenate((X_train,X_train_set),axis=0)
            X_test = np.concatenate((X_test_set),axis=0)
            if labels_not_same:
                y_train_set += highest_label + 1
                y_test_set += highest_label + 1
            y_train = np.concatenate((y_train,y_train_set),axis=0)
            y_test = np.concatenat((y_test,y_test_set),axis=0)
            highest_label+=y_train_set.max()+1 #accounting for zero label
    return X_train,X_test,y_train,y_test


def create_model(width, height,n_channels, n_labels,third_dimension=None):
    model = Sequential()
    if third_dimension:
        model.add(Conv3D(64, (3, 3, 3), activation="relu", input_shape=(depth, width, height, n_channels)))
        model.add(MaxPooling3D((2, 2, 2)))
        if n_channels !=1:
            model.add(Conv3D(1, (1, 1, 1), activation="relu"))
        model.add(Conv3D(32, (3, 3, 3), activation="relu"))
        model.add(MaxPooling3D((2, 2, 2)))
        
    else:
        #Accept RGB images
        model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(width, height, n_channels)))
        model.add(MaxPooling2D((2, 2)))
        #Reduce the 3 channels (RGB) to 1 channel using a Conv2D layer via 1x1 conv
        if n_channels!=1:
            model.add(Conv2D(1, (1, 1), activation="relu")) 
        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.4))
    model.add(Flatten())
    #MLP
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(n_labels, activation="softmax"))
    
    return model

def train_model(data,augmented_data,use_augmented_data=False):
    data_to_train = data

    if use_augmented_data:
        data_to_train = augmented_data

    X_train,X_test,y_train,y_test = data_to_train
    n_labels = y_train.max()

    if len(X_train.shape) == 5:
        model = create_model(X_train.shape[1],X_train.shape[2],X_train.shape[4],n_labels,third_dimension=X_train.shape[3])
    else:
        model = create_model(X_train.shape[1],X_train.shape[2],X_train.shape[3],n_labels)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy",AUC(multi_label=True)])
    early_stop = callbacks.EarlyStopping(monitor='val_loss',patience=20)
    model.fit(X_train, y_train, epochs=100, batch_size=32,callbacks=[early_stop],verbose=1,validation_split=0.2)

    _,model_acc,auc = model.evaluate(data[1],data[3])
    _,model_acc_aug,_ = model.evaluate(augmented_data[1],augmented_data[3])

    stats = (model_acc,model_acc_aug,auc)

    return model,stats


st.title("Welcome to the your Machine learning Image augmentation side")

n_pics = 0

if "f" in st.session_state:
    f = st.session_state.f

upload_files_same_labels = st.checkbox("Files have same labels")
if upload_files_same_labels:
    f = st.file_uploader("upload file same labels",accept_multiple_files=True,type=["npz"])
    st.session_state.f = f
else:
    f = st.file_uploader("upload file different labels",accept_multiple_files=True,type=["npz"])
    st.session_state.f = f

if f:
    if type(f) != list:
        f = [f]
    data = get_data_from_uploaded_list(f)
    X_train,X_test,y_train,y_test = combine_data(data,upload_files_same_labels)
    
    st.session_state.data = X_train[:300],X_test[:100],y_train[:300],y_test[:100]
    n_pics = X_train.shape[0] + X_test.shape[0]
    st.success(f"{n_pics} Pictures uploaded")

    left,middle,right = st.columns([0.3,0.5,0.2])
    with middle:
        st.session_state.selected_pic = image_select(
        label="Select a picture",
        images=[X_train[0,:,:,:],X_train[1,:,:,:],X_train[2,:,:,:],X_train[3,:,:,:],X_train[4,:,:,:]],
        captions=["", "", "", "",""],use_container_width=False
        )

#Make order in chaos here

    pic=st.session_state.selected_pic
    augmented_pics = generate_all_augmented_pics(pic)
    blur_img,median_blur_img,gaussian_blur_img,canny_img,csa_img,gaussian_noise_img,sharpened_img,rotated_img,sp_noise_img,flipped_img = augmented_pics

    st.header("Data Augmentation")
    first_row_left,first_row_middle,first_row_right = st.columns(3)
    second_row_left,second_row_middle,second_row_right = st.columns(3)
    third_row_left,third_row_middle,third_row_right = st.columns(3)
    fourth_row_left,fourth_row_middle,fourth_row_right = st.columns(3)


    with first_row_left:
        st.image(blur_img,width=400)
        l,middle,r = st.columns([0.15,0.5,0.2])
        with middle:
            if st.button("Use the Blur filter"):
                filter_one_numerical("Blur",pic,cv2.blur,blur_options)
    with first_row_middle:
        st.image(median_blur_img,width=400)
        l,middle,r = st.columns([0.12,0.5,0.2])
        with middle:
            if st.button("Use the Median Blur filter"):
                filter_one_numerical("Median_blur",pic,medianBlur,median_blur_options)
    with first_row_right:
        st.image(gaussian_blur_img,width=400)
        l,middle,r = st.columns([0.12,0.5,0.2])
        with middle:
            if st.button("Use the Gaussian Blur filter"):
                filter_two_numerical("Gaussian_Blur",pic,gaussianBlur,gaussian_blur_options)


    with second_row_left: 
        st.image(sharpened_img,width=400)
        l,middle,r = st.columns([0.15,0.5,0.2])
        with middle:
            if st.button("Use the sharpening"):
                two_bool_dialog("sharpening",pic,sharpening,sharpening_options)
    with second_row_middle: 
        st.image(gaussian_noise_img,width=400)
        l,middle,r = st.columns([0.1,0.5,0.2])
        with middle:
            if st.button("Use the Gaussian noise filter"):
                filter_one_numerical_one_bool("Gaussian_noise",pic,gaussian_noise,gaussian_noise_options)
    with second_row_right:
        st.image(sp_noise_img,width=400)
        l,middle,r = st.columns([0.1,0.5,0.2])
        with middle:
            if st.button("Use the Salt and Pepper noise"):
                filter_one_numerical("Salt_pepper_noise",pic,salt_pepper_noise,sp_noise_options)


    with third_row_left: 
        st.image(canny_img,width=400)
        l,middle,r = st.columns([0.1,0.5,0.2])
        with middle:
            if st.button("Use the Edge filter (Canny)"):
                filter_two_numerical("Canny",pic,cv2.Canny,canny_options)

    with third_row_middle:
        st.image(rotated_img,width=400)
        l,middle,r = st.columns([0.2,0.5,0.2])
        with middle:
            if st.button("Rotate Images"):
                rotate_dialog("rotate",pic,rotation,rotation_options)

    with third_row_right:
        st.image(flipped_img,width=400)
        l,middle,r = st.columns([0.23,0.5,0.2])
        with middle:
            if st.button("Flip Images"):
                two_bool_dialog("flip",pic,flip,flip_options)

    with fourth_row_middle:
        st.image(csa_img,width=400)
        l,middle,r = st.columns([0.1,0.5,0.2])
        with middle:
            if st.button("Contrast and Brightness adjusted"):
                filter_two_numerical("CSA_adjustment",pic,cv2.convertScaleAbs,contrast_brightness_options)


st.divider()
#Use on all pictures
downloadable_data=False
if st.button("Augment Data"):
    aX_train,aX_test,ay_train,ay_test = augment_uploaded_data(X_train[:300],X_test[:100],y_train[:300],y_test[:100])
    augmented_data = (aX_train,aX_test,ay_train,ay_test)
    st.session_state.augmented_data = augmented_data

if "augmented_data" in st.session_state:
    augmented_data = st.session_state.augmented_data
    aX_train,aX_test,ay_train,ay_test = augmented_data

    np.savez('data.npz', X_train=aX_train, X_test=aX_test, y_train=ay_train, y_test=ay_test)
    with open('data.npz', 'rb') as a_data:
        st.download_button('Download npz file', a_data, file_name='augmented_data.npz')

if "data" in st.session_state:
    data = st.session_state.data
#Train models and return comparison augmented vs not augmented
st.write("Here we can generate the augmented data and download it")
st.divider()
st.header("The Impact of Data augmentation on Deep Learning")
st.write("in the following we train two models. One on the original dataset and one on the newly created one. Then those models are compared")


if st.button("Train models"):
    #TODO Control this
    '''
    with st.spinner(text="In progress first model..."):
        model_without_augmentation,stats = train_model(data,augmented_data)
        st.session_state.trained_model = model_without_augmentation
        st.session_state.model_stats = stats
    with st.spinner(text="In progress second model..."):
        model_with_augmentation,stats_a = train_model(data,augmented_data,use_augmented_data=True)
        st.session_state.trained_augmented_model = model_with_augmentation
        st.session_state.model_stats_augmented = stats_a
    '''
    #TODO delete
    stats = (0.9,0.4,0.5)
    stats_a = (0.91,0.9,0.6)
    st.session_state.model_stats_augmented = stats_a
    st.session_state.model_stats = stats
    #TODO#####

st.divider()


st.header("Model Comparison")
col1, col2, col3 = st.columns(3)
if "model_stats_augmented" in st.session_state and "model_stats" in st.session_state:
    model1_acc_data,model1_acc_aug_data,model1_auc_data = st.session_state.model_stats
    model2_acc_data,model2_acc_aug_data,model2_auc_data = st.session_state.model_stats_augmented
    

    col1.metric("Accuracy", model2_acc_data, model2_acc_data - model1_acc_data)
    col2.metric("Accuracy on augmented", model2_acc_aug_data, model2_acc_aug_data - model1_acc_aug_data)
    col3.metric("AUC", model2_auc_data,model2_auc_data - model1_auc_data)


left,middle = st.columns([0.05,0.95],vertical_alignment="bottom")
with left:
    if "trained_model" in st.session_state:
        model.save("trained_model.h5")
        with open("trained_model.h5", 'rb') as mod:
            st.download_button("Download model trained on augmented data", data=mod,file="model_augmented.h5")

with middle:
    if "trained_augmented_model" in st.session_state:
        model.save("trained_model_augmented.h5")
        with open("trained_model_augmented.h5",'rb') as aug_mod:
            st.download_button("Download model trained on normal data", data=aug_mod,file="model_not_augmented")