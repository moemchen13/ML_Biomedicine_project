import streamlit as st


st.set_page_config(page_title="Hello", layout="wide",page_icon="🏡")
st.title("Main Page - Navigation")

col1, col2, col3 = st.columns(3)
with col2:
    st.header("Our Team")
    st.image("./data/pics/jonathan_moritz.jpg")
st.write("Team:")
st.markdown("<p>Moritz Christ<br> Master Machine Learning (3. Semester)<br>Jonathan Nemitz<br>Master Machine Learning (5. Semester)</p>",unsafe_allow_html=True)



st.write("<h4>Conclusions</h4>")
st.markdown("<ul><li>We never will be webdesigners</li> <li>streamlit is nice for demos</li> <li>streamlit is buggy</li> <li>Look up known issues with the cloud (cv2)</li></ul>",unsafe_allow_html=True)




st.write("<h4>Impressum</h4>")
st.markdown("Eberhard Karls Universität Tübingen <br> Project in the framework of the ML for Biomedicine internship<br><br> <h5>Special thanks to:</h5> <ol><li>Elisa Maske</li><li>Nintendo</li><li>Shirin Davis</li> <ol>",unsafe_allow_html=True)


