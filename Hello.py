import streamlit as st


st.set_page_config(page_title="Hello", layout="wide",page_icon="🏡")
st.title("Main Page - Navigation")

col1, col2, col3 = st.columns(3)
with col2:
    st.header("Our Team")
    st.image("./data/pics/i_me.jpg")
st.write("Team:")
st.markdown("<p>Moritz Christ<br> Master Machine Learning (3. Semester)<br>Jonathan Nemitz<br>Master Machine Learning (5. Semester)</p>",unsafe_allow_html=True)


st.write("Impressum")
st.markdown("Eberhard Karls Universität Tübingen <br> Project in the framework of the ML for Biomedicine internship<br><br> Special thanks to Elisa Maske,Shirin Davis and the Wise guys",unsafe_html_allow=True)


