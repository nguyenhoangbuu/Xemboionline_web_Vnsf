import streamlit as st
from tensorflow.keras.utils import load_img

import streamlit as st
import time

with st.empty():
    for seconds in range(15):
        st.write(f"⏳ {seconds} seconds have passed")
        time.sleep(1)
    st.write("✔️ 1 minute over!")

with st.form("Choice"):
    options = st.multiselect(
    'Lựa chọn điều bạn muốn biết',
    [ 'Tình duyên', 'Sự nghiệp', 'Lượng bia', 'Lượng rượu'])
    submitted = st.form_submit_button("Submit")
    st.write(options)
st.write(len(options))

