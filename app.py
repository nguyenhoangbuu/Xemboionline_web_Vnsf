import streamlit as st
import time
from mtcnn import MTCNN
import keras
import cv2
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
import pandas as pd
# from tempfile import NamedTemporaryFile
from joblib import dump, load
from PIL import  Image,ImageOps
# import pickle
from ultralytics import YOLO



@st.cache_resource(show_spinner="Mạnh mẽ tìm hiểu bản thân. Hãy tin tưởng vào khả năng của bạn. Trong thời gian một cây nhang, thầy bói sẽ được triệu hồi.")
def load_model_keras():
    model = keras.saving.load_model('model.keras')
    #model = 1
    model_detector = MTCNN()
    # model_detector = YOLO('yolov8n.pt')
    return model, model_detector

@st.cache_resource(show_spinner=False)
def load_enc():
    enc = load('encoder.joblib')
    return enc

def stream_tho(tho):
    for word in tho.split(" "):
        word_write = f"<p style='font-size:17px;font-family:Comic Sans MS;font-style:italic;'>{word}</p>"
        if st.write(word_write, unsafe_allow_html=True):
            yield st.write(word_write, unsafe_allow_html=True)
        time.sleep(0.75)

@st.cache_data(show_spinner=False)
def load_layer():
    col1, col2, col3 = st.columns([0.25,0.5,0.25])
    st.write('''<style>
            [data-testid="column"] {
                width: calc(70% - 1rem) !important;
                min-width: calc(10% - 1rem) !important;
                text-align: center;
            }
            </style>''', unsafe_allow_html=True)
    with col2:
        title = "Tướng Tự Tâm"
        title_write = f"<p style='font-size:23px;font-family:Helvetica;font-style:italic;'>{title}</p>"
        st.write(title_write, unsafe_allow_html=True)
        st.image(r'Data_web/boi1.jpg',use_column_width = True)
    st.write('''<style>
                [data-testid="column"] {
                    width: calc(30%  - 1rem) !important;
                    min-width: calc(10% - 1rem) !important;
                    text-align: center;
                }
                </style>''', unsafe_allow_html=True)
    with col1:
        tho1 = "Ngoài vòng cương toả chân cao thấp"
        # s1 = f"<p style='font-size:35px;font-family:Comic Sans MS;font-style:italic;'>{label1}</p>"
        st.write_stream(stream_tho(tho1))
        # st.markdown(s1, unsafe_allow_html=True)
        # st.text("Ngoài \n vòng \n cương \n toả \n chân \n cao \n thấp")
    with col3:
        tho2 = "Trong thú yên hà mặt tỉnh say"
        # s2 = f"<p style='font-size:35px;font-family:Comic Sans MS;font-style:italic;'>{label2}</p>"
        st.write_stream(stream_tho(tho2))


df = pd.read_csv('df.csv')
load_layer()
ans = {}
model, model_detector = load_model_keras()
enc = load_enc()
with st.form("Choice"):
    options = st.multiselect(
    'Lựa chọn điều bạn muốn biết',
    ['Tuổi khuôn mặt', 'Độ nhạy cảm', 'Số người thích thầm bạn', 'Đô bia', 'Đô rượu','Màu sắc thầm kín'])
    submitted = st.form_submit_button("Submit")
    #if submitted and len(options)>0:
        #st.write("Tải lên ảnh của bạn:")
if len(options)>0:
    file = st.file_uploader("Tải lên ảnh có mặt bạn: ", type=["jpg", "png", "jpeg"])
    max_coeff = 0
    if not (file is None):
        image0 = Image.open(file)
        image0 = ImageOps.exif_transpose(image0)
        #image = ImageOps.fit(image0, (224, 224), Image.LANCZOS)
        img = np.asarray(image0).astype(np.uint8)
        #model = load_model_keras()
        with st.spinner("Đang bói"):
            results = model_detector.detect_faces(img)
        max_coeff = 0
        if len(results) > 0:
            index = 0
            for i in range(len(results)):
                if results[i]['confidence'] > max_coeff:
                    max_coeff = results[i]['confidence']
                    index = i
            xyxy = np.array(results[index]['box'])
            xmin, ymin, width, height = xyxy[0],xyxy[1],xyxy[2],xyxy[3]
            xmax, ymax = xmin + width, ymin + height
            xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(img.shape[1], xmax), min(
                img.shape[0], ymax)
            img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),3)
            face = cv2.cvtColor(img[ymin:ymax, xmin:xmax], cv2.COLOR_RGB2BGR)
            cv2.imwrite(r'Static/1.png', face)
            face_load = load_img(r'Static/1.png', target_size=(56, 56))
            face_load = img_to_array(face_load)
            face_load = face_load / 255
            face_load = np.expand_dims(face_load, axis=0)
            pred = enc.inverse_transform(model.predict(face_load))[0][0]
            #st.write(pred)
            ans['Luong_bia'] = str(df[df['Ten']==pred].Luong_bia.values[0]) + ' ' + 'chai'
            ans['Luong_ruou'] = str(df[df['Ten']==pred].Luong_ruou.values[0]) + ' ' + 'xị'
            ans['sai_bia'] = str(df[df['Ten']==pred].sai_bia.values[0]) + ' ' + 'chai'
            ans['sai_ruou'] = str(df[df['Ten']==pred].sai_ruou.values[0]) + ' ' + 'xị'
            ans['so_nguoi_thich_tham'] = str(df[df['Ten']==pred].so_nguoi_thich_tham.values[0]) + ' ' + 'người'
            ans['do_nhay_cam'] = str(df[df['Ten']==pred].do_nhay_cam.values[0])
            ans['tuoi_khuon_mat'] = str(df[df['Ten']==pred].tuoi_khuon_mat.values[0]) + ' ' + 'tuổi'
            ans['mau_sac'] = str(df[df['Ten']==pred].mau_sac.values[0])
            ans['luan_bia_ruou'] = str(df[df['Ten'] == pred].luan_bia_ruou.values[0])
            ans['luan_nguoi_thich'] = str(df[df['Ten'] == pred].luan_nguoi_thich.values[0])
            ans['luan_do_nhay_cam'] = str(df[df['Ten'] == pred].luan_nhay_cam.values[0])
            ans['luan_tuoi'] = str(df[df['Ten'] == pred].luan_tuoi_tac.values[0])
            ans['luan_mau_sac'] = str(df[df['Ten'] == pred].luan_mau_sac.values[0])
        else:
            text = 'Không phát hiện khuôn mặt'
            st.text(text)
        placeholder = st.empty()
        # placeholder.image(image0, use_column_width=True)
        # time.sleep(1)
        # placeholder.empty()
        st.image(img, use_column_width=True)
tabs = []
if len(options)>0:
    tabs = st.tabs(options)

if len(options)>0 and not (file is None) and max_coeff>0:
    for i in range(len(options)):
        with tabs[i]:
            if options[i] == 'Đô bia':
                st.metric("", ans['Luong_bia'], ans['sai_bia'])
                st.write(ans['luan_bia_ruou'])
            elif options[i] == 'Đô rượu':
                st.metric("", ans['Luong_ruou'], ans['sai_ruou'])
                st.write(ans['luan_bia_ruou'])
            elif options[i] == 'Tuổi khuôn mặt':
                st.metric("", ans['tuoi_khuon_mat'], "-1 tuổi")
                st.write(ans['luan_tuoi'])
            elif options[i] == 'Độ nhạy cảm':
                st.metric("", ans['do_nhay_cam'], "+1")
                st.write(ans['luan_do_nhay_cam'])
            elif options[i] == 'Số người thích thầm bạn':
                st.metric("", ans['so_nguoi_thich_tham'], "+2 người")
                st.write(ans['luan_nguoi_thich'])
            elif options[i] == 'Màu sắc thầm kín':
                st.metric("", ans['mau_sac'], " ")
                st.write(ans['luan_mau_sac'])