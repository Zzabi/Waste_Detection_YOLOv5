import os
import av
import cv2
import torch
import shutil
import pathlib
import streamlit as st
from ffmpy import FFmpeg
from streamlit_webrtc import webrtc_streamer

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

MODEL_PATH = r"yolov5"
MODEL_FILE_PATH = r"models\eq_dry_wet.pt"
IMG_SAVE_PATH = r"temp\result"
TEMP_VID_SAVE_PATH = r"temp\temp.avi"
VID_SAVE_PATH = r"temp\result.mp4"
TEMP_PATH = r"temp"
FFMPEG_EXE_PATH = r"C:\Users\mypc\ffmpeg\bin\ffmpeg.exe"

class_colors = {
    'Cardboard': (0, 255, 0),   # Green
    'Plastic': (255, 0, 0),     # Blue
    'Paper': (0, 0, 255),       # Red
    'Metal': (0, 255, 255),     # Yellow
    'Glass': (255, 255, 0),     # Cyan
    'Thermocol': (128, 0, 128)  # Purple
}


class VideoProcessor():
    def __init__(self):
        self.model = load_model()

    def recv(self,frame):
        frame = frame.to_ndarray(format="bgr24")
        height,width = frame.shape[0:2]
        frame = cv2.resize(frame, (width,height))
        results = self.model(frame)
        for index, row in results.pandas().xyxy[0].iterrows():
            x1, y1, x2, y2, confidence, class_id, class_name = row
            color = class_colors.get(class_name, (0, 0, 255))
            cv2.rectangle(frame, (int(x1), int(y1)),
                            (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (int(
                x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")


def delete_temp(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def convert_compat():
    delete_temp(VID_SAVE_PATH)
    ff = FFmpeg(
        executable=FFMPEG_EXE_PATH,
        inputs={TEMP_VID_SAVE_PATH: None},
        outputs={VID_SAVE_PATH: '-c:v libx264'}
    )
    temp = ff.run()
    return temp


def load_model():
    model = torch.hub.load(
        MODEL_PATH, "custom",
        path=MODEL_FILE_PATH,
        source="local",
        force_reload=True
    )
    return model


def predict_image(location):
    model = load_model()
    results = model(location, size=640)
    delete_temp(IMG_SAVE_PATH)
    results.save(save_dir=IMG_SAVE_PATH)
    return results


def predict_video(location,change=False):
    stframe = st.empty()
    cap = cv2.VideoCapture(location)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    out = cv2.VideoWriter(TEMP_VID_SAVE_PATH,
                          cv2.VideoWriter_fourcc(*'XVID'), 15, size)
    model = load_model()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if change:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        results = model(frame)
        for index, row in results.pandas().xyxy[0].iterrows():
            x1, y1, x2, y2, confidence, class_id, class_name = row
            color = class_colors.get(class_name, (0, 0, 255))
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (int(
                x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        stframe.image(frame)
        out.write(frame)
    cap.release()
    out.release()
    stframe.empty()
    temp = convert_compat()
    return temp


st.title("Waste Object Detection")

tab1, tab2 = st.tabs(["Upload", "Webcam"])


with tab1:
    st.subheader("Please upload a file for detecting waste")
    uploaded_file = st.file_uploader(
        "Choose a file", ['jpg', 'png', 'webp', 'mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        file_name, file_ext = uploaded_file.name.split('.')
        temp_filename = TEMP_PATH+f"\\temp.{file_ext}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getvalue())
        with st.spinner("Running model (this may take a while)"):
            if file_ext in ['jpg', 'png', 'webp','jpeg']:
                prediction = predict_image(temp_filename)
                st.image(IMG_SAVE_PATH+f"\\temp.jpg")
            else:
                prediction = predict_video(temp_filename)
                st.video(VID_SAVE_PATH)
                delete_temp(TEMP_VID_SAVE_PATH)
        delete_temp(temp_filename)


with tab2:
    st.subheader(
        "Please click on start (make sure to allow permission for camera)")
    webrtc_streamer(key="example",video_processor_factory=VideoProcessor)

    # for laptops without webcam use ip webcam app
    # model = load_model()
    # start = st.toggle("Start")
    # if start:
    #     predict_video("http://192.0.0.4:8080/video",change=True)
