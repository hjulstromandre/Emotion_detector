import asyncio
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
import gdown
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import logging
import av

logging.basicConfig(level=logging.INFO)

load_dotenv()

@st.cache_resource
def load_models():
    model_drive_id = os.getenv('MODEL_DRIVE_ID')
    cascade_drive_id = os.getenv('CASCADE_DRIVE_ID')

    model_path = os.getenv('MODEL_PATH', 'modelv9.keras')
    face_cascade_path = os.getenv('FACE_CASCADE_PATH', 'haarcascade_frontalface_default.xml')

    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?id={model_drive_id}', model_path, quiet=False)
        logging.info("Model downloaded successfully.")

    if not os.path.exists(face_cascade_path):
        gdown.download(f'https://drive.google.com/uc?id={cascade_drive_id}', face_cascade_path, quiet=False)
        logging.info("Cascade Classifier downloaded successfully.")

    model = load_model(model_path)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    return model, face_cascade

model, face_cascade = load_models()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("Real-Time Emotion Detector")

def preprocess_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    roi = roi_gray.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    return roi

class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_frame_with_face = 0
        self.face_linger_time = 15

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        if self.frame_count % 3 != 0:
            if self.frame_count - self.last_frame_with_face < self.face_linger_time:
                if hasattr(self, 'last_coords') and hasattr(self, 'last_text'):
                    cv2.rectangle(img, self.last_coords[0], self.last_coords[1], (255, 0, 0), 2)
                    cv2.putText(img, self.last_text, (self.last_coords[0][0], self.last_coords[0][1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_coords = ((x, y), (x+w, y+h))
                face = img[y:y+h, x:x+w]
                preprocessed_face = preprocess_face(face)
                prediction = model.predict(preprocessed_face)[0]
                predicted_class_index = np.argmax(prediction)
                predicted_label = emotion_labels[predicted_class_index]
                confidence_score = prediction[predicted_class_index]
                text = f'{predicted_label} ({confidence_score*100:.2f}%)'

                self.last_frame_with_face = self.frame_count
                self.last_coords = face_coords
                self.last_text = text

                cv2.rectangle(img, face_coords[0], face_coords[1], (255, 0, 0), 2)
                cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="emotion-detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionDetector,
    rtc_configuration={
        "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
