import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
import gdown
import imutils
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, ClientSettings
import av

load_dotenv()

model_drive_id = os.getenv('MODEL_DRIVE_ID')
cascade_drive_id = os.getenv('CASCADE_DRIVE_ID')

model_path = os.getenv('MODEL_PATH', 'modelv9.keras')
face_cascade_path = os.getenv('FACE_CASCADE_PATH', 'haarcascade_frontalface_default.xml')

if not os.path.exists(model_path):
    gdown.download(f'https://drive.google.com/uc?id={model_drive_id}', model_path, quiet=False)

if not os.path.exists(face_cascade_path):
    gdown.download(f'https://drive.google.com/uc?id={cascade_drive_id}', face_cascade_path, quiet=False)

model = load_model(model_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)

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
    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = imutils.resize(img, width=800)  
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        st.write(f"Faces detected: {len(faces)}")

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = img[y:y+h, x:x+w]
            try:
                preprocessed_face = preprocess_face(face)
                st.write(f"Preprocessed face shape: {preprocessed_face.shape}")

                prediction = model.predict(preprocessed_face)[0]
                predicted_class_index = np.argmax(prediction)
                predicted_label = emotion_labels[predicted_class_index]
                confidence_score = prediction[predicted_class_index]
                st.write(f"Emotion: {predicted_label}, Confidence: {confidence_score*100:.2f}%")

                text = f'{predicted_label} ({confidence_score*100:.2f}%)'
                cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                st.write(f"Error processing face: {e}")

        return av.VideoFrame.from_ndarray(img, format='bgr24')

client_settings = ClientSettings(
    rtc_configuration={
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

webrtc_streamer(
    key="emotion-detector",
    mode=WebRtcMode.SENDRECV,
    client_settings=client_settings,
    video_processor_factory=EmotionDetector,
    async_transform=True
)
