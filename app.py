import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
import gdown
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = img[y:y+h, x:x+w]
            preprocessed_face = preprocess_face(face)
            prediction = model.predict(preprocessed_face)[0]
            predicted_class_index = np.argmax(prediction)
            predicted_label = emotion_labels[predicted_class_index]
            confidence_score = prediction[predicted_class_index]
            text = f'{predicted_label} ({confidence_score*100:.2f}%)'
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return img

webrtc_streamer(
    key="emotion-detector",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False}
)
