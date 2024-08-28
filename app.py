import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv


load_dotenv()

model_path = os.getenv('MODEL_PATH')
face_cascade_path = os.getenv('FACE_CASCADE_PATH')

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

def main():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            preprocessed_face = preprocess_face(face)
            prediction = model.predict(preprocessed_face)[0]
            predicted_class_index = np.argmax(prediction)
            predicted_label = emotion_labels[predicted_class_index]
            confidence_score = prediction[predicted_class_index]
            text = f'{predicted_label} ({confidence_score*100:.2f}%)'
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        stframe.image(frame, channels="BGR")
    cap.release()
if st.button('Start Webcam'):
    main()
