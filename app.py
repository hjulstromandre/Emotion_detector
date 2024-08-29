import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
import gdown
import imutils

# Load environment variables and models
load_dotenv()

model_drive_id = os.getenv('MODEL_DRIVE_ID')
cascade_drive_id = os.getenv('CASCADE_DRIVE_ID')

model_path = os.getenv('MODEL_PATH', 'modelv9.keras')
face_cascade_path = os.getenv('FACE_CASCADE_PATH', 'haarcascade_frontalface_default.xml')

# Download model files if they don't exist
if not os.path.exists(model_path):
    gdown.download(f'https://drive.google.com/uc?id={model_drive_id}', model_path, quiet=False)
    st.text("Model downloaded successfully.")

if not os.path.exists(face_cascade_path):
    gdown.download(f'https://drive.google.com/uc?id={cascade_drive_id}', face_cascade_path, quiet=False)
    st.text("Cascade Classifier downloaded successfully.")

# Load the TensorFlow model and Cascade Classifier
try:
    model = load_model(model_path)
    st.text("Model loaded successfully.")
except Exception as e:
    st.text(f"Error loading model: {e}")

try:
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    st.text("Cascade Classifier loaded successfully.")
except Exception as e:
    st.text(f"Error loading Cascade Classifier: {e}")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Streamlit app title
st.title("Real-Time Emotion Detector")

# Function to preprocess the face before feeding it to the model
def preprocess_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    roi = roi_gray.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    return roi

# Main function for the Streamlit app
def main():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)  # Access the webcam

    if not cap.isOpened():
        st.text("Failed to access webcam.")
        return

    st.text("Webcam accessed successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.text("Failed to capture video.")
            break
        
        st.text("Frame captured successfully.")
        
        frame = imutils.resize(frame, width=800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        st.text(f"Faces detected: {len(faces)}")

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            try:
                preprocessed_face = preprocess_face(face)
                prediction = model.predict(preprocessed_face)[0]
                predicted_class_index = np.argmax(prediction)
                predicted_label = emotion_labels[predicted_class_index]
                confidence_score = prediction[predicted_class_index]
                text = f'{predicted_label} ({confidence_score*100:.2f}%)'
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                st.text(f"Emotion detected: {predicted_label}, Confidence: {confidence_score*100:.2f}%")
            except Exception as e:
                st.text(f"Error processing face: {e}")

        stframe.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
