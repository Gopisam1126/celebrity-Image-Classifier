import streamlit as st
import cv2
import numpy as np
import pywt
import joblib
import json
import base64
from PIL import Image

@st.cache_resource
def load_model():
    clf = joblib.load('celeb_image_clf_LG.pkl')
    with open('class_dir.json', 'r') as f:
        class_dir = json.load(f)
    inv_class_dir = {v: k for k, v in class_dir.items()}
    return clf, inv_class_dir

clf, inv_class_dir = load_model()

def w2d(img, mode='db1', level=5):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray) / 255.0
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] = np.zeros_like(coeffs_H[0])
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.uint8(imArray_H * 255)
    return imArray_H

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

def get_cropped_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return img[y:y+h, x:x+w]
    return None

def get_prediction(img):
    face = get_cropped_face(img)
    if face is None:
        return None, None
    face_resized = cv2.resize(face, (32, 32))
    face_wt = w2d(face, mode='db1', level=5)
    face_wt_resized = cv2.resize(face_wt, (32, 32))
    combined = np.hstack((face_resized.reshape(-1), face_wt_resized.reshape(-1)))
    X = combined.reshape(1, -1).astype(float)
    prediction = clf.predict(X)[0]
    proba = clf.predict_proba(X).max()
    celeb_name = inv_class_dir[prediction]
    return celeb_name, proba

def img_to_base64(img_file):
    img_bytes = img_file.getvalue()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

st.title('Celebrity Face Recognition')
st.write('Upload an image and the app will predict the celebrity in it.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    b64_str = img_to_base64(uploaded_file)

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img_array = np.array(image.convert('RGB'))[:, :, ::-1]

    celeb_name, proba = get_prediction(img_array)
    if celeb_name:
        st.success(f'Predicted: {celeb_name} (Confidence: {proba:.2f})')
    else:
        st.error('Could not detect a face with two eyes. Please try another image.')