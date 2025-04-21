# Credits : https://www.youtube.com/@codebasics

import numpy as np
import pandas as pd
import cv2
import os
import shutil
import pywt
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import json
import joblib

face_cascade = cv2.CascadeClassifier(".venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(".venv/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml")

def get_cropped_img_if_2_eyes(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = face_img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >=2:
            return roi_color

path_to_data = "./dataset"
path_to_crop_data = "./dataset/cropped/"

img_dirs = []
for e in os.scandir(path_to_data):
    if e.is_dir():
        img_dirs.append(os.path.normpath(e.path))
        
if os.path.exists(path_to_crop_data):
    shutil.rmtree(path_to_crop_data)

os.mkdir(path_to_crop_data)

cropped_img_dirs = []
celeb_file_names = {}

for img_dir in img_dirs:
    count = 1
    celeb_name = os.path.basename(img_dir)
    celeb_file_names[celeb_name] = []
    
    for e in os.scandir(img_dir):
        roi_color = get_cropped_img_if_2_eyes(e.path)
        
        if roi_color is not None:
            cr_dir = path_to_crop_data + celeb_name
            
            if not os.path.exists(cr_dir):
                os.mkdir(cr_dir)
                cropped_img_dirs.append(cr_dir)
                print("Populating with Cropped Images : ", cr_dir)
                
            cr_file_name = celeb_name + str(count) + ".png"
            cr_file_path = cr_dir + "/" + cr_file_name
            
            cv2.imwrite(cr_file_path, roi_color)
            celeb_file_names[celeb_name].append(cr_file_path)
            count += 1
            
def w2d(img, mode="haar", level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    
    return imArray_H

y_dict = {}
count = 0

for celeb_name in celeb_file_names.keys():
    y_dict[celeb_name] = count
    count += 1

X = []
y = []

for celeb_name, t_data in celeb_file_names.items():
    for image_path in t_data:
        img = cv2.imread(image_path)
        if img is None:
            continue
        scaled_img = cv2.resize(img, (32, 32))
        img_wt = w2d(img, 'db1', 5)
        scaled_wt_img = cv2.resize(img_wt, (32, 32))
        stacked_img = np.vstack((scaled_img.reshape(32*32*3, 1), scaled_wt_img.reshape(32*32, 1)))
        X.append(stacked_img)
        y.append(y_dict[celeb_name])
        
X = np.array(X).reshape(len(X), 4096).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipe = Pipeline([('scalar', MinMaxScaler()), ('svc', SVC(kernel='rbf', C=10))])
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))

model_params = {
    'svm': {
        'model': SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [5,10,15]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear'),
        'params': {
            'logisticregression__C': [5,10,15]
        }
    }
}

scores = []
best_estimators = {}

for algo, mp in model_params.items():
    pipe = make_pipeline(MinMaxScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_
    

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)

best_clf = best_estimators['logistic_regression']

with open("class_dir.json", "w") as f:
    f.write(json.dumps(y_dict))

joblib.dump(best_clf, 'celeb_image_clf_LG.pkl')