import cv2
import mtcnn
import numpy as np
import os
import pickle
from architecture import *
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model

from sklearn.preprocessing import Normalizer

import pytz
from datetime import datetime

timestamp = str(int(datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).timestamp()))

l2_normalizer = Normalizer('l2')

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

CONFIDENCE_THRESH = 0.99
RECOGNITION_DISTANCE_THRESH = 0.5
REQUIRED_SIZE = (160, 160)

def extract_face(img, save_path='.output/extracted'):
    detector = mtcnn.MTCNN()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    i = 1
    for result in results:
        print('result => ', result)
        if result['confidence'] < CONFIDENCE_THRESH:
            continue
        # Extract faces
        box = result['box']
        x1, y1, width, height = box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]
        # Save faces
        file_path = f'{save_path}/face-{timestamp}_{i}.jpg'
        cv2.imwrite(file_path, face)
        print(f'Face {i} saved at path: {file_path}')
        i += 1

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    print('path => ', path)
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(
    img,
    model_path: str='models/weights/encodings.pkl',
    facenet_weight_path: str='models/pretrained/facenet_keras_weights.h5'
):
    if not os.path.exists(facenet_weight_path):
        raise Exception(f'Facenet weight path not exists: {facenet_weight_path}')

    if not os.path.exists(model_path):
        raise Exception(f'Model path not exists: {model_path}')

    detector = mtcnn.MTCNN()
    encoder = InceptionResNetV2()
    encoder.load_weights(facenet_weight_path)
    encodings = load_pickle(model_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    result = {
        'found': False,
        'distance': None,
        'confidence': 0,
        'box': None,
        'keypoints': None,
        'data': None,
    }
    for res in results:
        print('res => ', res)
        if res['confidence'] < CONFIDENCE_THRESH:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, REQUIRED_SIZE)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        print('distance => ', distance)
        for db_name, db_encode in encodings.items():
            # The smaller the better
            _distance = cosine(db_encode, encode)
            print(f'{db_name} => {_distance}')
            if _distance <= RECOGNITION_DISTANCE_THRESH and _distance < distance:
                name = db_name
                distance = _distance
                result.update({
                    'found': True,
                    'distance': distance,
                    'confidence': res['confidence'],
                    'keypoints': res['keypoints'],
                    'box': res['box'],
                    'data': {
                        'label': name,
                        'name': str(name).split('(')[0].replace('-', ' ').strip(),
                        'idcardno': str(name).split('-').pop().replace('(', '').replace(')', '')
                    },
                })

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)

    return result, img

storage_dir = '.output'
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir, exist_ok=True)

if __name__ == "__main__":
    model_path = 'runs/1705658453/encodings-1705647048-1705658453.pkl'
    # image = cv2.imread('datasets/tests/Jackie Chan.jpg')
    image = cv2.imread('datasets/tests/face-1704696193.6065_duong_thi_cam_1.jpg')
    result, labeled_image = detect(image, model_path=model_path)
    save_path = f'{storage_dir}/out-{timestamp}.jpg'
    cv2.imwrite(save_path, labeled_image)
    print(f'Result saved at path: {save_path}')
    print('Result => ', result)

    # extract_face(cv2.imread('datasets/tests/4b4fb1d3-2fbe-40bd-aafd-16d36826da78_001199005250_IDFront.png'))
