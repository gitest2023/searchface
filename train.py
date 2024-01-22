import cv2
import mtcnn
import numpy as np
import os
import pickle
import pytz
import shutil
from datetime import datetime
from sklearn.preprocessing import Normalizer

from architecture import *


def replace_chars(value: str='', replaces: list=[]) -> str:
    if not isinstance(replaces, list):
        return value
    for r in replaces:
        if isinstance(r, tuple) and len(r) > 1:
            value = value.replace(str(r[0]), str(r[1]))
    return value

def load_pickle(pickle_model_path: str):
    if not os.path.exists(pickle_model_path):
        raise Exception(f'Model path not exists: {pickle_model_path}')
    with open(pickle_model_path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def encodings(
    train_path: str,
    face_detector=None,
    face_encoder=None,
    l2_normalizer=None,
    required_shape= (160, 60),
    out_dir=''
):
    encodes = []
    encoding_dict = dict()
    face_dirs = os.listdir(train_path)
    total = len(face_dirs)
    count = 1
    for face_name in face_dirs:
        person_dir = os.path.join(train_path, face_name)
        face_name = replace_chars(face_name, replaces=[('\n', ''), ('\t', '')])
        print('face_name => ', face_name)
        try:
            i = 1
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)

                img_BGR = cv2.imread(image_path)
                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

                x = face_detector.detect_faces(img_RGB)
                print('box => ', x[0])
                x1, y1, width, height = x[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1+width, y1+height
                face = img_RGB[y1:y2, x1:x2]

                if out_dir != '':
                    if not os.path.exists(f'{out_dir}/faces/{face_name}'):
                        os.makedirs(f'{out_dir}/faces/{face_name}', exist_ok=True)
                    cv2.imwrite(f'{out_dir}/faces/{face_name}/face{i}.jpg', face)

                face = normalize(face)
                face = cv2.resize(face, required_shape)
                face_d = np.expand_dims(face, axis=0)
                encode = face_encoder.predict(face_d)[0]
                encodes.append(encode)
                print(f'{face_name} => face {i} encoded')
                i += 1

            if encodes:
                encode = np.sum(encodes, axis=0)
                encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
                encoding_dict[face_name] = encode
                print(f'Face ({face_name}) {count}/{total}: {face_name} encoded')
                count += 1
        except Exception as e:
            print('error => ', e)
            continue
    return encoding_dict

def train(
    train_path: str,
    facenet_weight_path: str=None,
    model_path: str=None,
    save_dir: str='runs'
):
    if facenet_weight_path == None or facenet_weight_path == '':
        facenet_weight_path = 'models/pretrained/facenet_keras_weights.h5'

    face_encoder = InceptionResNetV2()
    face_encoder.load_weights(facenet_weight_path)
    face_detector = mtcnn.MTCNN()
    l2_normalizer = Normalizer('l2')

    # Create save directory
    timestamp = str(int(datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).timestamp()))
    out_dir = f'{save_dir}/{timestamp}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    encoding_dict = encodings(
        train_path=train_path,
        face_detector=face_detector,
        face_encoder=face_encoder,
        l2_normalizer=l2_normalizer,
        required_shape=(160, 160),
        out_dir=out_dir
    )

    # Create encodings (.*pkl file)
    # timestamp = str(int(datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).timestamp()))
    # out_dir = f'{save_dir}/{timestamp}'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir, exist_ok=True)

    filename = f'encodings-{timestamp}.pkl'
    if isinstance(model_path, str) and os.path.exists(model_path):
        print(f'Model path existed: {model_path}')
        shutil.copy(model_path, out_dir)
        filename = model_path.split('/').pop()
        name = filename.split(".")[0]
        ext = filename.split(".")[1]
        filename = f'{name}-{timestamp}.{ext}'

    model_path = f'{out_dir}/{filename}'
    if not os.path.isfile(model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(dict, file)
            print('New pickle file created at:', model_path)

    with open(model_path, 'wb') as file:
        pickle.dump(encoding_dict, file)
        print(f'New model created at: {model_path}')
        print('Training successful')

if __name__ == '__main__':
    train_path = 'datasets/new/train5'
    facenet_weight_path = 'models/pretrained/facenet_keras_weights.h5'
    model_path = 'runs/1705648864/encodings-1705647048.pkl'

    train(
        train_path=train_path,
        facenet_weight_path=facenet_weight_path,
        model_path=model_path
    )
