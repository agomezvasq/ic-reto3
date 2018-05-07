import random

import cv2
import os
import numpy as np
import shutil
from sklearn.cluster import KMeans
import pickle

labels = { '18': 'Bloque 18',
           '19': 'Bloque 19',
           '26 - Bloque admon': 'Bloque 26',
           '38': 'Bloque 38',
           'admisiones': 'Bloque 29',
           'agora': 'El Ágora',
           'auditorio': 'Auditorio Fundadores',
           'biblioteca': 'Biblioteca',
           'Centro de Idiomas': 'Centro de Idiomas',
           'dogger': 'Dogger' }

TRAIN_TEST = False

if TRAIN_TEST:
    for dir in os.listdir('data'):
        files = os.listdir('data/' + dir)

        random.shuffle(files)
        cutoff = int(len(files) * 0.7)
        train = files[:cutoff]
        test = files[cutoff:]

        os.mkdir('train/' + dir)
        os.mkdir('test/' + dir)

        for file in train:
            shutil.copy('data/' + dir + '/' + file, 'train/' + dir + '/' + file)

        for file in test:
            shutil.copy('data/' + dir + '/' + file, 'test/' + dir + '/' + file)

OVERWRITE_SIFT_VECTORS = False

if not os.path.exists('sift_vectors') or OVERWRITE_SIFT_VECTORS:
    sift_vectors = {}

    for dir in os.listdir('train'):
        for file in os.listdir('train/' + dir):
            img = cv2.imread('train/' + dir + '/' + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, des = sift.detectAndCompute(gray, None)
            #print(des)
            if not dir in sift_vectors:
                sift_vectors[dir] = list(des)
            else:
                sift_vectors[dir].extend(list(des))

            img_keypoints = img.copy()
            #cv2.drawKeypoints(img, keypoints, img_keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            print('train/' + dir + '/' + file)

            #cv2.imshow('image', img_keypoints)
            #cv2.waitKey(0)

    print({key: len(value) for key, value in sift_vectors.items()})

    with open('sift_vectors', 'wb') as f:
        pickle.dump(sift_vectors, f)
else:
    with open('sift_vectors', 'rb') as f:
        sift_vectors = pickle.load(f)

OVERWRITE_KMEANS_MODEL = False

if not os.path.exists('kmeans_model') or OVERWRITE_KMEANS_MODEL:
    vectors = []
    for value in sift_vectors.values():
        vectors.extend(value)
    random.shuffle(vectors)
    X = np.asarray(vectors[:100000])
    print(X.shape)

    kmeans = KMeans(n_clusters=200, random_state=0).fit(X)

    with open('kmeans_model', 'wb') as f:
        pickle.dump(kmeans, f)
else:
    with open('kmeans_model', 'rb') as f:
        kmeans = pickle.load(f)

print(kmeans)