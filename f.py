import random

import cv2
import os
import numpy as np
import shutil

from sklearn import svm
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
classes = { '18': 0,
            '19': 1,
            '26 - Bloque admon': 2,
            '38': 3,
            'admisiones': 4,
            'agora': 5,
            'auditorio': 6,
            'biblioteca': 7,
            'Centro de Idiomas': 8,
            'dogger': 9 }

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
        sift_vectors[dir] = []

        for file in os.listdir('train/' + dir):
            img = cv2.imread('train/' + dir + '/' + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, des = sift.detectAndCompute(gray, None)
            #print(des)
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

N_CLUSTERS = 200

vectors = []
for value in sift_vectors.values():
    vectors.extend(value)
random.shuffle(vectors)

if not os.path.exists('kmeans_model') or OVERWRITE_KMEANS_MODEL:
    X = np.asarray(vectors[:100000])
    print(X.shape)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(X)

    with open('kmeans_model', 'wb') as f:
        pickle.dump(kmeans, f)
else:
    with open('kmeans_model', 'rb') as f:
        kmeans = pickle.load(f)

print(kmeans)

if not os.path.exists('histograms') or OVERWRITE_KMEANS_MODEL:
    histograms = {}

    for dir in os.listdir('train'):
        histograms[dir] = []

        for file in os.listdir('train/' + dir):
            img = cv2.imread('train/' + dir + '/' + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, des = sift.detectAndCompute(gray, None)
            # print(des)
            clusters = kmeans.predict(des)
            histogram, _ = np.histogram(clusters, bins=np.arange(N_CLUSTERS + 1))
            histograms[dir].append(histogram)

            img_keypoints = img.copy()
            # cv2.drawKeypoints(img, keypoints, img_keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            print('train/' + dir + '/' + file)

            # cv2.imshow('image', img_keypoints)
            # cv2.waitKey(0)

        histograms[dir] = np.array(histograms[dir])

    with open('histograms', 'wb') as f:
        pickle.dump(histograms, f)
else:
    with open('histograms', 'rb') as f:
        histograms = pickle.load(f)

print({key: len(value) for key, value in histograms.items()})

n_histograms = sum([len(value) for value in histograms.values()])

#print(np.array([np.array(value) for value in histograms.values()]))
X = np.concatenate(tuple([value for value in histograms.values()]))
y = np.concatenate(tuple([np.ones(histograms[key].shape[0]) * classes[key] for key in histograms.keys()]))

mean = np.mean(X)
std = np.std(X)
X = (X - mean) / std
print(mean)
print(std)
print(X)
print(np.mean(X))
print(np.std(X))

clf = svm.SVC(C=2.0)
clf.fit(X, y)
print(clf)

y_pred = clf.predict(X)
print(y_pred)
print(y_pred.shape)

accuracy = np.sum(y == y_pred) / n_histograms
print(accuracy)