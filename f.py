import random

import cv2
import os
import numpy as np
import shutil

from sklearn import svm
from sklearn.cluster import KMeans
import pickle

from server.FlaskApp.FlaskApp import process

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
            '26 - Bloque admon': 2,
            '38': 3,
            '19': 1,
            'admisiones': 4,
            'agora': 5,
            'auditorio': 6,
            'biblioteca': 7,
            'Centro de Idiomas': 8,
            'dogger': 9 }

TRAIN_TEST = False

if TRAIN_TEST:
    os.mkdir('train')
    os.mkdir('val')
    os.mkdir('test')

    for dir in os.listdir('data'):
        files = os.listdir('data/' + dir)

        random.shuffle(files)
        cutoff1 = int(len(files) * 0.6)
        cutoff2 = int(len(files) * 0.8)
        train = files[:cutoff1]
        val = files[cutoff1:cutoff2]
        test = files[cutoff2:]

        os.mkdir('train/' + dir)
        os.mkdir('val/' + dir)
        os.mkdir('test/' + dir)

        for file in train:
            shutil.copy('data/' + dir + '/' + file, 'train/' + dir + '/' + file)

        for file in val:
            shutil.copy('data/' + dir + '/' + file, 'val/' + dir + '/' + file)

        for file in test:
            shutil.copy('data/' + dir + '/' + file, 'test/' + dir + '/' + file)

OVERWRITE_SIFT_VECTORS = False

if not os.path.exists('sift_vectors') or OVERWRITE_SIFT_VECTORS:
    os.mkdir('sift_vectors')

    for d in ['train', 'val', 'test']:
        os.mkdir('sift_vectors/' + d)

        for dir in os.listdir(d):
            os.mkdir('sift_vectors/' + d + '/' + dir)

            for file in os.listdir(d + '/' + dir):
                img = cv2.imread(d + '/' + dir + '/' + file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                sift = cv2.xfeatures2d.SIFT_create()
                print(d + '/' + dir + '/' + file)
                keypoints, des = sift.detectAndCompute(gray, None)
                #print(des)
                with open('sift_vectors/' + d + '/' + dir + '/' + os.path.splitext(file)[0] + '.sift', 'wb') as f:
                    pickle.dump(des, f)

                #img_keypoints = img.copy()
                #cv2.drawKeypoints(img, keypoints, img_keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                #cv2.imshow('image', img_keypoints)
                #cv2.waitKey(0)

        print({key: len(value) for key, value in sift_vectors[d].items()})

    with open('sift_vectors', 'wb') as f:
        pickle.dump(sift_vectors, f)
else:
    #with open('sift_vectors', 'rb') as f:
    #    sift_vectors = pickle.load(f)
    r = 0

OVERWRITE_KMEANS_MODEL = False

N_CLUSTERS = 350

if not os.path.exists('kmeans_model_' + str(N_CLUSTERS)) or OVERWRITE_KMEANS_MODEL:
    vectors = []
    for dir in os.listdir('sift_vectors/train'):
        for file in os.listdir('sift_vectors/train/' + dir):
            with open('sift_vectors/train/' + dir + '/' + file, 'rb') as f:
                a = pickle.load(f)
            vectors.extend(list(a))
    # for value in sift_vectors['train'].values():
    #    vectors.extend(value)
    random.shuffle(vectors)
    print(len(vectors))

    X = np.asarray(vectors[:100000])
    print(X.shape)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(X)

    with open('kmeans_model_' + str(N_CLUSTERS), 'wb') as f:
        pickle.dump(kmeans, f)
else:
    with open('kmeans_model_' + str(N_CLUSTERS), 'rb') as f:
        kmeans = pickle.load(f)

print(kmeans)

if not os.path.exists('histograms_' + str(N_CLUSTERS)) or OVERWRITE_KMEANS_MODEL:
    histograms = {}

    for d in ['train', 'val', 'test']:
        histograms[d] = {}

        for dir in os.listdir(d):
            histograms[d][dir] = []

            for file in os.listdir(d + '/' + dir):
                with open('sift_vectors/' + d + '/' + dir + '/' + os.path.splitext(file)[0] + '.sift', 'rb') as f:
                    des = pickle.load(f)
                # print(des)
                clusters = kmeans.predict(des)
                histogram, _ = np.histogram(clusters, bins=np.arange(N_CLUSTERS + 1))
                histograms[d][dir].append(histogram)

                print(d + '/' + dir + '/' + file)

            histograms[d][dir] = np.array(histograms[d][dir])

    with open('histograms_' + str(N_CLUSTERS), 'wb') as f:
        pickle.dump(histograms, f)
else:
    with open('histograms_' + str(N_CLUSTERS), 'rb') as f:
        histograms = pickle.load(f)

print({key: len(value) for key, value in histograms['train'].items()})
print({key: len(value) for key, value in histograms['val'].items()})
print({key: len(value) for key, value in histograms['test'].items()})

n_histograms = sum([len(value) for value in histograms['train'].values()])

# print(np.array([np.array(value) for value in histograms.values()]))
X = np.concatenate(tuple([value for value in histograms['train'].values()]))
y = np.concatenate(
    tuple([np.ones(histograms['train'][key].shape[0]) * classes[key] for key in histograms['train'].keys()]))

mean = np.mean(X)
std = np.std(X)
X = (X - mean) / std
print(mean)
print(std)
print(X)
print(np.mean(X))
print(np.std(X))

OVERWRITE_MODEL = False

if not os.path.exists('model_' + str(N_CLUSTERS)) or OVERWRITE_MODEL:
    clf = svm.SVC(C=10000)
    clf.fit(X, y)
    print(clf)

    with open('model_' + str(N_CLUSTERS), 'wb') as f:
        pickle.dump((clf, mean, std), f)
else:
    with open('model_' + str(N_CLUSTERS), 'rb') as f:
        clf, mean, std = pickle.load(f)

y_pred = clf.predict(X)
print(y_pred)
print(y_pred.shape)

n_val_histograms = sum([len(value) for value in histograms['val'].values()])

val_X = np.concatenate(tuple([value for value in histograms['val'].values()]))
val_y = np.concatenate(tuple([np.ones(histograms['val'][key].shape[0]) * classes[key] for key in histograms['val'].keys()]))

val_X = (val_X - mean) / std

val_y_pred = clf.predict(val_X)

n_test_histograms = sum([len(value) for value in histograms['test'].values()])

test_X = np.concatenate(tuple([value for value in histograms['test'].values()]))
test_y = np.concatenate(tuple([np.ones(histograms['test'][key].shape[0]) * classes[key] for key in histograms['test'].keys()]))

test_X = (test_X - mean) / std

test_y_pred = clf.predict(test_X)

accuracy = np.sum(y == y_pred) / n_histograms
print('Training accuracy: ' + str(accuracy))
val_accuracy = np.sum(val_y == val_y_pred) / n_val_histograms
print('Validation accuracy: ' + str(val_accuracy))
test_accuracy = np.sum(test_y == test_y_pred) / n_test_histograms
#print('Test accuracy: ' + str(test_accuracy))

TEST = True

if TEST:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1280, 720)

    for dir in os.listdir('val'):
        for file in os.listdir('val/' + dir):
            print(process.process('val/' + dir + '/' + file))

            if False:
                img = cv2.imread('val/' + dir + '/' + file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                sift = cv2.xfeatures2d.SIFT_create()
                #print('val/' + dir + '/' + file)
                keypoints, des = sift.detectAndCompute(gray, None)
                # print(des)

                clusters = kmeans.predict(des)
                histogram, _ = np.histogram(clusters, bins=np.arange(N_CLUSTERS + 1))

                x = (histogram - mean) / std

                pred = clf.predict(np.array([x]))

                pred_class = list(labels.keys())[int(pred[0])]
                print(pred_class)
                if pred_class != dir:

                    img_keypoints = img.copy()
                    cv2.drawKeypoints(img, keypoints, img_keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    cv2.imshow('image', img_keypoints)
                    cv2.waitKey(0)
