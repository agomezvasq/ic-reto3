import cv2
import pickle
import numpy as np
import os

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

N_CLUSTERS = 350

with open(os.path.dirname(__file__) + '/kmeans_model_' + str(N_CLUSTERS), 'rb') as f:
    kmeans = pickle.load(f)

with open(os.path.dirname(__file__) + '/model_' + str(N_CLUSTERS), 'rb') as f:
    clf, mean, std = pickle.load(f)

def process(file):
    if not os.path.exists(file):
        raise Exception

    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    # print('val/' + dir + '/' + file)
    keypoints, des = sift.detectAndCompute(gray, None)
    # print(des)

    clusters = kmeans.predict(des)
    histogram, _ = np.histogram(clusters, bins=np.arange(N_CLUSTERS + 1))

    x = (histogram - mean) / std

    pred = clf.predict(np.array([x]))

    pred_class = list(labels.keys())[int(pred[0])]

    return labels[pred_class]