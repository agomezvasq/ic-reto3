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
lst = ['18', '19', '26 - Bloque admon', '38', 'admisiones', 'agora', 'auditorio', 'biblioteca', 'Centro de Idiomas', 'dogger']

N_CLUSTERS = 350

with open(os.path.dirname(__file__) + '/kmeans_model_' + str(N_CLUSTERS), 'rb') as f:
    kmeans = pickle.load(f)

with open(os.path.dirname(__file__) + '/model_' + str(N_CLUSTERS), 'rb') as f:
    clf, mean, std = pickle.load(f)

import datetime

def log(s):
    #with open('/var/www/FlaskApp/FlaskApp/log.log', 'a') as f:
    #    f.write(str(datetime.datetime.now()) + ': ' + s + '\n')
    print(s)

def process(file):
    if not os.path.exists(file):
        log('File not found: ' + file)
        raise FileNotFoundError



    log('Processing started')
    img = cv2.imread(file)
    log('Read img')

    if img.shape != (720, 1280, 3):
        log('Shape is ' + str(img.shape) + '. Resizing')
        img = cv2.resize(img, (1280, 720))
        log('Resized')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log('Converted to gray')

    sift = cv2.xfeatures2d.SIFT_create()
    # print('val/' + dir + '/' + file)
    keypoints, des = sift.detectAndCompute(gray, None)
    # print(des)
    log('Computed sifts')

    clusters = kmeans.predict(des)
    histogram, _ = np.histogram(clusters, bins=np.arange(N_CLUSTERS + 1))
    log('Calculated histogram')

    x = (histogram - mean) / std
    log('Normalized histogram')

    pred = clf.predict(np.array([x]))

    pred_class = lst[int(pred[0])]

    return labels[pred_class]
