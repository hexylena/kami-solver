import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import matplotlib
import scipy.spatial.distance as distance
import scipy.ndimage

font = cv2.FONT_HERSHEY_SIMPLEX
np.set_printoptions(linewidth=220)
debug = True


def griddect(img):
    gray = cv2.cvtColor(img,  cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

    lines = cv2.HoughLines(edges, 1, np.pi/90, 120)
    v = []
    h = []
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        if int(a) == 0:
            h.append(y1)
        else:
            v.append(x1)

        if debug:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    height, width, channels = img.shape
    DEC = 0

    def dist(numArr):
        for i in range(len(numArr) - 1):
            x = numArr[i + 1] - numArr[i]
            if x > 5:
                yield x

    v_points = np.unique(np.round(np.array(sorted(v)), decimals=DEC))
    v_dist = list(dist(np.sort(v_points)))
    v_point_median = np.median(v_dist)

    h_points = np.unique(np.round(np.array(sorted(h)), decimals=DEC))
    h_dist = list(dist(np.sort(h_points)))
    h_point_median = np.median(h_dist)

    return v_point_median, h_point_median


def get_inside_boxes(image, v, h):
    height, width, channels = img.shape
    data = []
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for i in range(0, int(height - h), int(h)):
        row = []
        for j in range(0, int(width - v), int(v)):
            x = int(i + v/2)
            y = int(j + h/2)
            row.append(hsv_img[x][y])
        data.append(row)
    data = np.round(np.array(data), decimals=-1)
    labels, numLabels = scipy.ndimage.label(data)

    return data


def gauss(img):
    kernel = np.ones((5, 5), np.float32)/25
    return cv2.filter2D(img, -1, kernel)

def binData(data, bins=4):
    w = data.shape[0]
    h = data.shape[1]
    image = data.reshape((w * h, 3))
    clt = KMeans(n_clusters=5)
    fit = clt.fit_predict(image)
    # quant = clt.cluster_centers_.astype("uint8")[fit]
    # return quant.reshape((w, h, 3))
    return fit.reshape((w, h))


img = cv2.imread(sys.argv[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
(v, h) = griddect(img)

color_data = get_inside_boxes(gauss(img), v, h)
bin_color_data = binData(color_data, bins=int(sys.argv[2]))

colors = [
    (255, 0, 255),
    (255, 255, 0),
    (0, 0, 255),
    (0, 255, 0),
    (0, 255, 255),
]

if debug:
    for i in range(bin_color_data.shape[0]):
        for j in range(bin_color_data.shape[1]):
            x = i * v + v / 2
            y = j * h + h / 2
            z = bin_color_data[i][j]

            cv2.circle(img, (int(y), int(x)), 4, colors[z], -1)


    plt.imshow(img)
    plt.show()
