import numpy as np


def get_roi(img, hits):
    centers = []
    for i in xrange(len(hits)):
        centers.append(np.array([hits[i][0] + hits[i][2]/2, hits[i][1] + hits[i][3]/2]))
