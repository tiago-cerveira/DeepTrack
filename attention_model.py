import numpy as np
import random


def get_roi(hits, window_sz, img_shape):
    # print("hits", hits)
    # mean, cov = [], []
    decision = random.random()

    if decision < 0.5:
        mean = np.array([hits[0][1] + hits[0][3] / 2, hits[0][2] + hits[0][4] / 2])
        cov = np.array([[hits[0][3] * 10, 0], [0, hits[0][4] * 10]])

        # print "mean", mean
        roi_center = np.random.multivariate_normal(mean, cov, 1).astype(np.int).transpose()

    else:
        roi_center = [random.randint(0, img_shape[0]), random.randint(0, img_shape[1])]
    # print roi_center

    square = [roi_center[0] - window_sz[0]/2, roi_center[1] - window_sz[1]/2, window_sz[0], window_sz[1]]
    return square


def detect(roi, truth_line, detections):
    rst = None
    id = ''
    bb = ''
    # print('detections', detections)
    truth = truth_line.astype(int)


    # there is a detection on the attention box
    if truth[1] > roi[0] and \
        truth[2] > roi[1] and \
        truth[1] + truth[3] < roi[0] + roi[2] and \
        truth[2] + truth[4] < roi[1] + roi[3] and \
            truth_line[6] > 0.90:

        # there is also a tracker on the attention box
        if detections[0][1] > roi[0] and \
            detections[0][2] > roi[1] and \
            detections[0][1] + detections[0][3] < roi[0] + roi[2] and \
                detections[0][2] + detections[0][4] < roi[1] + roi[3]:

            a = np.array((detections[0][1] + detections[0][3]/2, detections[0][2] + detections[0][4]/2))
            b = np.array((truth[1] + truth[3]/2, truth[2] + truth[4]/2))
            dist = np.linalg.norm(a - b)
            print round(dist, 1), round(float(detections[0][3]) / truth[3], 2), round(float(detections[0][4]) / truth[4], 2)

            if dist > 20 or \
                0.75 > float(detections[0][3]) / truth[3] > 1.25 or \
                    0.75 < float(detections[0][4]) / truth[4] > 1.25:

                rst = 'UPDATE'
                id = str(detections[0][0])
                bb = str(truth[1]) + ' ' + str(truth[2]) + ' '  + str(truth[3]) + ' '  + str(truth[4])

        # there is a detection but not a tracker
        else:
            rst = 'INSERT'

    # there is a tracker but not a detection on the box
    elif detections[0][1] > roi[0] and \
        detections[0][2] > roi[1] and \
        detections[0][1] + detections[0][3] < roi[0] + roi[2] and \
        detections[0][2] + detections[0][4] < roi[1] + roi[3]:

        rst = 'DELETE'

    return rst, id, bb