import numpy as np


def get_roi(hits, window_sz, img_shape):
    # print("hits", hits)
    # mean, cov = [], []
    # for i in xrange(len(hits)):
    #     mean.append(np.array([hits[i][0] + hits[i][2] / 2, hits[i][1] + hits[i][3] / 2]))
    #     # cov.append(np.array([[1000 * hits[i][2], 0], [0, 1000 * hits[i][3]]]))
    #     cov.append(np.array([[hits[i][2], 0], [0, hits[i][3]]]))
    mean = np.array([hits[0][1] + hits[0][3] / 2, hits[0][2] + hits[0][4] / 2])
    cov = np.array([[hits[0][3] * 10, 0], [0, hits[0][4] * 10]])

    # print "mean", mean
    roi_center = np.random.multivariate_normal(mean, cov, 1).astype(np.int).transpose()
    # print roi_center

    square = [roi_center[0] - window_sz[0]/2, roi_center[1] - window_sz[1]/2, window_sz[0], window_sz[1]]
    return square


def detect(roi, truth, detections):
    rst = None
    command = ''

    truth = truth.astype(int)

    if truth[1] > roi[0] and \
        truth[2] > roi[1]  and \
        truth[1] + truth[3] < roi[0] + roi[2] and \
        truth[2] + truth[4] < roi[1] + roi[3]:


        if detections[0][1] > roi[0] and \
            detections[0][2] > roi[1] and \
            detections[0][1] + detections[0][3] < roi[0] + roi[2] and \
            detections[0][2] + detections[0][4] < roi[1] + roi[3]:

            rst = 'UPDATE'
            command = '0 ' + str(truth[1]) + ' ' + str(truth[2]) + ' '  + str(truth[3]) + ' '  + str(truth[4])

        else:
            rst = 'INSERT'

    elif detections[0][1] > roi[0] and \
        detections[0][2] > roi[1] and \
        detections[0][1] + detections[0][3] < roi[0] + roi[2] and \
        detections[0][2] + detections[0][4] < roi[1] + roi[3]:

        rst = 'DELETE'

    return rst, command