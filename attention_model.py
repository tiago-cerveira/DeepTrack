import cv2
import numpy as np
import random


class AttentionModel:
    def __init__(self, init_img):
        self.decision_threshold = 0.5
        self.dist_threshold = 20
        self.min_scale_threshold = 0.7
        self.max_scale_threshold = 1.3

        self.feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.old_gray = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)

    def get_optical_flow(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        # Select good points
        try:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                # frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            # img = cv2.add(frame,mask)
            # cv2.imshow('frame',img)
            # cv2.waitKey(100)
            print(a - c, b - d)
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
        except:
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            print("no optical flow")

    def get_roi(self, hits, window_sz, img):
        # print("hits", hits)
        # mean, cov = [], []
        self.get_optical_flow(img)
        decision = random.random()

        if decision < self.decision_threshold:
            mean = np.array([hits[0][1] + hits[0][3] / 2, hits[0][2] + hits[0][4] / 2])
            cov = np.array([[hits[0][3] * 10, 0], [0, hits[0][4] * 10]])

            # print "mean", mean
            roi_center = np.random.multivariate_normal(mean, cov, 1).astype(np.int).transpose()

        else:
            roi_center = [random.randint(0 + window_sz[1]/2, img.shape[1] - window_sz[1]/2), random.randint(0 + window_sz[0]/2, img.shape[0] - window_sz[0]/2)]
        # print roi_center

        square = [roi_center[0] - window_sz[0]/2, roi_center[1] - window_sz[1]/2, window_sz[0], window_sz[1]]
        return square

    def detect(self, roi, truth_line, detections):
        rst = None
        id = ''
        bb = ''
        # print('detections', detections)
        truth = truth_line.astype(int)

        # there is a detection on the attention box with high confidence
        if truth[1] > roi[0] and \
            truth[2] > roi[1] and \
            truth[1] + truth[3] < roi[0] + roi[2] and \
            truth[2] + truth[4] < roi[1] + roi[3] and \
                truth_line[6] > 0.90:

            # there is also a tracker on the attention box
            for detection in detections:
                if detection[1] > roi[0] and \
                    detection[2] > roi[1] and \
                    detection[1] + detection[3] < roi[0] + roi[2] and \
                        detection[2] + detection[4] < roi[1] + roi[3]:

                    a = np.array((detection[1] + detection[3]/2, detection[2] + detection[4]/2))
                    b = np.array((truth[1] + truth[3]/2, truth[2] + truth[4]/2))
                    dist = np.linalg.norm(a - b)
                    print round(dist, 1), round(float(detection[3]) / truth[3], 2), round(float(detection[4]) / truth[4], 2)

                    if dist > self.dist_threshold or \
                        float(detection[3]) / truth[3] < self.min_scale_threshold or \
                        float(detection[3]) / truth[3] > self.max_scale_threshold or \
                        float(detection[4]) / truth[4] < self.min_scale_threshold or \
                        float(detection[4]) / truth[4] > self.max_scale_threshold:

                        rst = 'UPDATE'
                        id = str(detection[0])
                        bb = str(truth[1]) + ' ' + str(truth[2]) + ' ' + str(truth[3]) + ' ' + str(truth[4])
                    return rst, id, bb

            # there is a detection but not a tracker
            # rst = 'INSERT'
            # bb = str(truth[1]) + ' ' + str(truth[2]) + ' ' + str(truth[3]) + ' ' + str(truth[4])

        # there is a tracker but not a detection on the box
        elif detections[0][1] > roi[0] and \
            detections[0][2] > roi[1] and \
            detections[0][1] + detections[0][3] < roi[0] + roi[2] and \
                detections[0][2] + detections[0][4] < roi[1] + roi[3]:

            rst = 'DELETE'

        return rst, id, bb
