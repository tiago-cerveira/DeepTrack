from __future__ import print_function
import cv2
import numpy as np
import random
import time


class ROI:
    # max uncertainty (1 - boat, 0 - no boat)
    uncertainty = 0.5
    slope = 0.005
    detected, undetected = False, False

    def __init__(self, center, window_sz, top=False, bottom=False, left=False, right=False):
        self.center = center
        self.window_sz = window_sz
        # print(self.center)
        # TODO: Don't need to know the location of the cell
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def update_uncertainty(self, new_uncertainty):
        if self.detected:
            self.uncertainty = 1
            self.detected = False
        elif self.undetected:
            self.uncertainty = 0
            self.undetected = False
        else:
            self.uncertainty = new_uncertainty
            self.uncertainty -= self.func(self.slope)

    # propagates uncertainty from one frame to the other
    def func(self, slope):
        return 2 * slope * (self.uncertainty - 0.5)

    # has a maximum response when uncertainty is higher and minimum when it is lower
    def arg(self):
        return -4*(self.uncertainty-0.5)**2 + 1


class AttentionModel:
    decision_threshold = 0.0
    dist_threshold = 20
    min_scale_threshold = 0.7
    max_scale_threshold = 1.3
    window_sz = np.array([300, 300])
    rois = []
    flow = None

    def __init__(self, init_img):
        n_cols = np.floor(float(init_img.shape[1])/self.window_sz[1]).astype(int)
        n_rows = np.floor(float(init_img.shape[0])/self.window_sz[0]).astype(int)

        self.create_rois(n_cols, n_rows)

        self.roi_selected = None

        self.old_gray = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)

    def create_rois(self, n_cols, n_rows):
        for i in xrange(n_cols):
            for j in xrange(n_rows):
                # print(i, j)
                x = int(i * self.window_sz[0] + self.window_sz[0] / 2)
                y = int(j * self.window_sz[1] + self.window_sz[1] / 2)
                if i == 0 and j == 0:
                    # print('top left', end=' ')
                    self.rois.append(ROI((x, y), self.window_sz, top=True, left=True))
                elif i == n_cols - 1 and j == n_rows - 1:
                    # print('bottom right', end=' ')
                    self.rois.append(ROI((x, y), self.window_sz, bottom=True, right=True))
                elif i == 0 and j == n_rows - 1:
                    # print('bottom left', end=' ')
                    self.rois.append(ROI((x, y), self.window_sz, bottom=True, left=True))
                elif i == n_cols - 1 and j == 0:
                    # print('top right', end=' ')
                    self.rois.append(ROI((x, y), self.window_sz, top=True, right=True))
                elif i == 0 and j != 0:
                    # print('left', end=' ')
                    self.rois.append(ROI((x, y), self.window_sz, left=True))
                elif i != 0 and j == 0:
                    # print('top', end=' ')
                    self.rois.append(ROI((x, y), self.window_sz, top=True))
                elif i == n_cols - 1 and j != n_rows - 1:
                    # print('right', end=' ')
                    self.rois.append(ROI((x, y), self.window_sz, right=True))
                elif i != n_cols - 1 and j == n_rows - 1:
                    # print('bottom', end=' ')
                    self.rois.append(ROI((x, y), self.window_sz, bottom=True))
                else:
                    self.rois.append(ROI((x, y), self.window_sz))
                    # time.sleep(2)

    def compute_optical_flow(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.flow = cv2.calcOpticalFlowFarneback(self.old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        max_val = np.max(self.flow)
        print(max_val)

        self.old_gray = frame_gray.copy()

    @staticmethod
    def area(rect_a, rect_b):
        dx = min(rect_a.center[0] + rect_a.window_sz[0] / 2, rect_b.center[0] + rect_b.window_sz[0] / 2) \
             - max(rect_a.center[0] - rect_a.window_sz[0] / 2, rect_b.center[0] - rect_b.window_sz[0] / 2)

        dy = min(rect_a.center[1] + rect_a.window_sz[1] / 2, rect_b.center[1] + rect_b.window_sz[1] / 2) \
             - max(rect_a.center[1] - rect_a.window_sz[1] / 2, rect_b.center[1] - rect_b.window_sz[1] / 2)

        if dx > 0 and dy > 0:
            return dx * dy

    @staticmethod
    def normalize_prop(prop):
        # too much info, need to normalize
        if prop[1] > 1:
            prop[0] /= prop[1]
        # too little propagation, uncertainty arises
        elif prop[1] < 1:
            prop[0] = prop[0] * prop[1] + 0.5 * (1 - prop[1])
        return prop

    def prop_uncertainty(self):
        for roi in self.rois:
            # how much the optical flow says the window moved relative to the previous frame
            # mov_center = flow[roi.center[1], roi.center[0], :] # just for the centers
            mov_center = self.flow[roi.center[1] - self.window_sz[1] / 2:roi.center[1] + self.window_sz[1] / 2,
                         roi.center[0] - self.window_sz[0] / 2:roi.center[0] + self.window_sz[0] / 2, :]
            mov_center = mov_center.mean(axis=(0, 1), dtype=np.float64)
            mov_center = np.flip(mov_center, 0)

            roi_mov = ROI(roi.center + mov_center, self.window_sz)
            roi_mov.uncertainty = roi.uncertainty
            for i, static_roi in enumerate(self.rois):
                area_abs = self.area(static_roi, roi_mov)
                if area_abs is not None:
                    area_rel = area_abs / np.prod(roi.window_sz)
                    # TODO: Careful if should sum or subtract
                    static_roi.uncertainty += area_rel * roi_mov.uncertainty
                    static_roi.voverage += area_rel

        for i, roi in enumerate(self.rois):
            new_uncertainty = self.normalize_prop((roi.uncertainty, roi.coverage))
            roi.update_uncertainty(roi.uncertainty)

    def get_roi(self, hits, img):

        start = time.time()
        self.compute_optical_flow(img)
        print("took", round(time.time() - start, 2), "seconds to compute optical flow")
        decision = random.random()

        # adjust known boat
        # TODO: make it work for multiple boats
        if decision < self.decision_threshold:
            mean = np.array([hits[0][1] + hits[0][3] / 2, hits[0][2] + hits[0][4] / 2])
            cov = np.array([[hits[0][3] * 10, 0], [0, hits[0][4] * 10]])

            roi_center = np.random.multivariate_normal(mean, cov, 1).astype(np.int).transpose()

        # look for new boats
        else:
            baseline_uncertainty = self.rois[0].arg()
            self.roi_selected = 0
            roi_center = np.flip(self.rois[0].center, 0)
            for i, roi in enumerate(self.rois):
                # print(roi.uncertainty)
                if roi.arg() >= baseline_uncertainty:
                    roi_center = np.flip(roi.center, 0)
                    self.roi_selected = i
                    # print(self.roi_selected)
                    # roi_center = [random.randint(0 + self.window_sz[1]/2, img.shape[1] - self.window_sz[1]/2), random

        return [roi_center[0] - self.window_sz[0]/2, roi_center[1] - self.window_sz[1]/2, self.window_sz[0], self.window_sz[1]]

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

            for detection in detections:
                # there is also a tracker on the attention box
                if detection[1] > roi[0] and \
                                detection[2] > roi[1] and \
                                        detection[1] + detection[3] < roi[0] + roi[2] and \
                                        detection[2] + detection[4] < roi[1] + roi[3]:
                    if self.roi_selected is not None:
                        self.rois[self.roi_selected].detected = True

                    a = np.array((detection[1] + detection[3] / 2, detection[2] + detection[4] / 2))
                    b = np.array((truth[1] + truth[3] / 2, truth[2] + truth[4] / 2))
                    dist = np.linalg.norm(a - b)
                    print(round(dist, 1), round(float(detection[3]) / truth[3], 2),
                          round(float(detection[4]) / truth[4], 2))

                    if dist > self.dist_threshold or \
                                            float(detection[3]) / truth[3] < self.min_scale_threshold or \
                                            float(detection[3]) / truth[3] > self.max_scale_threshold or \
                                            float(detection[4]) / truth[4] < self.min_scale_threshold or \
                                            float(detection[4]) / truth[4] > self.max_scale_threshold:
                        rst = 'UPDATE'
                        id = str(detection[0])
                        bb = str(truth[1]) + ' ' + str(truth[2]) + ' ' + str(truth[3]) + ' ' + str(truth[4])
                        # return rst, id, bb



                        # there is a detection but not a tracker
                        # rst = 'INSERT'
                        # bb = str(truth[1]) + ' ' + str(truth[2]) + ' ' + str(truth[3]) + ' ' + str(truth[4])

        # there is a tracker but not a detection on the box
        elif detections[0][1] > roi[0] and \
                        detections[0][2] > roi[1] and \
                                detections[0][1] + detections[0][3] < roi[0] + roi[2] and \
                                detections[0][2] + detections[0][4] < roi[1] + roi[3]:

            rst = 'DELETE'

        # nothing on detection box
        else:
            if self.roi_selected is not None:
                self.rois[self.roi_selected].undetected = True

        self.prop_uncertainty()
        self.roi_selected = None

        return rst, id, bb
