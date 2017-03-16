from __future__ import print_function
import cv2
import numpy as np
import os
import time
import sys
from threading import Thread
from comm_publisher import Publisher
from comm_consumer import Consumer
from utils import *
import attention_model


class InitParams:
    def __init__(self):
        self.video_path = '/home/tiago/maritime_data_seq/lanchaArgos_clip3'
        self.groundtruth_file = 'groundtruth_rect_detection.txt'

        self.command = 'python tracker_handler.py ' + self.video_path + ' '

        self.num_trackers = 0
        self.num_detections = 0
        self.tracker_counter = 0

        self.display = True

        self.window_sz = np.array([300, 300])


def main():
    params = InitParams()

    pub = Publisher('5551')
    sub1 = Consumer('5552', 'bb')
    sub2 = Consumer('5552', 'alive')

    # load ground truth file and images files
    groundtruth, img_seq = load_video_sequence(params.groundtruth_file, params.video_path)

    detections = []

    # iterate over every frame of the sequence selected
    for img_index in xrange(len(img_seq)):
        # if img_index > 30:
        #     time.sleep(2)
        img = cv2.imread(params.video_path + '/img/' + img_seq[img_index], 1)

        # full detection mode
        if params.num_trackers == 0:
            if groundtruth[params.num_detections][0].astype(np.int) == img_index:
                if groundtruth[params.num_detections][6] > 0.90:
                    print("detection on frame", img_index)

                    # initialize tracker on previous detection
                    # TODO: what if there are multiple detections
                    bounding_box = ' '.join(map(str, groundtruth[params.num_detections][1:5].astype(np.int)))
                    detections = [0]
                    detections.extend(groundtruth[params.num_detections][1:5].astype(np.int))
                    detections = [detections]
                    # print(detections)
                    # print(params.command + bounding_box)

                    # TODO: Put in function
                    thread = Thread(target=threaded_function, args=((params.command + bounding_box + ' ' + str(params.tracker_counter)), ))
                    thread.start()

                    if sub2.recv_msg() == str(params.tracker_counter):
                        # print("sending index:", img_index)
                        params.num_detections += 1
                        params.num_trackers += 1
                        params.tracker_counter += 1
                        pub.send('next_img', str(img_index))

                params.num_detections += 1
                continue

        # using attention model
        else:
            roi = [0, 0, 0, 0]
            print("sending frame to trackers:", img_index)
            pub.send('next_img', str(img_index))

            # in this case the ground truth file contains a detection for the current frame
            if groundtruth[params.num_detections][0].astype(np.int) == img_index:
                params.num_detections += 1

                # perform detection on patch
                roi = attention_model.get_roi(detections, params.window_sz, img.shape)
                # print(roi)
                rst, id, bb = attention_model.detect(roi, groundtruth[params.num_detections], detections)

                if rst == 'UPDATE':
                    # print("tracker counter:", params.tracker_counter)
                    pub.send('update', id)
                    thread = Thread(target=threaded_function, args=((params.command + bb + ' ' + str(params.tracker_counter)),))
                    thread.start()

                    if sub2.recv_msg() == str(params.tracker_counter):
                        # print("sending index:", img_index)
                        params.num_detections += 1
                        params.tracker_counter += 1
                        pub.send('next_img', str(img_index))

                elif rst == 'INSERT':
                    pass
                elif rst == 'DELETE':
                    pass
                else:
                    pass

            # no detection for the whole frame
            else:
                print("No detection for frame:", img_index)

        detections = []
        for i in xrange(params.num_trackers):
            aux = sub1.recv_msg()
            print(aux)
            aux = aux.split(' (', 1)
            entry = [int(aux[0])]
            aux = aux[1][: -1]
            # aux = aux[: -1]
            entry.extend([int(s) for s in aux.split(', ')])
            detections.append(entry)
            # print("received detection:", detections)

        if params.display:
            # display image
            for i in xrange(params.num_trackers):
                cv2.rectangle(img, (detections[i][1], detections[i][2]),
                              (detections[i][1] + detections[i][3], detections[i][2] + detections[i][4]), (0, 255, 0), 2)
            try:
                top_left = (roi[0], roi[1])
                bottom_right = (roi[0] + roi[2], roi[1] + roi[3])
                # print("top bottom:", top_left, bottom_right)
                cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
            except:
                pass
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", int(img.shape[1]/2), int(img.shape[0]/2))
            cv2.imshow("Image", img)
            cv2.waitKey(1)
        if len(detections) > 0:
            if detections[0][1] + detections[0][3]/2 < 0 or detections[0][2] + detections[0][4]/2 < 0:
                print(detections[0][0])
                pub.send('kill', str(detections[0][0]))
                params.num_trackers -= 1

    pub.send('kill', 'all')


if __name__ == "__main__":
    main()
