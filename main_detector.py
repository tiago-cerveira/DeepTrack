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

        self.display = True


def main():
    params = InitParams()

    pub = Publisher('5557')
    sub1 = Consumer('5558', 'bb')
    sub2 = Consumer('5558', 'alive')

    # load ground truth file and images files
    groundtruth, img_seq = load_video_sequence(params.groundtruth_file, params.video_path)

    detections = []

    # iterate over every frame of the sequence selected
    for img_index in xrange(len(img_seq)):


        img = cv2.imread(params.video_path + '/img/' + img_seq[img_index], 1)

        # full detection mode
        if params.num_trackers == 0:
            if groundtruth[params.num_detections][0].astype(np.int) == img_index:
                print("detection on frame", img_index)

                # initialize tracker on previous detection
                # TODO: what if there are multiple detections
                bounding_box = ' '.join(map(str, groundtruth[params.num_detections][1:5].astype(np.int)))
                detections.append(groundtruth[params.num_detections][1:5].astype(np.int))
                # print(detections)
                print(params.command + bounding_box)

                thread = Thread(target=threaded_function, args=((params.command + bounding_box), ))
                thread.start()
                params.num_detections += 1
                params.num_trackers += 1
                sub2.recv_msg()
                print("sending index:", img_index)
                pub.send('next_img', str(img_index))
                continue

        # using attention model
        else:
            print("sending frame to trackers:", img_index)
            pub.send('next_img', str(img_index))

            # in this case the ground truth file contains a detection for the current frame
            if groundtruth[params.num_detections][0].astype(np.int) == img_index:
                params.num_detections += 1

                # TODO: Perform detection on patch
                roi = attention_model.get_roi(img, detections)

            else:
                print("No detection for frame:", img_index)

        detections = []
        for i in xrange(params.num_trackers):
            aux = sub1.recv_msg()
            aux = aux[1:-1]
            detections.append([int(s) for s in aux.split()])
            print(detections)


        if params.display:
            # display image
            for i in xrange(params.num_trackers):
                cv2.rectangle(img, (detections[i][0], detections[i][1]),
                              (detections[i][2], detections[i][3]), (0, 255, 0), 2)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", int(img.shape[1]/2), int(img.shape[0]/2))
            cv2.imshow("Image", img)
            cv2.waitKey(1)

        if img_index == 70:
            pub.send('kill', 'True')
            params.num_trackers -= 1


if __name__ == "__main__":
    main()
