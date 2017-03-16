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

        self.window_sz = np.array([300, 300])


def main():
    params = InitParams()

    pub = Publisher('5551')
    sub1 = Consumer('5557', 'bb')
    sub2 = Consumer('5557', 'alive')

    # load ground truth file and images files
    groundtruth, img_seq = load_video_sequence(params.groundtruth_file, params.video_path)

    detections = []

    # iterate over every frame of the sequence selected
    for img_index in xrange(len(img_seq)):
        time.sleep(0.5)
        img = cv2.imread(params.video_path + '/img/' + img_seq[img_index], 1)

        # full detection mode
        if params.num_trackers == 0:
            if groundtruth[params.num_detections][0].astype(np.int) == img_index:
                print("detection on frame", img_index)

                # initialize tracker on previous detection
                # TODO: what if there are multiple detections
                bounding_box = ' '.join(map(str, groundtruth[params.num_detections][1:5].astype(np.int)))
                detections = [0]
                detections.extend(groundtruth[params.num_detections][1:5].astype(np.int))
                detections = [detections]
                # print(detections)
                # print(params.command + bounding_box)

                thread = Thread(target=threaded_function, args=((params.command + bounding_box + ' ' + str(params.num_trackers)), ))
                thread.start()

                if sub2.recv_msg() == str(params.num_trackers):
                    # print("sending index:", img_index)
                    params.num_detections += 1
                    params.num_trackers += 1
                    pub.send('next_img', str(img_index))
                continue

        # using attention model
        else:
            print("sending frame to trackers:", img_index)
            pub.send('next_img', str(img_index))

            # in this case the ground truth file contains a detection for the current frame
            if groundtruth[params.num_detections][0].astype(np.int) == img_index:
                params.num_detections += 1

                # perform detection on patch
                roi = attention_model.get_roi(detections, params.window_sz, img.shape)
                # print(roi)
                rst, command = attention_model.detect(roi, groundtruth[params.num_detections], detections)

                if rst == 'UPDATE' and img_index > 40:
                    # print(command)
                    pub.send('update', command)

                elif rst == 'INSERT':
                    pass
                elif rst == 'DELETE':
                    pass
                else:
                    pass


            else:
                print("No detection for frame:", img_index)

        detections = []
        for i in xrange(params.num_trackers):
            aux = sub1.recv_msg()
            entry = [int(aux[0])]
            aux = aux[3:-1]
            entry.extend([int(s) for s in aux.split(', ')])
            detections.append(entry)
            print("received detection:", detections)


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

        if img_index == 70:
            pub.send('kill', '0')
            params.num_trackers -= 1

    pub.send('kill', 'True')


if __name__ == "__main__":
    main()
