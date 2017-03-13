from __future__ import print_function
import cv2
import sys
import pika
from time import sleep
import numpy as np
from comm_consumer import Consumer
from comm_publisher import Publisher
import zmq
from utils import *
from tracker import Tracker


class InitParams:

    def __init__(self):
        self.frame = 0


def main():

    params = InitParams()

    bounding_box = get_bounding_box()
    print("i'm the tracker and received:", bounding_box)

    sub1 = Consumer('next_img')
    sub2 = Consumer('kill')

    while True:

        img_index = int(sub1.recv_msg())

        if sub2.recv_msg_no_block():
            print("received kill command")
            break

        # Initialize the tracker using the first frame
        if params.frame == 0:
            tracker1 = Tracker(im, parameters)
            tracker1.train(im, True)
            results[frame, :] = np.array(
                (pos[0] + np.floor(target_sz[0] / 2), pos[1] + np.floor(target_sz[1] / 2), target_sz[0], target_sz[1]))
        else:
            results[frame, :], lost, xtf = tracker1.detect(im)  # Detect the target in the next frame
            if not lost:
                tracker1.train(im, False, xtf)  # Update the model with the new infomation
        if parameters.visualization:
            # Draw a rectangle in the estimated location and show the result
            cvrect = np.array((results[frame, 1] - results[frame, 3] / 2,
                               results[frame, 0] - results[frame, 2] / 2,
                               results[frame, 1] + results[frame, 3] / 2,
                               results[frame, 0] + results[frame, 2] / 2))
            cv2.rectangle(im, (cvrect[0].astype(int), cvrect[1].astype(int)),
                          (cvrect[2].astype(int), cvrect[3].astype(int)), (0, 255, 0), 2)
            cv2.imshow('Window', im)
            cv2.waitKey(1)
            print(frame, end='\t')


if __name__ == "__main__":
    main()