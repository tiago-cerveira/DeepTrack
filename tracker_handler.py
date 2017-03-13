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
        self.padding = 1.5                              # extra area surrounding the target
        self.output_sigma_factor = 0.1                  # standard deviation for the desired translation filter output
        self.lmbda = 1e-4                               # regularization weight
        self.learning_rate = 0.02

        self.scale_sigma_factor = 1.0/4                 # standard deviation for the desired scale filter output
        self.number_of_scales = 1                       # number of scale levels
        self.scale_step = 1.02                          # Scale increment factor
        self.scale_model_max_area = 512                 # maximum scale

        self.features = "CNN_TF"
        self.cell_size = 4.0
        self.high_freq_threshold = 2 * 10 ** 66
        self.peak_to_sidelobe_ratio_threshold = 6       # Set to 0 to disable (Detect if the target is lost)
        self.rigid_transformation_estimation = False    # Try to detect camera rotation

        self.visualization = True
        self.debug = False

        self.init_pos = np.array((0, 0))
        self.pos = np.array((0, 0))
        self.target_size = np.array((0, 0))
        self.img_files = None
        self.video_path = None

        self.kernel = Kernel()

        self.frame = 0


# Structure with kernel parameters
class Kernel:
    def __init__(self):
        self.kernel_type = "Linear"  # Or Gaussian
        self.kernel_sigma = 0.5


def main():

    params = InitParams()

    video_path = get_video_path()
    img_seq = load_video_sequence(groundtruth_file=None, video_path=video_path)
    bounding_box = get_bounding_box()
    # print("i'm the tracker and received:", bounding_box)

    # Initial position
    pos = np.array([int(bounding_box[1]), int(bounding_box[0])])
    target_sz = np.array([int(bounding_box[3]), int(bounding_box[2])])
    params.init_pos = np.floor(pos + np.floor(target_sz / 2))

    # Current position
    params.pos = params.init_pos

    # Size of target
    params.target_size = np.floor(target_sz)
    params.img_files = img_seq

    # List of image files
    params.video_path = video_path

    results = np.zeros((len(img_seq), 4))

    sub1 = Consumer('next_img')
    sub2 = Consumer('kill')

    print("Ready to consume messages")
    while True:

        img_index = int(sub1.recv_msg())
        # print(img_index)
        # print(video_path + '/img/' + img_seq[img_index])
        img = cv2.imread(video_path + '/img/' + img_seq[img_index], 1)

        if sub2.recv_msg_no_block():
            print("received kill command, now exiting")
            break

        # Initialize the tracker using the first frame
        if params.frame == 0:
            tracker1 = Tracker(img, params)
            tracker1.train(img, True)
            results[params.frame, :] = np.array(
                (pos[0] + np.floor(target_sz[0] / 2), pos[1] + np.floor(target_sz[1] / 2), target_sz[0], target_sz[1]))
        else:
            results[params.frame, :], lost, xtf = tracker1.detect(img)  # Detect the target in the next frame
            if not lost:
                tracker1.train(img, False, xtf)  # Update the model with the new infomation

        if params.visualization:
            # Draw a rectangle in the estimated location and show the result
            cvrect = np.array((results[params.frame, 1] - results[params.frame, 3] / 2,
                               results[params.frame, 0] - results[params.frame, 2] / 2,
                               results[params.frame, 1] + results[params.frame, 3] / 2,
                               results[params.frame, 0] + results[params.frame, 2] / 2))
            cv2.rectangle(img, (cvrect[0].astype(int), cvrect[1].astype(int)),
                          (cvrect[2].astype(int), cvrect[3].astype(int)), (0, 255, 0), 2)
            cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Window", int(img.shape[1] / 2), int(img.shape[0] / 2))
            cv2.imshow('Window', img)
            cv2.waitKey(1)

        params.frame += 1

    np.savetxt('results.txt', results, delimiter=',', fmt='%d')



if __name__ == "__main__":
    main()