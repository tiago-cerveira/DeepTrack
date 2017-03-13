from __future__ import print_function
import cv2
import numpy as np
import os
import time
import pika
import sys
from threading import Thread



def threaded_function(arg):
    os.system(arg)

class Comm:

    def __init__(self):

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
                host='localhost'))
        self.channel = self.connection.channel()

        self.channel.exchange_declare(exchange='images', type='fanout')

    def end_connection(self):
        self.connection.close()

    def send(self, message):
        self.channel.basic_publish(exchange='images',
                              routing_key='',
                              body=message.tostring())
        # print(" [x] Sent %r" % message)



video_path = '/home/tiago/maritime_data_seq/lanchaArgos_clip3'
groundtruth_file = 'groundtruth_rect_detection.txt'

command = 'python tracker_handler.py '


def main():
    # load ground truth file and images files
    groundtruth = np.loadtxt(video_path + '/' + groundtruth_file, delimiter=" ")
    img_seq = sorted([f for f in os.listdir(video_path + '/img') if os.path.isfile(os.path.join(video_path + '/img', f))])

    comm = Comm()

    # iterate over ground truth file until the first detection is performed
    for i in xrange(len(img_seq)):
        if groundtruth[0][0].astype(np.int) == i:
            print("detection on frame", i)

            # initialize tracker on previous detection
            # TODO: what if there are multiple detections
            bounding_box = ' '.join(map(str, groundtruth[0][1:5].astype(np.int)))
            print(command + bounding_box)
            thread = Thread(target=threaded_function, args=((command + bounding_box), ))
            thread.start()
            break

    start = time.time()

    # "real" program starts here
    for j in xrange(i, len(img_seq)):


        img = cv2.imread(video_path + '/img/' + img_seq[j], 1)
        comm.send(img)
        print(j)

        # display image
        # # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Image", int(img.shape[1]/2), int(img.shape[0]/2))
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)

    comm.end_connection()

    numframes = len(img_seq) - i
    secs = time.time() - start
    print()
    print(numframes)
    print(secs)
    print(round(numframes / secs, 2))

if __name__ == "__main__":
    main()