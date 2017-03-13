from __future__ import print_function
import cv2
import sys
import pika
from time import sleep
import numpy as np

def get_bounding_box():
    return sys.argv[1:5]

class Comm():

    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()

        self.channel.exchange_declare(exchange='images', type='fanout')

        result = self.channel.queue_declare(exclusive=True)
        self.queue_name = result.method.queue

        self.channel.queue_bind(exchange='images', queue=self.queue_name)

        print(' [*] Waiting for images.')

    def callback(self, ch, method, properties, body):
        img = np.fromstring(body, dtype=np.uint8).reshape(1080, 1920, -1)
        # display image
        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Image", int(img.shape[1]/2), int(img.shape[0]/2))
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)
        return

    def start_consuming(self):
        self.channel.basic_consume(self.callback,
                              queue=self.queue_name,
                              no_ack=True)

        self.channel.start_consuming()


def main():

    bounding_box = get_bounding_box()
    print("i'm the tracker and received:", bounding_box)

    comm = Comm()
    comm.start_consuming()

if __name__ == "__main__":
    main()