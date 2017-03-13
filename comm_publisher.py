import zmq
import random
import sys
import time


class Publisher:

    def __init__(self):
        port = "5557"

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://*:%s" % port)

    def send(self, topic, message_data):

        # print "sent : %s %s" % (topic, message_data)
        self.socket.send("%s %s" % (topic, message_data))
