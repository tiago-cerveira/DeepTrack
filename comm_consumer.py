import sys
import zmq


class Consumer:

    def __init__(self, port, topicfilter=None):

        # Socket to talk to server
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)

        self.socket.connect("tcp://localhost:%s" % port)

        self.socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

    def recv_msg(self):
        string = self.socket.recv()
        # split string on first blank space
        topic, messagedata = string.split(' ', 1)
        # print messagedata
        return messagedata

    def recv_msg_no_block(self):
        try:
            string = self.socket.recv(flags=zmq.NOBLOCK)
            topic, messagedata = string.split(' ', 1)
            return messagedata

        except zmq.Again as e:
            return False
