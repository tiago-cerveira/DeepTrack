import sys
import zmq


class Consumer:

    def __init__(self, topicfilter=None):

        port = "5557"

        # Socket to talk to server
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)

        self.socket.connect("tcp://localhost:%s" % port)

        self.socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

    def recv_msg(self):
        string = self.socket.recv()
        topic, messagedata = string.split()
        return messagedata

    def recv_msg_no_block(self):
        try:
            string = self.socket.recv(flags=zmq.NOBLOCK)
            # topic, messagedata = string.split()
            return True

        except zmq.Again as e:
            pass
