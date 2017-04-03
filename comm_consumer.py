import sys
import zmq


class Consumer:

    def __init__(self, port, topicfilter=None, bind=False):

        # Socket to talk to server
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        if bind:
            self.socket.bind("tcp://127.0.0.101:%s" % port)
        else:
            self.socket.connect("tcp://127.0.0.101:%s" % port)

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

    def close(self):
        self.socket.close()
        self.context.term()


def main():
    sub = Consumer(5000, 'VIP', bind=True)

    while True:
        msg = sub.recv_msg()
        print msg


if __name__ == "__main__":
    main()
