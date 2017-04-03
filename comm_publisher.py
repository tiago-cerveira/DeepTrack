import zmq
import random
import sys
import time


class Publisher:

    def __init__(self, port, bind=True):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        if bind:
            self.socket.bind("tcp://127.0.0.101:%s" % port)
        else:
            self.socket.connect("tcp://127.0.0.101:%s" % port)

    def send(self, topic, message_data):

        # print "sent : %s %s" % (topic, message_data)
        self.socket.send("%s %s" % (topic, message_data))

    def close(self):
        self.socket.close()
        self.context.term()


def main():
    pub = Publisher(5000, bind=False)

    topic = 'VIP'

    for i in xrange(50):
        pub.send(topic, str(i))
        time.sleep(1)
    pub.close()


if __name__ == "__main__":
    main()
