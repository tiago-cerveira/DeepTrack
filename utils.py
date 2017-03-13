import numpy as np
import os
import sys


def load_video_sequence(groundtruth_file, video_path):
    groundtruth = np.loadtxt(video_path + '/' + groundtruth_file, delimiter=" ")
    img_seq = sorted([f for f in os.listdir(video_path + '/img') if os.path.isfile(os.path.join(video_path + '/img', f))])
    return groundtruth, img_seq


def threaded_function(arg):
    os.system(arg)

def get_bounding_box():
    return sys.argv[1:5]