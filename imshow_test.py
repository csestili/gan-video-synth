from example import *
import numpy as np
import cv2

gvs = GanVideoSynth()
num_frames = 25

ims = None

z0 = np.random.normal(size=(1, 128))


def sample(z0):
    zn = np.random.normal(size=(1, 128))
    zs = np.zeros((num_frames, 128))
    for i in range(num_frames):
        zs[i] = z0 * (num_frames - i)/num_frames + zn * i / num_frames

    ys = np.zeros((num_frames, 1000))
    ys[:, 0] = 1
    return gvs.sample(zs, ys), zs[-1]


ims, z0 = sample(z0)
cur_frame = 0
num_resets = 0
try:
    while True:
        if cur_frame >= num_frames:
            cur_frame = 0
            ims, z0 = sample(z0)

        cv2.imshow('win', ims[cur_frame])
        cv2.waitKey(25)
        cur_frame += 1
except KeyboardInterrupt:
    cv2.destroyAllWindows()
