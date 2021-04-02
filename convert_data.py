import cv2
import numpy as np
from sys import argv
import os


def max_pooling(frames):
    # MAX POOLING
    w, h, d = frames.shape
    w2, h2 = 5, 5
    w_max_pooling, h_max_pooling = w // w2, h // h2
    frames = frames[:w_max_pooling * w2, :h_max_pooling * h2, :].reshape(w_max_pooling, w2, h_max_pooling, h2, d).max(
        axis=(1, 3))
    return frames

def get_frames(sec, video_fn):
    # load video
    vidcap = cv2.VideoCapture(video_fn)
    # set how many frames per second (1/frameRate)
    amount_of_frames = 4
    frame_rate = 1/amount_of_frames
    #
    frames = []
    # hij pakt alleen de eerste 4 frames binnen 1 seconde
    for _ in range(amount_of_frames):
        # set to particular image and read image
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        hasFrames, frame = vidcap.read()

        # if not empty
        if hasFrames:
            frames.append(frame)
        # to the next frame
        sec += frame_rate
    return frames


def make_data_arrays():
    X = list()
    y = list()
    sec = 0
    folders = os.listdir("data/videos")
    for folder in folders:
        videos = os.listdir("data/videos/" + folder)
        for video_fn in videos:
            video_fn = "data/videos/" + folder + "/" + video_fn
            frames = np.array(get_frames(sec, video_fn))
            frames = np.concatenate(frames, axis=2)

            frames = max_pooling(frames)

            X.append(frames)
            y.append(folder)

    np.save('data/npy/X.npy', np.array(X, dtype=object))
    np.save('data/npy/labels.npy', np.array(y, dtype=object))


def main():
    make_data_arrays()


if __name__ == '__main__':
    main()
