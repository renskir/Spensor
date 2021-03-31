import tensorflow as tf
import numpy as np
import cv2


def predict(X, labels):
    model = tf.keras.models.load_model('data/model/model')

    y = model.predict(X)
    arg = np.argmax(y, axis=1)
    prediction = labels[arg]

    return prediction


def main():
    labels = np.load('data/npy/labels.npy', allow_pickle=True)

    # turns on camera
    # STILL NEED TO IMPLEMENT THAT ONLY 4 FRAMES PER SECOND ARE READ!!
    vidcap = cv2.VideoCapture(0)
    sec = 0

    frames = []
    max_len = 4

    while vidcap.isOpened():
        # set to particular image and read image
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, frame = vidcap.read()
        # if not empty
        if hasFrames:
            frame = np.array(frame[:, 100:-100, :])

            frames.append(frame)

        if len(frames) > 4:
            frames.pop(0)

        if len(frames) == 4:
            X = np.array([np.concatenate(frames, axis=2)])
            prediction = predict(X, labels)
            print(prediction)
            # connect to sound here
        # add frame_rate to sec
        amount_of_frames = 4
        frame_rate = 1 / amount_of_frames
        # break

if __name__ == "__main__":
    main()
