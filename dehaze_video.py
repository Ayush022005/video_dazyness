import cv2
import math
import numpy as np
import sys

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value=255, high_value=255):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)
    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2

        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]
        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        thresholded = apply_threshold(channel, low_val, high_val)
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

if __name__ == '__main__':
    cap = cv2.VideoCapture('./haze-videos/fire.mp4')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = simplest_cb(frame, 1)
        cv2.imshow("Before", frame)
        cv2.imshow("After", out)
        
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
