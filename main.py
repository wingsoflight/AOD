import cv2
import numpy as np


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    alpha_short_term = 1
    alpha_long_term = .01
    background = camera.read()[1]
    lt_img = background.copy()
    st_img = background.copy()
    n = 0
    while 1:
        ret, frame = camera.read()
        if not ret:
            raise Exception('No signal')
        lt_img = lt_img.astype(np.float64)
        st_img = st_img.astype(np.float64)
        frame = frame.astype(np.float64)
        lt_img += (lt_img - frame) * alpha_long_term
        frame = frame.astype(np.uint8)
        lt_img = lt_img.astype(np.uint8)
        cv2.imshow('Short term foreground', st_img)
        cv2.imshow('Long term foreground', lt_img)
        key = cv2.waitKeyEx(1)
        if key == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
