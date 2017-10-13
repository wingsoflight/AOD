import cv2
import numpy as np

def cmp_foreground(foreground, X):
    return np.take(foreground, -1, 2) == np.take(X, -1, 2)

def calcE(_E, _lt_f, _st_f, X):
    fl_tm = cmp_foreground(_lt_f, X)
    fs_tm = cmp_foreground(_st_f, X)
    fl_fm = np.invert(fl_tm)
    fs_fm = np.invert(fs_tm)
    _E[fl_tm & fs_fm] += 1
    _E[fl_fm | fs_tm] -= k
    _E[_E > max_e] = max_e
    _E[_E < 0] = 0
    return _E


if __name__ == '__main__':
    camera = cv2.VideoCapture('video1.avi')
    camera.set(cv2.CAP_PROP_FPS, 25.0)
    background = camera.read()[1]
    background = cv2.GaussianBlur(background, (5, 5), 0)
    lt_f = background
    st_f = background
    E = np.zeros((background.shape[0], background.shape[1]))
    k = 300
    max_e = 300
    cnt = 0
    while 1:
        ret, frame = camera.read()
        if not ret:
            raise Exception('No signal')
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.absdiff(frame, background)
        E = calcE(E, lt_f,st_f,frame)
        _, frame = cv2.threshold(frame, 25, 255, cv2.THRESH_BINARY)
        cv2.imshow('Frame', frame)
        key = cv2.waitKeyEx(20)
        if key == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()