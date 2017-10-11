import cv2
import numpy as np

def F_l(_lt_f, X):
    return _lt_f == X

def F_s(_st_f, X):
    return _st_f == X

def calcE(_E, _lt_f, _st_f, X):
    fl_tm = F_l(_lt_f, X)
    fs_tm = F_s(_st_f, X)
    fl_fm = np.invert(fl_tm)
    fs_fm = np.invert(fs_tm)
    _E[fl_tm & fs_fm] += 1
    _E[fl_fm | fs_tm] -= k
    _E[_E > max_e] = max_e
    _E[_E < 0] = 0
    return _E

if __name__ == '__main__':
    camera = cv2.VideoCapture('video1.avi')
    background = camera.read()[1]
    lt_f = background
    st_f = background
    E = np.zeros((background.shape))
    k = 300
    max_e = 300
    while 1:
        ret, frame = camera.read()
        if not ret:
            raise Exception('No signal')
        E = calcE(E, lt_f, st_f, frame)
        #print(E.sum())
        cv2.imshow('Frame', frame)
        key = cv2.waitKeyEx(int(1000 / camera.get(cv2.CAP_PROP_FPS)))
        if key == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
