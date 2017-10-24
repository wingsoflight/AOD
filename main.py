import cv2
import numpy as np


def calc_e():
    ltf_tm = lt_f.astype(np.bool)
    stf_tm = st_f.astype(np.bool)
    ltf_fm = np.invert(ltf_tm)
    stf_fm = np.invert(stf_tm)
    E[ltf_tm & stf_fm] += 1
    E[ltf_fm | stf_tm] -= k
    E[E < 0] = 0
    E[E > max_e] = max_e


if __name__ == '__main__':
    camera = cv2.VideoCapture('video1.avi')
    ret, sample = camera.read()
    if not ret:
        raise Exception('No signal')
    E = np.zeros((sample.shape[0], sample.shape[1]))
    bg = np.zeros(shape=sample.shape, dtype=np.uint8)
    k = 300
    max_e = 300
    lt_bs = cv2.createBackgroundSubtractorMOG2(1, 900, False)
    st_bs = cv2.createBackgroundSubtractorMOG2(1, 200, False)
    while 1:
        ret, frame = camera.read()
        if not ret:
            raise Exception('No signal')
        lt_f = lt_bs.apply(frame, bg, learningRate=0.0001)
        st_f = st_bs.apply(frame, bg, learningRate=0.01)
        calc_e()
        _, _E = cv2.threshold(E, 299, 255, 0)
        _E = _E.astype(np.uint8)
        _, contours, _ = cv2.findContours(_E, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 256:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('LongTerm', lt_f)
        cv2.imshow('ShortTerm', st_f)
        cv2.imshow('Original', frame)
        key = cv2.waitKeyEx(15)
        if key == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
