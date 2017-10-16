import cv2
import numpy as np

def cmp_foreground(foreground, X):
    return np.take(foreground, -1, 2) != np.take(X, -1, 2)

def calcE(frame):
    mask = cmp_foreground(np.zeros((frame.shape)), frame)
    nmask = np.invert(mask)
    E[mask] += 1
    E[nmask] -= k
    E[E > max_e] = max_e
    E[E < 0] = 0

if __name__ == '__main__':
    camera = cv2.VideoCapture('video1.avi')
    camera.set(cv2.CAP_PROP_FPS, 25.0)
    background = camera.read()[1]
    background = cv2.GaussianBlur(background, (5, 5), 0)
    lt_f = background
    st_f = background
    E = np.zeros((background.shape[0], background.shape[1]))
    k = 255
    max_e = 255
    cnt = 0
    while 1:
        ret, frame = camera.read()
        if not ret:
            raise Exception('No signal')
        _frame = cv2.GaussianBlur(frame, (5, 5), 0)
        _frame = cv2.absdiff(_frame, background)
        _, _frame = cv2.threshold(_frame, 25, 255, cv2.THRESH_BINARY)
        calcE(_frame)
        _, _E = cv2.threshold(E, 250, 255, cv2.THRESH_BINARY)
        _E = _E.astype(np.uint8)
        (_, cnts, _) = cv2.findContours(_E.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 900:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('E', _E)
        cv2.imshow('Frame', frame)
        key = cv2.waitKeyEx(20)
        if key == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()