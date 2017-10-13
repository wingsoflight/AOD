import cv2
import numpy as np

def cmp_foreground(foreground, X):
    return np.take(foreground, -1, 2) != np.take(X, -1, 2)

def calcE(frame):
    mask = cmp_foreground(np.zeros((frame.shape)), frame)
    nmask = np.invert(mask)
    E[nmask] -= k
    E[mask] += 1
    E[E > max_e] = max_e
    E[E < 0] = 0
    print(len(E[E == max_e]))

if __name__ == '__main__':
    camera = cv2.VideoCapture('video1.avi')
    camera.set(cv2.CAP_PROP_FPS, 25.0)
    background = camera.read()[1]
    background = cv2.GaussianBlur(background, (5, 5), 0)
    lt_f = background
    st_f = background
    E = np.zeros((background.shape[0], background.shape[1]))
    k = 255
    max_e = 254
    cnt = 0
    while 1:
        ret, frame = camera.read()
        if not ret:
            raise Exception('No signal')
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.absdiff(frame, background)
        _, frame = cv2.threshold(frame, 25, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        calcE(frame)
        cv2.imshow('E', E)
        cv2.imshow('Frame', frame)
        key = cv2.waitKeyEx(20)
        if key == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()