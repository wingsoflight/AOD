import cv2
import numpy as np
import argparse
import datetime
import requests
from requests_toolbelt.multipart import MultipartEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--URI', help = 'URI of input video stream', default = 0)


class AbandonedObjectDetector:
    def __init__(self, uri = 0, k = 300, max_e = 300, long_term_lr = .0001, short_term_lr = .01, long_term_h = 1, short_term_h = 1,
                 long_term_thresh = 900, short_term_thresh = 450, min_area = 625, detect_shadows = False):
        self.uri = uri
        self.camera = cv2.VideoCapture(uri)
        self.k = k
        self.max_e = max_e
        self.long_term_lr = long_term_lr
        self.short_term_lr = short_term_lr
        self.long_term_subs = cv2.createBackgroundSubtractorMOG2(long_term_h, long_term_thresh, detect_shadows)
        self.short_term_subs = cv2.createBackgroundSubtractorMOG2(short_term_h, short_term_thresh, detect_shadows)
        self.min_area = min_area
        ret, frame = self.camera.read()
        if not ret:
            raise Exception('Unable to access camera')
        self.shape = (int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print(self.shape)
        self.E = np.zeros(shape = self.shape)
        self.sample = np.zeros(shape = self.shape, dtype=np.uint8)

    def calc_e(self, lt_f, st_f):
        ltf_tm = lt_f.astype(np.bool)
        stf_tm = st_f.astype(np.bool)
        ltf_fm = np.invert(ltf_tm)
        stf_fm = np.invert(stf_tm)
        self.E[ltf_tm & stf_fm] += 1
        self.E[ltf_fm | stf_tm] -= self.k
        self.E[self.E < 0] = 0
        self.E[self.E > self.max_e] = self.max_e

    def start(self):
        post = True
        num_objs = 0
        counter = 900
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print('Camera disconnected at', datetime.datetime.now())
                break
            t_num_objs = 0
            lt_f = self.long_term_subs.apply(frame, learningRate = self.long_term_lr)
            st_f = self.short_term_subs.apply(frame, learningRate = self.short_term_lr)
            self.calc_e(lt_f, st_f)
            _, _E = cv2.threshold(self.E, 299, 255, 0)
            _E = _E.astype(np.uint8)
            _, contours, _ = cv2.findContours(_E, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < self.min_area:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                t_num_objs += 1
            if t_num_objs > num_objs:
                _, img = cv2.imencode('.jpg', frame)
                data = MultipartEncoder(
                    fields={
                        'json': 'Detected object at ' + str(datetime.datetime.now()) + ' URI: ' + str(self.uri),
                        'file': ('frame.jpg', bytes(img), 'text/plain')
                    })
                requests.post('http://212.19.138.141/ksk/upload.php', data=data, headers={'Content-Type': data.content_type})
            num_objs = t_num_objs
            t_num_objs = 0
            if counter <= 0:
                counter = 900
                num_objs = 0
            elif num_objs:
                counter -= 1
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key is ord('q'):
                break
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parser.parse_args()
    AbandonedObjectDetector(args.URI).start()
