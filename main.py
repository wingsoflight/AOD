import cv2
import numpy as np
import argparse
import datetime
import requests
from requests_toolbelt.multipart import MultipartEncoder
import subprocess as sp
from imutils.object_detection import non_max_suppression
import imutils

parser = argparse.ArgumentParser()
parser.add_argument('--ID', help='Identifier of camera', type=int)
parser.add_argument('--URI', help='URI of input video stream', default=0)


class AbandonedObjectDetector:
    def __init__(self, id, uri = 0, k = 300, max_e = 300, long_term_lr = .0001, short_term_lr = .01, long_term_h = 1, short_term_h = 1,
                 long_term_thresh = 900, short_term_thresh = 450, min_area = 625, detect_shadows = False):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.id = id
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

    def detect_people(self, frame):
        (rects, weights) = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.06
        )
        rects = non_max_suppression(rects, probs=None, overlapThresh=.65)
        return rects


    def start(self):
        post = True
        num_objs = 0
        counter = 900
        v_resolution = '{}x{}'.format(self.shape[1], self.shape[0])
        fps = str(int(self.camera.get(cv2.CAP_PROP_FPS)))
        command = ['ffmpeg',
                   '-f', 'rawvideo',
                   '-c:v', 'rawvideo',
                   '-s', v_resolution,
                   '-pix_fmt', 'bgr24',
                   '-r', fps,
                   '-i', '-',
                   '-f', 'flv',
                   '-an',
                   '-c:v', 'libx264',
                   '-preset', 'ultrafast',
                   'rtmp://localhost/live/'+str(self.id)]
        proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
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
            humans = self.detect_people(frame)
            for x,y,w,h in humans:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
                        'json': 'Detected object at ' + str(datetime.datetime.now()) + 'Camera ID: ' + str(self.id),
                        'file': ('frame.jpg', bytes(img), 'text/plain')
                    }
                )
                requests.post('http://212.19.138.141/ksk/upload.php', data=data, headers={'Content-Type': data.content_type})
            num_objs = t_num_objs
            t_num_objs = 0
            if counter <= 0:
                counter = 900
                num_objs = 0
            elif num_objs:
                counter -= 1
            proc.stdin.write(frame.tostring())
            key = cv2.waitKey(1) & 0xFF
            if key is ord('q'):
                break
        self.camera.release()
        cv2.destroyAllWindows()
        proc.stdin.close()
        proc.stderr.close()
        proc.wait()


if __name__ == '__main__':
    args = parser.parse_args()
    AbandonedObjectDetector(args.ID, args.URI).start()
