import subprocess as sp
from subprocess import check_output
import time
import json
import urllib.request
import signal
import os


link = 'http://localhost:8000/video/getList'
jsonData = None
procDict = {}  # [Camera id]{key} to [Process pid]{value} dictionary
timeDict = {}  # [Process pid]{key} to [Elapsed time]{value} dictionary


def startproc(obj_id, obj_url):
    cmd = [
        'python3',
        'main.py',
        '--ID', str(obj_id),
        '--URI', obj_url,
    ]
    return sp.Popen(cmd, stderr=sp.PIPE).pid


while 1:
    with urllib.request.urlopen(link) as url:
        jsonData = json.loads(url.read().decode())

    for obj in jsonData:
        obj_id = obj['id']
        obj_url = obj['url']
        if procDict.get(obj_id) is None:
            pid = startproc(obj_id, obj_url)
            procDict[obj_id] = pid
            timeDict[pid] = check_output(['ps', '-o', 'time=', str(pid)])
        else:
            pid = procDict[obj_id]
            tmp = check_output(['ps', '-o', 'time=', str(pid)])
            if tmp == timeDict[pid]:
                print('Process', pid, 'has frozen')
                os.kill(pid, signal.SIGTERM)
                del timeDict[pid]
                pid = startproc(obj_id, obj_url)
                procDict[obj_id] = pid
            timeDict[pid] = check_output(['ps', '-o', 'time=', str(pid)])
    print(procDict)
    print(timeDict)
    time.sleep(10)
