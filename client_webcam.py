#!/usr/bin/env python

import numpy as np
import cv2
import base64
import urllib.request
import json
import argparse
from chainercv.datasets.voc import voc_utils 

parser = argparse.ArgumentParser(
    prog='client_webcam',
    usage='python client_webcam.py -i <host server ip> -p <api port>',
    description='',
    add_help=True,
    )
parser.add_argument('-i', '--ip', default='192.168.12.33', type=str, help='ip address of API host server')
parser.add_argument('-p', '--port', default='3001', type=str, help='port number for object detection API')

args = parser.parse_args()

url = 'http://' + args.ip + ':' + args.port + '/api/detection'
method = 'POST'
headers = {'Content-Type': 'application/json'}
dic = {}

# capture video from web camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resize image to reduce time to transfer data
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    #print(frame.shape)
    encoded = base64.b64encode(frame)
    dic['height'] = frame.shape[0]
    dic['width'] = frame.shape[1]
    dic['image'] = str(encoded)[2:-1]
    jsonstring = json.dumps(dic).encode('utf-8')
    #print(jsonstring)
    request = urllib.request.Request(url, data=jsonstring, method=method, headers=headers)
    with urllib.request.urlopen(request) as response:
        response_body = json.loads(response.read().decode('utf-8').strip('\n'))
        #print(repr(response_body))
        bbox = response_body['bboxes']
        label = response_body['labels']
        score = response_body['scores']
        if len(bbox) != 0:
            for i, bb in enumerate(bbox):
                # Interpret output, only one frame is used
                #print(i)
                lb = label[i]
                conf = score[i]
                ymin = int(bb[0])
                xmin = int(bb[1])
                ymax = int(bb[2])
                xmax = int(bb[3])

                # Draw the box on top of the to_draw image
                class_num = int(lb)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                voc_utils.voc_semantic_segmentation_label_colors[class_num], 2)
                text = voc_utils.voc_bbox_label_names[class_num] + " " + ('%.2f' % conf)
                #print(text)

                text_top = (xmin, ymin - 10)
                text_bot = (xmin + 80, ymin + 5)
                text_pos = (xmin + 5, ymin)
                cv2.rectangle(frame, text_top, text_bot, voc_utils.voc_semantic_segmentation_label_colors[class_num], -1)
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
