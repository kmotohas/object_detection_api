#!/usr/bin/env python

import numpy as np
import cv2
import base64
import urllib.request
import json
import argparse
from chainercv.links import YOLOv3
from chainercv.datasets.voc import voc_utils 
import sqlite3
import time

parser = argparse.ArgumentParser(
    prog='client_webcam',
    usage='python client_webcam.py -i <host server ip> -p <api port>',
    description='',
    add_help=True,
    )
parser.add_argument('-i', '--ip', default='192.168.12.33', type=str, help='ip address of API host server')
parser.add_argument('-p', '--port', default='3001', type=str, help='port number for object detection API')
parser.add_argument('-l', '--local', action='store_true', help='run object detection on local if this option is set')

args = parser.parse_args()

url = 'http://' + args.ip + ':' + args.port + '/api/detection'
method = 'POST'
headers = {'Content-Type': 'application/json'}
dic = {}
response_body = {}
process_time = {}
yolo_input_size = 416
period_list = ['cap_read', 'cv2_resize', 'image_encode', 'image_post', 'object_detection', 
               'bbox_draw', 'textbox_draw', 'text_put', 'cv2_imshow']
period_dict = {period: 0 for period in period_list}
period_dict['environment'] = 'local' if args.local else 'remote'

# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/yolo/yolo_v3.py
# input image size = 416 x 416 pix
model = YOLOv3(pretrained_model='voc0712') if args.local else None

# capture video from web camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    event_number = int(time.time() * 1000000)
    start_time = time.time()
    ret, orig_frame = cap.read()  # orig_frame.shape = (720, 1280, 3)
    period_dict['cap_read'] = time.time() - start_time

    # resize image to reduce time to transfer data
    start_time = time.time()
    frame = cv2.resize(orig_frame, (yolo_input_size, yolo_input_size))  # x, y
    period_dict['cv2_resize'] = time.time() - start_time
    # keep the ratio of orig shape to yolo input shape to resize boundary box later
    fx = orig_frame.shape[1] / yolo_input_size
    fy = orig_frame.shape[0] / yolo_input_size
    if not args.local:
        start_time = time.time()
        encoded = base64.b64encode(frame)
        dic['height'] = frame.shape[0]
        dic['width'] = frame.shape[1]
        dic['image'] = str(encoded)[2:-1]
        dic['event_number'] = event_number
        jsonstring = json.dumps(dic).encode('utf-8')
        period_dict['image_encode'] = time.time() - start_time
        request = urllib.request.Request(url, data=jsonstring, method=method, headers=headers)
        start_time = time.time()
        with urllib.request.urlopen(request) as response:
            period_dict['image_post'] = time.time() - start_time
            response_body = json.loads(response.read().decode('utf-8').strip('\n'))
        bbox, label, score = response_body['bboxes'], response_body['labels'], response_body['scores']
    else:
        # be proper as input for chainercv
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame = frame.transpose((2,0,1)) # transpose the dimensions from H-W-C to C-H-W
        # object detection
        start_time = time.time()
        bboxes, labels, scores = model.predict([frame])
        period_dict['object_detection'] = time.time() - start_time
        # numpy array to list
        bbox = bboxes[0].tolist()
        label = labels[0].tolist()
        score = scores[0].tolist()
    if len(bbox) != 0:
        for i, bb in enumerate(bbox):
            # Interpret output, only one frame is used
            lb = label[i]
            conf = score[i]
            ymin, xmin, ymax, xmax = int(bb[0] * fy), int(bb[1] * fx), int(bb[2] * fy), int(bb[3] * fx)
            # Draw the box on top of the to_draw image
            class_num = int(lb)
            start_time = time.time()
            cv2.rectangle(orig_frame, (xmin, ymin), (xmax, ymax),
                          voc_utils.voc_semantic_segmentation_label_colors[class_num], 2)
            period_dict['bbox_draw'] = time.time() - start_time
            text = voc_utils.voc_bbox_label_names[class_num] + " " + ('%.2f' % conf)
            text_top = (xmin, ymin - 10)
            text_bot = (xmin + 80, ymin + 5)
            text_pos = (xmin + 5, ymin)
            start_time = time.time()
            cv2.rectangle(orig_frame, text_top, text_bot, voc_utils.voc_semantic_segmentation_label_colors[class_num], -1)
            period_dict['textbox_draw'] = time.time() - start_time
            start_time = time.time()
            cv2.putText(orig_frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            period_dict['text_put'] = time.time() - start_time
    start_time = time.time()
    cv2.imshow('frame', orig_frame)
    period_dict['cv2_imshow'] = time.time() - start_time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #
    dbfile = sqlite3.connect('speed.db')
    c = dbfile.cursor()
    # speed(event_number int, cap_read float, cv2_resize float, image_encode float, image_post float, object_detection float, bbox_draw float, textbox_draw float, text_put float, cv2_imshow float, environment text)
    sql = "insert into speed02 values( \
            {event_number}, {cap_read}, {cv2_resize}, {image_encode}, {image_post}, {object_detection}, {bbox_draw}, {textbox_draw}, {text_put}, {cv2_imshow}, \"{environment}\" \
           );".format(event_number=event_number, **period_dict)
    c.execute(sql)
    dbfile.commit()
    dbfile.close()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
dbfile.close()
