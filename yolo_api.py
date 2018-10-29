#!/usr/bin/env python
# coding: utf-8

# 必要なモジュールの読み込み
from flask import Flask, jsonify, abort, make_response, request

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv3
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox

import json
import base64
from PIL import Image
from io import BytesIO
import urllib
import numpy as np
import cv2
import chainer
import sqlite3
import argparse
import time

parser = argparse.ArgumentParser(
    prog='yolo_api',
    usage='python yolo_api.py -g 1 -p <api port>',
    description='',
    add_help=True,
    )
parser.add_argument('-g', '--gpu', default=1, type=int, help='Boolean. Set 1 if you want to use GPUs, otherwise set 0.')
parser.add_argument('-p', '--port', default='3001', type=str, help='port number for object detection API')

args = parser.parse_args()

period_list = ['image_decode', 'image_transform', 'object_detection', 'pack_result']
period_dict = {period: 0 for period in period_list}
#period_dict['environment'] = 'remote'
#dbfile = sqlite3.connect('speed_server.db')
#c = dbfile.cursor()

# Flaskクラスのインスタンスを作成
# __name__は現在のファイルのモジュール名
api = Flask(__name__)

# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/yolo/yolo_v3.py
# input image size = 416 x 416 pix
model = YOLOv3(pretrained_model='voc0712')

if args.gpu:
    chainer.backends.cuda.get_device_from_id(0)
    model.to_gpu()

# POSTの実装
@api.route('/api/detection', methods=['POST'])
def post():
    """
    model.predict([img])の出力例
    bboxes: [array([[ 99.97439  ,   4.5523834, 192.06816  , 134.67502  ],
                    [199.90709  , 143.38562  , 352.9582   , 195.82318  ],
                    [ 75.87911  , 204.2211   , 360.09158  , 426.09985  ],
                    [ 11.633713 , 264.68134  , 225.85754  , 356.1368   ],
                    [126.243256 , 429.72067  , 178.3947   , 447.91504  ]], dtype=float32)]
    labels: [array([ 6, 11, 12, 14, 14], dtype=int32)]
    scores: [array([0.9986385 , 0.82413423, 0.9998919 , 0.99992335, 0.9953939 ], dtype=float32)]
    """
    # POSTでbase64 encodeされた画像をjson形式で受け取り
    start_time = time.time()
    json_dict = json.loads(request.data)
    # numpy arrayに変換
    decoded_image = base64.decodestring(json_dict['image'].encode('utf-8'))
    image = np.frombuffer(decoded_image, dtype=np.uint8)
    period_dict['image_decode'] = time.time() - start_time
    # 1d array to 3d array
    start_time = time.time()
    image = np.resize(image, (json_dict['height'], json_dict['width'], 3))
    # be proper as input for chainercv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = image.transpose((2,0,1)) # transpose the dimensions from H-W-C to C-H-W
    period_dict['image_transform'] = time.time() - start_time
    # 物体認識
    start_time = time.time()
    bboxes, labels, scores = model.predict([image])
    period_dict['object_detection'] = time.time() - start_time
    # jsonに詰めて返す
    start_time = time.time()
    result = {}
    result['bboxes'] = bboxes[0].tolist()
    result['labels'] = labels[0].tolist()
    result['scores'] = scores[0].tolist()
    period_dict['pack_result'] = time.time() - start_time
    # sqlite3に速度測定結果を詰める
    start_time = time.time()
    dbfile = sqlite3.connect('speed_server.db')  # TODO: SQLite objects created in a thread can only be used in that same thread
    c = dbfile.cursor()
    period_dict['environment'] = json_dict['environment']
    # speed_server(event_number int, image_decode float, image_transform float, object_detection float, pack_result float)
    sql = "insert into speed_server02 values( \
            {event_number}, {image_decode}, {image_transform}, {object_detection}, {pack_result}, {environment}\
            );".format(event_number=json_dict['event_number'], **period_dict)
    c.execute(sql)
    dbfile.commit()
    dbfile.close()
    print('database process time: ', start_time - time.time())  # check
    return make_response(jsonify(result))

# エラーハンドリング
@api.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

# ファイルをスクリプトとして実行した際に
# ホスト0.0.0.0, ポート3001番でサーバーを起動
if __name__ == '__main__':
    api.run(host='0.0.0.0', port=args.port)

