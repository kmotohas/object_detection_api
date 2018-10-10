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

# Flaskクラスのインスタンスを作成
# __name__は現在のファイルのモジュール名
api = Flask(__name__)

model = YOLOv3(pretrained_model='voc0712')
chainer.cuda.get_device_from_id(0)
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
    json_dict = json.loads(request.data)
    # numpy arrayに変換
    decoded_image = base64.decodestring(json_dict['image'].encode('utf-8'))
    image = np.frombuffer(decoded_image, dtype=np.uint8)
    # 1d array to 3d array
    image = np.resize(image, (json_dict['height'], json_dict['width'], 3))
    # be proper as input for chainercv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = image.transpose((2,0,1)) # transpose the dimensions from H-W-C to C-H-W
    # 物体認識
    bboxes, labels, scores = model.predict([image])
    # jsonに詰めて返す
    result = {}
    result['bboxes'] = bboxes[0].tolist()
    result['labels'] = labels[0].tolist()
    result['scores'] = scores[0].tolist()
    return make_response(jsonify(result))

# エラーハンドリング
@api.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

# ファイルをスクリプトとして実行した際に
# ホスト0.0.0.0, ポート3001番でサーバーを起動
if __name__ == '__main__':
    api.run(host='0.0.0.0', port=3001)

