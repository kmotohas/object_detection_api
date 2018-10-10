# object_detection_api
chainercvとFlaskを用いた物体認識API

## 準備
Anaconda + Python3系で動く。
```
# on local client
conda install chainer chainercv opencv
# on server with GPU
conda install chainer chainercv opencv
pip install flask
```

## 使い方
```
# on server with GPU
python yolo_api.py  # これでFlaskがAPI立ち上げてくれる
# on local client
python client_webcam.py -i <ip address of API server> -p <API port, default 3001>
```