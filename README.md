# object_detection_api

chainercvとFlaskを用いた物体認識APIです。
今は物体認識アルゴリズムとしてYOLOv3を用いています。

## 準備

Anaconda + Python3系で動く。

```bash
# on local client
conda install chainer chainercv opencv seaborn
# on server with GPU
conda install chainer chainercv opencv
pip install flask  # 環境によっては sudo が必要
```

## 使い方

```bash
# on server with GPU
python yolo_api.py  # これでFlaskがAPI立ち上げてくれる
# on local client
python client_webcam.py -i <ip address of API server> -p <API port, default 3001>  # localで物体認識もしたいときは -l オプションを足す
# localかつGPUで物体認識をしたいときは -g オプションをつける。このとき -l オプションは外すこと。
```

`speed.db`にクライアント側の処理時間、`speed_server.db`にサーバ側の処理時間が記録される。

プロットを作成するには`speed_server.db`をクライアント側に持ってきて、`draw.py`を走らせる。

```bash
scp <user>@<ip>:<path to speed_server.db> .
python draw.py  # LocalかつGPUの結果を描画したいときは -g オプションをつける
```
