import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import sqlite3
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import deque
from statistics import mean, median, variance, stdev
from copy import deepcopy

sns.set()
sns.set_style('whitegrid', {'grid.linestyle': '--'})
sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2})

# speed(event_number int, cap_read float, cv2_resize float, image_encode float, image_post float, bbox_draw float, textbox_draw float, text_put float, cv2_imshow float, environment text)
conn_sql = sqlite3.connect('speed.db')
cursor = conn_sql.cursor()
df = pd.read_sql_query('select * from speed;', conn_sql)
# speed_server(event_number int, image_decode float, image_transform float, object_detection float, pack_result float)
conn_sql_server = sqlite3.connect('speed_server.db')
cursor_server = conn_sql_server.cursor()
df_server = pd.read_sql_query('select * from speed_server;', conn_sql_server)


class SpeedTest:
    def __init__(self):
        self.keys = ['event_number', 'cap_read', 'cv2_resize', 'image_encode', 'object_detection',
                     'bbox_draw', 'textbox_draw', 'text_put', 'cv2_imshow', 'datetime',
                     'image_post', 'image_decode', 'image_transform', 'pack_result', 'http_post']
        self.some_keys = deepcopy(self.keys)
        self.measurements = {key: [] for key in self.keys}
        # keys replacement for stack bar plot
        self.some_keys.remove('object_detection')
        self.some_keys.remove('http_post')
        self.some_keys.append('object_detection')
        self.some_keys.append('http_post')
        #
        self.keys.remove('event_number')
        self.keys.remove('datetime')
        self.keys.remove('image_post')
        self.some_keys.remove('event_number')
        self.some_keys.remove('datetime')
        self.some_keys.remove('image_post')
        # negiligible
        self.some_keys.remove('cv2_resize')
        self.some_keys.remove('pack_result')
        self.some_keys.remove('image_transform')
        self.some_keys.remove('bbox_draw')
        self.some_keys.remove('textbox_draw')
        self.some_keys.remove('text_put')
        self.stack_mean = {}
        self.stack_median = {}
        self.stack_stdev = {}
    
    def retrieve_data(self, df, df_server=None):
        pass

    def stack_measurements(self):
        for key in self.keys:
            self.stack_mean[key] = mean(self.measurements[key]) if len(self.measurements[key]) > 0 else 0 
            self.stack_median[key] = median(self.measurements[key]) if len(self.measurements[key]) > 0 else 0 
            self.stack_stdev[key] = stdev(self.measurements[key]) if len(self.measurements[key]) > 0 else 0


class SpeedTestLocal(SpeedTest):
    def __init__(self, arch='CPU', net='Local'):
        super().__init__()
        self.arch = arch
        self.net = net
        self.legend = '({}, {})'.format(arch, net)
    
    def retrieve_data(self, df, df_server=None):
        self.stack_measurements()
        pass


class SpeedTestRemote(SpeedTest):
    def __init__(self, arch='GPU', net='WiFi'):
        super().__init__()
        self.arch = arch
        self.net = net
        self.legend = '({}, {})'.format(arch, net)
    
    def retrieve_data(self, df, df_server=None):
        if df_server is None:
            print('argument dl_server is not set')
            return False
        col_list = list(df_server)
        col_list.remove('event_number')
        # retrieve data whose event numbers match on both local and remote
        # もっと効率いいやり方ないかな
        for _, event_number in df_server.event_number.iteritems():
            df_searched = df.query('event_number == {}'.format(event_number))
            if len(df_searched) == 0:  # the event number is not found on local db
                continue
            df_server_searched = df_server.query('event_number == {}'.format(event_number))
            # client
            self.measurements['event_number'].append(event_number)
            self.measurements['cap_read'].append(df_searched.cap_read.iat[0])
            self.measurements['cv2_resize'].append(df_searched.cv2_resize.iat[0])
            self.measurements['image_encode'].append(df_searched.image_encode.iat[0])
            self.measurements['bbox_draw'].append(df_searched.bbox_draw.iat[0])
            self.measurements['textbox_draw'].append(df_searched.textbox_draw.iat[0])
            self.measurements['text_put'].append(df_searched.text_put.iat[0])
            self.measurements['cv2_imshow'].append(df_searched.cv2_imshow.iat[0])
            self.measurements['datetime'].append(datetime.fromtimestamp(event_number/1e6, timezone(timedelta(hours=+9), 'JST')))
            self.measurements['image_post'].append(df_searched.image_post.iat[0])
            # server
            self.measurements['image_decode'].append(df_server_searched.image_decode.iat[0])
            self.measurements['image_transform'].append(df_server_searched.image_transform.iat[0])
            self.measurements['object_detection'].append(df_server_searched.object_detection.iat[0])
            self.measurements['pack_result'].append(df_server_searched.pack_result.iat[0])
            self.measurements['http_post'].append(df_searched.image_post.iat[0] - df_server_searched[col_list].sum(axis=1).iat[0])
        self.stack_measurements()


def main():
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    speed_test_local = SpeedTestLocal(arch='CPU', net='Local')
    speed_test_local.retrieve_data(df=df)
    speed_test_remote = SpeedTestRemote(arch='GPU', net='WiFi')
    speed_test_remote.retrieve_data(df=df, df_server=df_server)
    speed_test_remote_5G = SpeedTestRemote(arch='GPU', net='5G')
    # ox
    # xx
    #sns.distplot(df.image_post - df_server[col_list].sum(axis=1), norm_hist=True, bins=100, ax=ax[0][0], label='HTTP POST (WiFi)')
    sns.distplot(speed_test_remote.measurements['http_post'], norm_hist=True, bins=100, ax=ax[0][0], label='HTTP POST ' + speed_test_remote.legend)
    sns.distplot(speed_test_remote.measurements['object_detection'], norm_hist=True, bins=10, ax=ax[0][0], label='YOLO Prediction ' + speed_test_remote.legend)
    sns.distplot(speed_test_remote.measurements['image_decode'], norm_hist=True, bins=300, ax=ax[0][0], label='Image Decoding ' + speed_test_remote.legend)
    ax[0][0].set_xlabel('Process Time [s]')
    ax[0][0].set_ylabel('Arbitrary Unit')
    ax[0][0].set_xlim(xmin=0, xmax=0.8)
    ax[0][0].legend()
    # xo
    # xx
    ax[0][1].plot(speed_test_remote.measurements['datetime'], speed_test_remote.measurements['http_post'], 'o', 
                  label='HTTP POST ' + speed_test_remote.legend)
    ax[0][1].plot(speed_test_remote.measurements['datetime'], speed_test_remote.measurements['object_detection'], 'D', 
                  label='YOLO Prediction ' + speed_test_remote.legend)
    ax[0][1].plot(speed_test_remote.measurements['datetime'], speed_test_remote.measurements['image_decode'], 'x', 
                  label='Image Decoding ' + speed_test_remote.legend)
    ax[0][1].set_xlabel('Time (JST)')
    ax[0][1].set_ylabel('Process Time [s]')
    #ax[0][1].set_ylim(ymin=0.008, ymax=2.0)
    ax[0][1].set_yscale('log')
    ax[0][1].get_yaxis().set_major_locator(tkr.LogLocator(base=10, subs='all'))
    ax[0][1].legend()
    # xx
    # ox
    nbars = 3
    indexes = np.arange(nbars)
    bars = []
    cumulated = [0 for _ in indexes]
    for i, key in enumerate(speed_test_remote.keys):
        if key in speed_test_remote.some_keys:
            bars.append(ax[1][0].bar(indexes, [speed_test_local.stack_median[key], speed_test_remote.stack_median[key], 0], width=0.5, bottom=cumulated))
            cumulated[0] += speed_test_local.stack_median[key]
            cumulated[1] += speed_test_remote.stack_median[key]
            cumulated[2] += 0
    bars.append(ax[1][0].bar(indexes, [sum(speed_test_local.stack_median.values()) - cumulated[0], 
                                       sum(speed_test_remote.stack_median.values()) - cumulated[1],
                                       0], width=0.5, bottom=cumulated))
    ax[1][0].set_ylabel('Median of Process Time [s]')
    ax[1][0].set_xticks(indexes)
    ax[1][0].set_xticklabels((speed_test_local.legend, speed_test_remote.legend, speed_test_remote_5G.legend))
    ax[1][0].legend((bar for bar in bars), speed_test_remote.some_keys + ['others'])
    # xx
    # xo
    # ax[1][1]
    plt.show()

if __name__ == '__main__':
    main()
