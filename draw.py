import argparse
from datetime import datetime, timezone, timedelta
from copy import deepcopy
import sqlite3
from statistics import mean, median, stdev
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pandas as pd
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser(
    prog='yolo_api',
    usage='python yolo_api.py -g 1 -p <api port>',
    description='',
    add_help=True,
    )
parser.add_argument('-g', '--local_gpu', action='store_true', help='show measurements with GPU for local as well')
args = parser.parse_args()

sns.set()
sns.set_style('whitegrid', {'grid.linestyle': '--'})
sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2})

# speed02(event_number int, cap_read float, cv2_resize float, image_encode float, image_post float, object_detection float, 
#         bbox_draw float, textbox_draw float, text_put float, cv2_imshow float, environment text)
conn_sql = sqlite3.connect('speed.db')
cursor = conn_sql.cursor()
df = pd.read_sql_query('select * from speed02;', conn_sql)
# speed_server(event_number int, image_decode float, image_transform float, object_detection float, pack_result float)
conn_sql_server = sqlite3.connect('speed_server.db')
cursor_server = conn_sql_server.cursor()
df_server = pd.read_sql_query('select * from speed_server02;', conn_sql_server)


class SpeedTest:
    """
    """
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
        self.stack_mean = {key: 0 for key in self.keys}
        self.stack_median = {key: 0 for key in self.keys}
        self.stack_stdev = {key: 0 for key in self.keys}
    
    def retrieve_data(self, df, df_server=None):
        pass

    def stack_measurements(self):
        for key in self.keys:
            self.stack_mean[key] = mean(self.measurements[key]) if len(self.measurements[key]) > 0 else 0 
            self.stack_median[key] = median(self.measurements[key]) if len(self.measurements[key]) > 0 else 0 
            self.stack_stdev[key] = stdev(self.measurements[key]) if len(self.measurements[key]) > 0 else 0


class SpeedTestLocal(SpeedTest):
    """
    """
    def __init__(self, arch='CPU', net='Local'):
        super().__init__()
        self.arch = arch
        self.net = net
        self.legend = '({}, {})'.format(arch, net)
    
    def retrieve_data(self, df, df_server=None):
        self.measurements = df.query('environment == "{}"'.format('local' if self.arch == 'CPU' else 'local_gpu'))
        self.stack_measurements(df)
    
    def stack_measurements(self, df):
        df_searched = df.query('environment == "{}"'.format('local' if self.arch == 'CPU' else 'local_gpu'))
        df_mean = df_searched.mean()
        df_median = df_searched.median() 
        df_stdev = df_searched.std() 
        for key in self.keys:
            try:
                self.stack_mean[key] = df_mean[key]
                self.stack_median[key] = df_median[key]
                self.stack_stdev[key] = df_stdev[key] 
            except KeyError:
                continue


class SpeedTestRemote(SpeedTest):
    """
    """
    def __init__(self, arch='GPU', net='WiFi'):
        super().__init__()
        self.arch = arch
        self.net = net
        self.legend = '({}, {})'.format(arch, net)
    
    def retrieve_data(self, df, df_server=None):
        if df_server is None:
            print('argument dl_server is not set')
            return False
        if self.net == 'WiFi':
            df_reduced = df.query('environment == "remote"')
        elif self.net == '5G':
            df_reduced = df.query('environment == "remote_5g"')
        col_list = list(df_server)
        col_list.remove('event_number')
        # retrieve data whose event numbers match on both local and remote
        # もっと効率いいやり方ないかな
        for _, event_number in df_server.event_number.iteritems():
            df_searched = df_reduced.query('event_number == {}'.format(event_number))
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
    ''' main
    '''
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    arch = 'GPU' if args.local_gpu else 'CPU'
    speed_test_local = SpeedTestLocal(arch=arch, net='Local')
    speed_test_local.retrieve_data(df=df)
    speed_test_remote = SpeedTestRemote(arch='GPU', net='WiFi')
    speed_test_remote.retrieve_data(df=df, df_server=df_server)
    speed_test_remote_5g = SpeedTestRemote(arch='GPU', net='5G')
    speed_test_remote_5g.retrieve_data(df=df, df_server=df_server)
    # ox
    # xx
    #sns.distplot(df.image_post - df_server[col_list].sum(axis=1), norm_hist=True, bins=100, ax=ax[0][0], label='HTTP POST (WiFi)')
    bins = np.arange(0, 3, 0.02)
    sns.distplot(speed_test_remote.measurements['http_post'], bins=bins, kde=False, norm_hist=True, ax=ax[0][0], label='HTTP POST ' + speed_test_remote.legend)
    sns.distplot(speed_test_remote_5g.measurements['http_post'], bins=bins, kde=False, norm_hist=True, ax=ax[0][0], label='HTTP POST ' + speed_test_remote_5g.legend)
    sns.distplot(speed_test_remote.measurements['object_detection'], bins=bins, kde=False, norm_hist=True, ax=ax[0][0], label='YOLO Prediction ' + speed_test_remote.legend)
    sns.distplot(speed_test_remote.measurements['image_decode'], bins=bins, kde=False, norm_hist=True, ax=ax[0][0], label='Image Decoding ' + speed_test_remote.legend)
    sns.distplot(speed_test_local.measurements['object_detection'], bins=bins, kde=False, norm_hist=True, ax=ax[0][0], label='YOLO Prediction ' + speed_test_local.legend)
    ax[0][0].set_xlabel('Process Time [s]')
    ax[0][0].set_ylabel('Arbitrary Unit')
    #ax[0][0].set_ylabel('Number of Events')
    #ax[0][0].set_ylim(ymin=0.8)
    ax[0][0].set_yscale('log')
    ax[0][0].set_xlim(xmin=0, xmax=2.8)
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
    for key in speed_test_remote.keys:
        if key in speed_test_remote.some_keys:
            bars.append(ax[1][0].bar(indexes, [speed_test_local.stack_median[key], 
                                               speed_test_remote.stack_median[key], 
                                               speed_test_remote_5g.stack_median[key]], 
                                     width=0.5, bottom=cumulated, label=key))
            cumulated[0] += speed_test_local.stack_median[key]
            cumulated[1] += speed_test_remote.stack_median[key]
            cumulated[2] += speed_test_remote_5g.stack_median[key]
    bars.append(ax[1][0].bar(indexes, [sum(speed_test_local.stack_median.values()) - cumulated[0], 
                                       sum(speed_test_remote.stack_median.values()) - cumulated[1],
                                       sum(speed_test_remote_5g.stack_median.values()) - cumulated[2]],
                                       width=0.5, bottom=cumulated, label='others'))
    ax[1][0].set_ylabel('Median of Process Time [s]')
    ax[1][0].set_xticks(indexes)
    ax[1][0].set_xticklabels((speed_test_local.legend, speed_test_remote.legend, speed_test_remote_5g.legend))
    #ax[1][0].legend((bar for bar in bars), speed_test_remote.some_keys + ['others'])
    ax[1][0].legend()
    # xx
    # xo
    nbars = 2
    indexes = np.arange(nbars)
    bars = []
    cumulated = [0 for _ in indexes]
    for key in speed_test_remote.keys:
        if key in speed_test_remote.some_keys:
            bars.append(ax[1][1].bar(indexes, [speed_test_remote.stack_median[key], speed_test_remote_5g.stack_median[key]], 
                                     width=0.5, bottom=cumulated, label=key))
            cumulated[0] += speed_test_remote.stack_median[key]
            cumulated[1] += speed_test_remote_5g.stack_median[key]
    bars.append(ax[1][1].bar(indexes, [sum(speed_test_remote.stack_median.values()) - cumulated[0],
                                       sum(speed_test_remote_5g.stack_median.values()) - cumulated[1]], 
                                       width=0.5, bottom=cumulated, label='others'))
    ax[1][1].set_ylabel('Median of Process Time [s]')
    ax[1][1].set_xticks(indexes)
    ax[1][1].set_xticklabels((speed_test_remote.legend, speed_test_remote_5g.legend))
    #ax[1][0].legend((bar for bar in bars), speed_test_remote.some_keys + ['others'])
    ax[1][1].legend()
    fig.savefig('plot.png')
    plt.show()

if __name__ == '__main__':
    main()
