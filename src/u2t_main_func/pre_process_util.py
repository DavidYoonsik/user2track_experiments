import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from collections import Counter, defaultdict
from sklearn.preprocessing import normalize

from u2t_util.aws_util import download_w2v_from_s3, download_data_from_s3


def prep_dataset(path_dict, track_cnt_threshold, data_max_length, unk_track):
    embedding = dict()
    # Download w2v_dic, w2v_mat .npy file
    if os.path.isfile(path_dict['w2v_dic_path']) and os.path.isfile(path_dict['w2v_mat_path']):
        w2v_dic = np.load(path_dict['w2v_dic_path'], allow_pickle=True)
        w2v_mat = normalize(np.load(path_dict['w2v_mat_path'], allow_pickle=True))
        w2v_key = list(str(k) for k in w2v_dic.item().keys())
        w2v_result = pd.DataFrame(w2v_mat, index=w2v_key)
    else:
        download_w2v_from_s3(path_dict)
        w2v_dic = np.load(path_dict['w2v_dic_path'], allow_pickle=True)
        w2v_mat = normalize(np.load(path_dict['w2v_mat_path'], allow_pickle=True))
        w2v_key = list(str(k) for k in w2v_dic.item().keys())
        w2v_result = pd.DataFrame(w2v_mat, index=w2v_key)

    # Download meta_dict, meta_mat
    if os.path.isfile(path_dict['meta_data_path']):
        meta_df = pd.read_pickle(path_dict['meta_data_path'])
    else:
        bucket = path_dict['hive_meta_path']['bucket']
        key = f"database/{path_dict['hive_meta_path']['database']}/{path_dict['hive_meta_path']['table']}"
        meta_df = download_data_from_s3(bucket, key)
        meta_df.to_pickle(path_dict['meta_data_path'])

    meta_df = meta_df.set_index('track_id')

    key_dict = dict()
    for i, k in enumerate(meta_df.index):
        key_dict[k] = i
    np.save(path_dict['meta_dic_path'], np.array(key_dict))
    np.save(path_dict['meta_mat_path'], meta_df.to_numpy())
    
    meta_dic = np.load(path_dict['meta_dic_path'], allow_pickle=True)
    meta_mat = normalize(np.load(path_dict['meta_mat_path'], allow_pickle=True))
    meta_key = list(str(k) for k in meta_dic.item().keys())
    meta_result = pd.DataFrame(meta_mat, index=meta_key)
    
    # Download train data
    train_tmp = None
    if os.path.isfile(path_dict['train_data_path']):
        train_tmp = pd.read_pickle(path_dict['train_data_path'])
    else:
        bucket = path_dict['hive_train_path']['bucket']
        key = f"database/{path_dict['hive_train_path']['database']}/{path_dict['hive_train_path']['table']}"
        train_tmp = download_data_from_s3(bucket, key)
        train_tmp.to_pickle(path_dict['train_data_path'])
    train_tmp = pd.read_pickle(path_dict['train_data_path'])

    # Download inference data
    infer_tmp = None
    if os.path.isfile(path_dict['infer_data_path']):
        infer_tmp = pd.read_pickle(path_dict['infer_data_path'])
    else:
        bucket = path_dict['hive_infer_path']['bucket']
        key = f"database/{path_dict['hive_infer_path']['database']}/{path_dict['hive_infer_path']['table']}"
        infer_tmp = download_data_from_s3(bucket, key)
        infer_tmp.to_pickle(path_dict['infer_data_path'])
    infer_tmp = pd.read_pickle(path_dict['infer_data_path'])
    
    
    # simple test
    mode = int(os.environ.get('PROD', 0))
    if mode == 0:
        train_tmp = train_tmp.head(300000)
        infer_tmp = infer_tmp.head(300000)
    
    embedding['w2v_vector'] = w2v_result
    embedding['meta_vector'] = meta_result
    
    cols = [c for c in train_tmp.columns if '_tracks' in c]
    counter = Counter('|'.join([train_tmp[c].str.cat(sep='|') for c in cols]).split('|'))
    
    track_list = [k for k, v in counter.most_common() if v >= track_cnt_threshold]
    track_index = list(set(embedding['w2v_vector'].index) & set(track_list))
    track_index = [str(i) for i in sorted([int(w) for w in track_index])]
    track_index = pd.DataFrame([unk_track] + track_index).set_index(0)
    
    embedding['w2v_vector2'] = pd.merge(track_index, embedding['w2v_vector'],
                                        how='left',
                                        left_index=True, right_index=True).fillna(0.0).values
    
    embedding['meta_vector2'] = pd.merge(track_index, embedding['meta_vector'],
                                         how='left',
                                         left_index=True, right_index=True).fillna(0.0).values
    
    def _default_unk_index():
        return 0
    
    def _default_unk_word():
        return unk_track
    
    def word2index(tracks):
        return [track_to_index[x] for x in tracks]
    
    def index2word(tracks):
        return [index_to_track[x] for x in tracks]
    
    track_to_index = defaultdict(_default_unk_index)
    track_to_index.update({t: i for i, t in enumerate(track_index.index)})
    
    index_to_track = defaultdict(_default_unk_word)
    index_to_track.update({t: i for i, t in track_to_index.items()})
    
    train_tmp2 = train_tmp.copy()
    
    train_tmp2['x_play_tracks'] = train_tmp2['x_play_tracks'].apply(
            lambda x: word2index(x.split('|')[:data_max_length]))
    train_tmp2['x_skip_tracks'] = train_tmp2['x_skip_tracks'].apply(
            lambda x: word2index(x.split('|')[:data_max_length]))
    train_tmp2['y_play_tracks'] = train_tmp2['y_play_tracks'].apply(
            lambda x: word2index(x.split('|')[:data_max_length]))
    
    infer_tmp2 = infer_tmp.copy()
    
    infer_tmp2['x_play_tracks'] = infer_tmp2['x_play_tracks'].apply(
            lambda x: word2index(x.split('|')[:data_max_length]))
    infer_tmp2['x_skip_tracks'] = infer_tmp2['x_skip_tracks'].apply(
            lambda x: word2index(x.split('|')[:data_max_length]))
    
    freq_counter = counter.copy()
    freq_result = {track_to_index[str(k)]: v for k, v in freq_counter.items() if str(k) in track_index.index}
    freq_result_ = sorted(freq_result.items(), key=lambda x: x[0], reverse=False)
    freq_result_.insert(0, (0, 0.))
    freq_prob = np.power(np.array(freq_result_).astype("int32")[:, 1], 3 / 4) / np.sum(
            np.power(np.array(freq_result_).astype("int32")[:, 1], 3 / 4))
    freq_prob_ = list(freq_prob)
    
    uniq_idx = train_tmp2['character_no'].unique()
    random.shuffle(uniq_idx)
    n_train = int(len(uniq_idx) * 0.8)
    train_idx, test_idx = uniq_idx[:n_train], uniq_idx[n_train:]
    x_train, x_test = [train_tmp2[train_tmp2['character_no'].isin(idx)].sample(frac=1) for idx in (train_idx,
                                                                                                   test_idx)]
    
    character_train = x_train.character_no.values
    character_test = x_test.character_no.values
    
    print("x_play data padding... step01")
    x_play_train = tf.keras.preprocessing.sequence.pad_sequences(x_train.x_play_tracks, maxlen=data_max_length,
                                                                 dtype='int32',
                                                                 padding='post', truncating='post', value=0)
    y_play_train = tf.keras.preprocessing.sequence.pad_sequences(x_train.y_play_tracks, maxlen=data_max_length,
                                                                 dtype='int32',
                                                                 padding='post', truncating='post', value=0)
    
    print("x_skip data padding... step02")
    x_skip_train = tf.keras.preprocessing.sequence.pad_sequences(x_train.x_skip_tracks, maxlen=data_max_length,
                                                                 dtype='int32',
                                                                 padding='post', truncating='post', value=0)
    x_skip_test = tf.keras.preprocessing.sequence.pad_sequences(x_test.x_skip_tracks, maxlen=data_max_length,
                                                                dtype='int32',
                                                                padding='post', truncating='post', value=0)
    
    print("y_play padding... step03")
    x_play_test = tf.keras.preprocessing.sequence.pad_sequences(x_test.x_play_tracks, maxlen=data_max_length,
                                                                dtype='int32',
                                                                padding='post', truncating='post', value=0)
    y_play_test = tf.keras.preprocessing.sequence.pad_sequences(x_test.y_play_tracks, maxlen=data_max_length,
                                                                dtype='int32',
                                                                padding='post', truncating='post', value=0)
    
    print("infer data padding... step04")
    character_infer = infer_tmp2.character_no.values
    x_play_infer_df = tf.keras.preprocessing.sequence.pad_sequences(infer_tmp2.x_play_tracks, maxlen=data_max_length,
                                                                    dtype='int32', padding='post', truncating='post',
                                                                    value=0)
    x_skip_infer_df = tf.keras.preprocessing.sequence.pad_sequences(infer_tmp2.x_skip_tracks, maxlen=data_max_length,
                                                                    dtype='int32', padding='post', truncating='post',
                                                                    value=0)
    
    train_dt = (x_play_train, x_skip_train, y_play_train, character_train)
    test_dt = (x_play_test, x_skip_test, y_play_test, character_test)
    infer_dt = (x_play_infer_df, x_skip_infer_df, character_infer)
    
    return train_dt, test_dt, infer_dt, freq_prob_, track_index, embedding, track_to_index, index_to_track
