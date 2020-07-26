import numpy as np
import argparse
import os
import glob
import tensorflow as tf

from tensorflow.keras import backend as K

from u2t_util.aws_util import export_npy, upload_file_to_s3
from u2t_main_func.inference_util import u2t_inference
from u2t_main_func.pre_process_util import prep_dataset
from u2t_main_func.train_util import u2t_train
from u2t_util.candidate_util import u2t_candidates
from u2t_util.conf_util import init_config
from u2t_util.metric_util import u2t_metric, u2t_metric_upload

np.random.seed(7)
unk_track = '<UKN>'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yyyymmdd', '-y', help='e.g., YYYYMMDD', default='20200202', required=True)
    parser.add_argument('--config_path', '-c', help='e.g., ./config.yml', required=True)
    args = parser.parse_args()
    print(args.yyyymmdd, args.config_path)
    
    cf_config = init_config(args.config_path)
    tf.compat.v1.logging.set_verbosity(cf_config['system_logging'])
    epochs = cf_config['epochs']
    verboses = cf_config['verboses']
    batch_size = cf_config['batch_size']
    lr = cf_config['lr']
    num_inputs = cf_config['num_inputs']
    num_outputs = cf_config['num_outputs']
    neg_k = cf_config['neg_k']
    top_k = cf_config['top_k']
    track_cnt_threshold = cf_config['track_cnt_threshold']
    
    path_dict = {
        's3_path'         : cf_config['s3'],
        'hive_train_path' : cf_config['hive']['train'],
        'hive_infer_path' : cf_config['hive']['infer'],
        'hive_meta_path'  : cf_config['hive']['meta'],
        'hive_metric_path': cf_config['hive']['metric'],
        'w2v_dic_path'    : cf_config['w2v_dic_path'],
        'w2v_mat_path'    : cf_config['w2v_mat_path'],
        'meta_dic_path'   : cf_config['meta_dic_path'],
        'meta_mat_path'   : cf_config['meta_mat_path'],
        'meta_data_path'  : cf_config['meta_data_path'],
        'infer_data_path' : cf_config['infer_data_path'],
        'train_data_path' : cf_config['train_data_path'],
        'demo_data_path'  : cf_config['demo_data_path'],
        }
    
    data_max_length = num_inputs

    print('#####' * 10)
    print('start to prepare data-set...')
    
    
    train_dt, test_dt, infer_dt, freq_prob, track_index, \
    embedding, track_to_index, index_to_track = prep_dataset(path_dict, track_cnt_threshold, data_max_length,
                                                             unk_track)
    print('end to prepare data-set...')
    
    x_play_train, x_skip_train, y_play_train, character_train = train_dt
    x_play_test, x_skip_test, y_play_test, character_test = test_dt
    x_play_infer, x_skip_infer, character_infer = infer_dt

    print('#####' * 10)
    print('start to train...')
    u2t_model, train_generator, validation_generator, callbacks = u2t_train(x_play_train, x_skip_train, y_play_train,
                                                                            x_play_test, x_skip_test, y_play_test,
                                                                            track_index, batch_size, num_inputs,
                                                                            num_outputs, neg_k, lr, embedding, freq_prob)
    
    hist = u2t_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(x_play_train) // batch_size,
            validation_data=validation_generator,
            validation_steps=len(x_play_test) // batch_size,
            epochs=epochs,
            verbose=verboses,
            callbacks=callbacks)
    print(hist.history)
    print('end to train...')

    print('#####' * 10)
    print('start to inference...')
    user_vector_, track_vector, track_bias = u2t_inference(x_play_infer, x_skip_infer, character_infer, u2t_model,
                                                           K.get_session(), index_to_track)
    print('end to inference...')

    print('#####' * 10)
    print('start to save *.npy...')
    export_npy(track_vector[1:], list(index_to_track.values())[1:],
               os.path.join(os.path.join(path_dict['demo_data_path'], 'item_weight')))
    export_npy(track_bias[1:], list(index_to_track.values())[1:],
               os.path.join(os.path.join(path_dict['demo_data_path'], 'item_bias')))
    export_npy(user_vector_, character_infer,
               os.path.join(os.path.join(path_dict['demo_data_path'], 'user_vec')))
    print('end to save *.npy...')
    print('#####' * 10)
    
    print('start to upload *.npy...')
    latest_npy = [os.path.basename(i) for i in glob.glob(os.path.join(path_dict['demo_data_path'], '*.npy'))]
    for npy_files in latest_npy:
        bucket = path_dict['s3_path']['bucket']
        key = f"{path_dict['s3_path']['key']}/latest/{npy_files}"
        res = upload_file_to_s3(os.path.join(path_dict['demo_data_path'], npy_files), bucket, key)
        print(f"upload... {bucket}/{key} --> {npy_files}")
    print('end to upload *.npy...')
    
    print('#####' * 10)
    print('start to generate candidates.json...')
    u2t_candidates(x_play_infer, x_skip_infer, character_infer, u2t_model, K.get_session(), track_index,
                   index_to_track, top_k)
    print('end to generate candidates.json...')
    
    print('#####' * 10)
    print('start to upload candidates.json...')
    bucket = path_dict['s3_path']['bucket']
    key = f"{path_dict['s3_path']['key']}/latest/{cf_config['candidates_file']}"
    res = upload_file_to_s3(cf_config['candidates_path'], bucket, key)
    print('end to upload candidates.json...')
    
    print('#####' * 10)
    print('start to generate metric...')
    metric_result = u2t_metric(x_play_test, x_skip_test, y_play_test, character_test, u2t_model, K.get_session(),
                               track_index, top_k)
    print('end to generate metric...')

    print('#####' * 10)
    print('start to upload metric...')
    bucket = path_dict['hive_metric_path']['bucket']
    key = f"database/{path_dict['hive_metric_path']['database']}/{path_dict['hive_metric_path']['table']}"
    u2t_metric_upload(metric_result, bucket, key)
    print('end to upload metric')
