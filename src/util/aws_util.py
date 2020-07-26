import os
import json
import tqdm
import boto3
import numpy as np
import io
import pyarrow
import pyarrow.orc
from botocore.exceptions import NoCredentialsError


def upload_file_to_s3(filename, bucket, key):
    try:
        s3 = boto3.resource('s3')
        s3.meta.client.upload_file(filename, bucket, key)
    except FileNotFoundError as e:
        print(e)
        return False
    except NoCredentialsError as n:
        print(n)
        return False
    finally:
        return True
    
    
def upload_data_to_s3(data, bucket, key):
    s3 = boto3.client('s3')
    data_ = json.dumps(data)
    response = s3.put_object(Bucket=bucket, Key=key, Body=data_)
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('Data To S3 Upload Error!')
        return False
    else:
        print('Data To S3 Upload Done!')
        return True
    

def export_npy(in_array, in_key, _path):
    assert len(in_array) == len(in_key)
    key_dict = dict()
    in_array_ = dict()

    i = 0
    for k, arr in zip(in_key, in_array):
        if k not in key_dict:
            key_dict[k] = i
            in_array_[k] = arr
            i += 1

    np.save(_path + '.dic.npy', np.array(key_dict))
    np.save(_path + '.mat.npy', list(in_array_.values()))


def download_data_from_s3(bucket, key):
    s3_cli = boto3.client('s3')
    response = s3_cli.list_objects_v2(Bucket=bucket, Prefix=key)
    keys = [content['Key'] for content in response['Contents'] if content['Key'][-8:] != '_SUCCESS']
    tables = []
    
    with tqdm.tqdm(total=len(keys), position=0, mininterval=5, maxinterval=20) as pbar:
        for key in keys:
            obj = io.BytesIO()
            s3_cli.download_fileobj(bucket, key, obj)
            data = pyarrow.orc.ORCFile(obj)
            tables.append(data.read())
            pbar.update(1)

    meta_df = pyarrow.concat_tables(tables).to_pandas().fillna(0)
    return meta_df


def download_w2v_from_s3(w2v_path):
    s3_cli = boto3.client('s3')
    with open(w2v_path['w2v_dic_path'], 'wb') as f:
        s3_cli.download_fileobj('flo-reco', 'model/log-track2track/latest/log-track2track.dic.npy', f)
    
    with open(w2v_path['w2v_mat_path'], 'wb') as f:
        s3_cli.download_fileobj('flo-reco', 'model/log-track2track/latest/log-track2track.mat.npy', f)
    
    if os.path.exists(w2v_path['w2v_dic_path']) and os.path.exists(w2v_path['w2v_mat_path']):
        return True
    else:
        return False


def download_json_from_s3(bucket, key):
    s3_cli = boto3.client('s3')
    s3_object = s3_cli.get_object(Bucket=bucket, Key=key)
    body = s3_object['Body'].read()
    return json.loads(body)

