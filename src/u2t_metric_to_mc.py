import argparse

from u2t_util.conf_util import init_config
from u2t_util.metric_util import s3_mc_for_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yyyymmdd', '-y', help='e.g., YYYYMMDD', default='20200202', required=True)
    parser.add_argument('--config_path', '-c', help='e.g., ./config.yml', required=True)
    args = parser.parse_args()

    cf_config = init_config(args.config_path)
    memCache_ip, memCache_port = cf_config['memcache']['ip'], cf_config['memcache']['port']

    print('#####' * 10)
    print('start to upload metric to memCache...')
    bucket = 'flo-tmp'
    key = 'database/flo_tmp/tmp_metric'
    s3_mc_for_metric(bucket, key, memCache_ip, memCache_port)
    print('end to upload metric to memCache...')
