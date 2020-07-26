import json
import boto3
import numpy as np

from u2t_model.create_topk_model import u2t_pred_inference
from u2t_util.aws_util import download_json_from_s3


def upload_metric_data_to_s3(data, bucket, key):
    s3 = boto3.client('s3')
    data_ = json.dumps(data)
    response = s3.put_object(Bucket=bucket, Key=key, Body=data_)
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('Data To S3 Upload Error!')
        return False
    else:
        print('Data To S3 Upload Done!')
        return True


def compute_precision(gt, cand):
    if len(cand) > 0:
        return len(list(set(cand) & set(gt))) / float(len(list(set(cand))))
    else:
        return 0.0


def compute_recall(gt, cand):
    if len(gt) > 0:
        return len(list(set(cand) & set(gt))) / float(len(list(set(gt))))
    else:
        return 0.0


def compute_ndcg(gt, cand, use_rank=True):
    dcg = 0.0
    maxDcg = 0.0
    gt_set = set(gt)
    for idx in range(len(cand)):
        c = cand[idx]
        
        # use (1 / rank) as relevance
        rel = 1.0 / (idx + 1.0) if use_rank else 1.0
        gain = rel / np.log2(idx + 2.0)
        if idx < len(gt_set):
            maxDcg += gain
        
        if c in gt_set:
            dcg += gain
    
    if maxDcg == 0.0:
        return 0.0
    elif dcg == 0.0:
        return 0.0
    else:
        return dcg / maxDcg


def compute_metric(gt, cand, metric):
    if metric == 'Precision':
        return compute_precision(gt, cand)
    elif metric == 'Recall':
        return compute_recall(gt, cand)
    elif metric == 'NDCG_Rank':
        return compute_ndcg(gt, cand, True)


def init_dict(key1, key2, default=0.0):
    result = dict()
    for k1 in key1:
        result[k1] = dict()
        if key2 is not None:
            for k2 in key2:
                result[k1][k2] = default
    
    return result


def compute_avg_metrics(char_no_metric, candidate_metric, score_metric, gt_metric, prefix='Average', topk=50,
                        rep_score=0.8):
    metrics = ['Precision', 'Recall', 'NDCG_Rank']
    probs = rep_score
    
    metric_dict = init_dict(probs, metrics, 0.0)
    counter_dict = init_dict(probs, metrics, 0.0)
    result_dict = init_dict(probs, None, 0.0)
    
    for i, (char_no_metric_, candidate_metric_, score_metric_, gt_metric_) in enumerate(
            zip(char_no_metric, candidate_metric, score_metric, gt_metric)):
        if i % 3000 == 0:
            print('Processing line {}'.format(i))
        
        gt = gt_metric_
        cand = candidate_metric_[:topk]
        scores = score_metric_
        
        try:
            assert len(gt) > 0
        except:
            raise ValueError('ground_truth는 0개보다 많아야 함')
        
        for p in probs:
            cand_p = []
            for c, s in zip(cand, scores):
                if s < p:
                    break
                cand_p.append(c)
            if len(cand_p) > 0:
                selected_metrics = metrics  # else [m for m in metrics if m not in ['Precision','NDCG_Rank',
            else:
                continue
            
            for m in selected_metrics:
                inner_res = compute_metric(gt, cand_p, m)
                if inner_res == 0.0:
                    continue
                else:
                    counter_dict[p][m] += 1.0
                    metric_dict[p][m] += inner_res

    # aggregate
    for p in probs:
        for m in metrics:
            m_ = '{} {}@{}'.format(prefix, m, topk)
            try:
                result_dict[p][m_] = metric_dict[p][m] / counter_dict[p][m]
            except ZeroDivisionError as e:
                print(e)
    
    return result_dict


def merge_metrics(model_nm, metrics_list):
    result = {'model_nm': model_nm}
    metrics = dict()
    
    try:
        for m in metrics_list:
            for k, v in m.items():
                precision = 0
                recall = 0
                num = 0
                if k not in metrics.keys():
                    metrics[k] = dict()
                for kk, vv in v.items():
                    if 'Average Precision@' in kk:
                        num = kk.split('@')[1]
                        precision = vv
                    elif 'Average Recall@' in kk:
                        recall = vv
                    metrics[k].update({kk: vv})
                
                tmp_key = 'Average F1_Score@' + num
                tmp_value = 2 * (precision * recall) / (precision + recall)
                metrics[k].update({tmp_key: tmp_value})
        
        result['metrics'] = metrics
    except ZeroDivisionError as e:
        print(e)
    
    return result


def u2t_metric(x_play_test, x_skip_test, y_play_test, character_test, u2t_model, session, track_index, top_k):
    limit_row = 20000
    infer_batch_size = 128
    pred_inference_top_k = u2t_pred_inference(u2t_model, track_index, top_k)
    
    gt_metric, candidate_metric, score_metric = None, [], []
    gt_metric = y_play_test
    
    character_no = character_test[:limit_row]
    x_play_ = x_play_test[:limit_row]
    x_skip_ = x_skip_test[:limit_row]
    
    for i in range(0, 100):
        sp, ep = i * infer_batch_size, (i + 1) * infer_batch_size
        result = session.run(pred_inference_top_k, feed_dict={
            u2t_model.get_layer('x_play').input: x_play_[sp:ep],
            u2t_model.get_layer('x_skip').input: x_skip_[sp:ep]
            })
        
        score_metric.append(result.values)
        candidate_metric.append(result.indices)
    
    candidate_metric_ = np.vstack(np.array(candidate_metric))
    score_metric_ = np.around(np.vstack(np.array(score_metric)), 4)
    
    rep_score = [0.6, 0.7, 0.8, 0.9]
    metrics = [
        compute_avg_metrics(character_no, candidate_metric_, score_metric_, gt_metric, topk=topk,
                            rep_score=rep_score)
        for topk in [50, 100, 250, 500]]
    
    result = merge_metrics(model_nm='FLO_U2T_METRICS', metrics_list=metrics)
    print(result)
    
    return result


def u2t_metric_upload(metric_result):
    metric_bucket = 'flo-tmp'
    metric_key = 'database/flo_tmp/tmp_metric'
    
    res = upload_metric_data_to_s3(metric_result, metric_bucket, metric_key)
    if res:
        print('Metric Upload To S3 Succeeded...')
    else:
        print('Metric Upload To S3 Failed...')


def s3_mc_for_metric(bucket, key, m_ip, m_port):
    from pymemcache.client import Client

    key_ = f'database/flo_tmp/{key}'
    data = download_json_from_s3(bucket, key_)

    data_ = dict()
    data_['model_nm'] = json.dumps(data['model_nm'])
    data_['metrics'] = json.dumps(data['metrics'])

    client = Client(server=(m_ip, m_port))
    response = client.set_many(data, noreply=False)

    if len(response) == 0:
        print("Push to Memcached Succeeded...")
    else:
        raise Exception('Push to Memcached Failed...')