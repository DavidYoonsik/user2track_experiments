import numpy as np
import tqdm
import json
import gzip

from u2t_model.create_topk_model import u2t_pred_inference


def u2t_candidates(x_play_infer, x_skip_infer, character_infer, u2t_model, session, track_index, index_to_track, top_k):
    batch_size = 512
    candidate_metric, score_metric = [], []
    pred_inference_top_k = u2t_pred_inference(u2t_model, track_index, top_k)
    
    print('    start to extract candidates...')
    for i in tqdm.tqdm(range(0, len(character_infer) // batch_size + 1), position=0, mininterval=5, maxinterval=20):
        sp, ep = i * batch_size, (i + 1) * batch_size
        result = session.run(pred_inference_top_k, feed_dict={
            u2t_model.get_layer('x_play').input: x_play_infer[sp:ep],
            u2t_model.get_layer('x_skip').input: x_skip_infer[sp:ep]
            })
        
        score_metric.append(np.round(result.values, 4))
        candidate_metric.append(result.indices)
    
    score_metric_ = np.vstack(np.array(score_metric))
    candidate_metric_ = np.vstack(np.array(candidate_metric))
    print('    end to extract candidates...')

    print('    start to write candidates.gz file...')
    with gzip.open('/data01/candidates.gz', 'wb') as f:
        for idx, (tracks, scores) in enumerate(zip(candidate_metric_, score_metric_)):
            candidates_ = []
            scores = scores.astype('float')
            for track, score in zip(tracks, scores):
                if score > 0.6:
                    candidates_.append(
                            {'track_id': int(index_to_track[track]), 'score': score})
                else:
                    break
            
            if len(candidates_) >= 50:
                txt = json.dumps({'character_no': character_infer[idx], 'candidates': candidates_})
                f.write(txt.encode())
                f.write('\n'.encode())
            
            if idx % 10000 == 0:
                print(idx, ' ... ', str(len(candidate_metric_)), end='\r\r')
    print('    end to write candidates.gz file...')
