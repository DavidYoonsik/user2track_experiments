import numpy as np
import tqdm
from scipy.special import expit as sigmoid


def u2t_inference(x_play_infer, x_skip_infer, character_infer, u2t_model, session, index_to_track):
    infer_batch_size = 512
    user_vector = []
    
    track_vector = u2t_model.get_layer('track_vector').get_weights()[0]
    track_bias = u2t_model.get_layer('track_bias').get_weights()[0].reshape(1, -1)[0]
    
    for i in tqdm.tqdm(range(0, len(x_play_infer) // infer_batch_size + 1), position=0, mininterval=10):
        sp, ep = i * infer_batch_size, (i + 1) * infer_batch_size
        res = session.run(
                u2t_model.get_layer('user_vector').output,
                feed_dict={
                    u2t_model.get_layer('x_play').input: x_play_infer[sp:ep],
                    u2t_model.get_layer('x_skip').input: x_skip_infer[sp:ep]
                    }
                )
        user_vector.append(res)
    
    user_vector_ = np.vstack(np.array(user_vector))
    
    character_list = ['2783409', '3733392', '5834557', '4194319', '8388625']
    
    for i, v in enumerate(character_infer):
        if v in character_list:
            print(v)
            score = sigmoid(np.dot(track_vector, user_vector_[i]) + track_bias)
            print((np.sort(score)[::-1][:50]))
            print([int(index_to_track[i]) for i in np.argsort(-score) if i != 0][:50])
    
    print(user_vector_.shape, track_vector.shape, track_bias.shape)
    
    return user_vector_, track_vector, track_bias