import numpy as np
import tensorflow as tf


def u2t_pred_inference(u2t_model, track_index, top_k):
    infer_play = np.array((range(len(track_index)))).reshape(1, -1)
    infer_vector = tf.squeeze(tf.keras.backend.constant(infer_play), axis=0)
    
    all_track_inference_vector = u2t_model.get_layer('track_vector')(infer_vector)
    all_track_inference_bias = tf.squeeze(u2t_model.get_layer('track_bias')(infer_vector), 1)
    
    pred_inference = tf.sigmoid(tf.add(tf.reduce_mean(
            tf.matmul(tf.keras.backend.expand_dims(u2t_model.get_layer('user_vector').output, 1),
                      all_track_inference_vector, transpose_b=True), 1),
            all_track_inference_bias))
    
    pred_inference_top_k = tf.math.top_k(pred_inference,
                                         k=top_k,
                                         sorted=True
                                         )
    
    return pred_inference_top_k
