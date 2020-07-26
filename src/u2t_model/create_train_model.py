import tensorflow as tf


def user2track_model(num_inputs, num_outputs, neg_k, embedding, track_index, lr):
    x_play = tf.keras.Input(shape=(num_inputs,), dtype='int32', name='x_play')
    x_skip = tf.keras.Input(shape=(num_inputs,), dtype='int32', name='x_skip')
    
    cf_embedding = tf.keras.layers.Embedding(embedding['w2v_vector2'].shape[0], embedding['w2v_vector2'].shape[1],
                                             embeddings_initializer=tf.keras.initializers.Constant(
                                                     embedding['w2v_vector2']),
                                             input_length=num_inputs,
                                             trainable=False, name='w2v_vector')
    
    meta_embedding = tf.keras.layers.Embedding(embedding['meta_vector2'].shape[0], embedding['meta_vector2'].shape[1],
                                               embeddings_initializer=tf.keras.initializers.Constant(
                                                       embedding['meta_vector2']),
                                               input_length=num_inputs,
                                               trainable=False, name='meta_vector')
    
    x_play_cf_flatten_i = cf_embedding(x_play)
    x_skip_meta_flatten_i = meta_embedding(x_skip)
    x_play_meta_flatten_i = meta_embedding(x_play)
    
    x_play_cf_flatten = tf.keras.layers.Conv1D(filters=256, kernel_size=100, padding='same', activation=tf.nn.relu)(
            x_play_cf_flatten_i)
    print(x_play_cf_flatten.shape)
    
    x_play_cf_flatten = tf.keras.layers.Conv1D(filters=256, kernel_size=100, padding='same', activation=tf.nn.relu)(
            x_play_cf_flatten)
    
    print(x_play_cf_flatten.shape)
    
    x_play_cf_flatten = tf.keras.layers.Dropout(0.5)(x_play_cf_flatten)
    x_play_cf_flatten = tf.keras.layers.MaxPooling1D(pool_size=5, padding='same')(x_play_cf_flatten)
    x_play_cf_flatten = tf.keras.layers.Flatten()(x_play_cf_flatten)
    x_play_cf_flatten = tf.keras.layers.Dense(512, activation=tf.nn.selu, use_bias=True)(x_play_cf_flatten)
    x_play_cf_flatten = tf.keras.layers.Dense(256, activation=tf.nn.selu, use_bias=True)(x_play_cf_flatten)
    
    x_skip_meta_flatten = tf.keras.layers.Conv1D(filters=32, kernel_size=14, activation=tf.nn.selu, padding='same',
                                                 use_bias=True)(x_skip_meta_flatten_i)
    x_skip_meta_flatten = tf.keras.layers.Conv1D(filters=32, kernel_size=14, activation=tf.nn.selu, padding='same',
                                                 use_bias=True)(x_skip_meta_flatten)
    
    x_skip_meta_flatten = tf.keras.layers.Dropout(0.5)(x_skip_meta_flatten)
    x_skip_meta_flatten = tf.keras.layers.MaxPooling1D(pool_size=5, padding='same')(x_skip_meta_flatten)
    x_skip_meta_flatten = tf.keras.layers.Flatten()(x_skip_meta_flatten)
    x_skip_meta_flatten = tf.keras.layers.Dense(64, activation=tf.nn.selu, use_bias=True)(x_skip_meta_flatten)
    x_skip_meta_flatten = tf.keras.layers.Dense(32, activation=tf.nn.selu, use_bias=True)(x_skip_meta_flatten)
    
    x_play_meta_flatten = tf.keras.layers.Conv1D(filters=32, kernel_size=14, activation=tf.nn.selu, padding='same',
                                                 use_bias=True)(x_play_meta_flatten_i)
    x_play_meta_flatten = tf.keras.layers.Conv1D(filters=32, kernel_size=14, activation=tf.nn.selu, padding='same',
                                                 use_bias=True)(x_play_meta_flatten)
    
    x_play_meta_flatten = tf.keras.layers.Dropout(0.5)(x_play_meta_flatten)
    x_play_meta_flatten = tf.keras.layers.MaxPooling1D(pool_size=5, padding='same')(x_play_meta_flatten)
    x_play_meta_flatten = tf.keras.layers.Flatten()(x_play_meta_flatten)
    x_play_meta_flatten = tf.keras.layers.Dense(64, activation=tf.nn.selu, use_bias=True)(x_play_meta_flatten)
    x_play_meta_flatten = tf.keras.layers.Dense(32, activation=tf.nn.selu, use_bias=True)(x_play_meta_flatten)
    
    user_vector = tf.keras.layers.concatenate([x_play_cf_flatten, x_play_meta_flatten, x_skip_meta_flatten], axis=1)
    
    user_vector = tf.keras.layers.Dropout(0.3)(
            tf.keras.layers.Dense(num_outputs, activation=tf.nn.selu, use_bias=True)(user_vector))
    user_vector = tf.keras.layers.Dense(num_outputs, name='user_vector')(user_vector)
    
    pos_input = tf.keras.Input(shape=(num_inputs,), dtype=tf.int32, name='pos_input')
    neg_input = tf.keras.Input(shape=(num_inputs * neg_k,), dtype=tf.int32, name='neg_input')
    
    track_embedding = tf.keras.layers.Embedding(len(track_index.index), num_outputs, trainable=True,
                                                embeddings_initializer='glorot_uniform',
                                                name='track_vector')
    
    bias_embedding = tf.keras.layers.Embedding(len(track_index.index), 1, trainable=True,
                                               embeddings_initializer='glorot_uniform',
                                               embeddings_regularizer=tf.keras.regularizers.l2(l=0.0001),
                                               name='track_bias')
    
    pos_output = track_embedding(pos_input)
    pos_bias = bias_embedding(pos_input)
    pos_bias = tf.reshape(pos_bias, [-1, num_inputs])
    
    neg_output = track_embedding(neg_input)
    neg_bias = bias_embedding(neg_input)
    neg_bias = tf.reshape(neg_bias, [-1, num_inputs * neg_k])
    
    pos_output = tf.add(
            tf.reduce_mean(tf.matmul(tf.keras.backend.expand_dims(user_vector, 1), pos_output, transpose_b=True),
                           axis=1),
            pos_bias)
    
    neg_output = tf.add(
            tf.reduce_mean(tf.matmul(tf.keras.backend.expand_dims(user_vector, 1), neg_output, transpose_b=True),
                           axis=1),
            neg_bias)
    
    outputs = tf.keras.layers.Concatenate(axis=1, name='outputs')([pos_output, neg_output])
    
    # Model Define
    u2t_model = tf.keras.Model(
            inputs=[x_play, x_skip, pos_input, neg_input],
            outputs=[outputs]
            )
    
    u2t_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='hinge',
            metrics=[tf.keras.metrics.cosine_proximity]
            )
    
    return u2t_model