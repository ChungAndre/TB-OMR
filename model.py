import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


# Attention mechanism
def attention_layer(inputs, attention_dim):
    """
    Attention mechanism layer
    :param inputs: Tensor of shape [time_steps, batch_size, num_features]
    :param attention_dim: Dimension for the attention layer
    :return: Tensor after applying attention weights
    """
    # Hidden representation of inputs
    hidden_rep = tf.layers.dense(inputs, attention_dim, activation=tf.nn.tanh)

    # Attention scores
    attention_scores = tf.layers.dense(hidden_rep, 1, activation=None)  # [time_steps, batch_size, 1]
    attention_scores = tf.squeeze(attention_scores, axis=-1)  # [time_steps, batch_size]

    # Attention weights
    attention_weights = tf.nn.softmax(attention_scores, axis=0)  # Apply softmax over time steps

    # Weighted sum of the input features
    attention_output = inputs * tf.expand_dims(attention_weights, -1)  # [time_steps, batch_size, num_features]

    return attention_output, attention_weights

def leaky_relu(features, alpha=0.2, name=None):
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    return math_ops.maximum(alpha * features, features)



#
# params["height"] = height of the input image
# params["width"] = width of the input image

def default_model_params(img_height, vocabulary_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = 16
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [ [3,3], [3,3], [3,3], [3,3] ]
    params['conv_pooling_size'] = [ [2,2], [2,2], [2,2], [2,2] ]
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size
    return params


def ctc_crnn(params):
    # TODO Assert parameters

    input = tf.compat.v1.placeholder(shape=(None,
                                   params['img_height'],
                                   params['img_width'],
                                   params['img_channels']),  # [batch, height, width, channels]
                            dtype=tf.float32,
                            name='model_input')

    input_shape = tf.shape(input)

    width_reduction = 1
    height_reduction = 1


    # Convolutional blocks
    x = input
    for i in range(params['conv_blocks']):

        x = tf.layers.conv2d(
            inputs=x,
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            padding="same",
            activation=None)

        x = tf.layers.batch_normalization(x)
        x = leaky_relu(x)

        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=params['conv_pooling_size'][i],
                                    strides=params['conv_pooling_size'][i])

        width_reduction = width_reduction * params['conv_pooling_size'][i][1]
        height_reduction = height_reduction * params['conv_pooling_size'][i][0]


    # Prepare output of conv block for recurrent blocks
    features = tf.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
    feature_width = input_shape[2] / width_reduction
    features = tf.reshape(features, tf.stack([tf.cast(feature_width,'int32'), input_shape[0], tf.cast(feature_dim,'int32')]))  # -> [width, batch, features]

    tf.constant(params['img_height'],name='input_height')
    tf.constant(width_reduction,name='width_reduction')


     # Attention Layer
    attention_dim = 128  # Dimension of attention layer
    attention_output, attention_weights = attention_layer(features, attention_dim)


    # Recurrent block
    rnn_keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, name="keep_prob")
    rnn_hidden_units = params['rnn_units']
    rnn_hidden_layers = params['rnn_layers']

    rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_hidden_units), input_keep_prob=rnn_keep_prob)
             for _ in range(rnn_hidden_layers)]),
        tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_hidden_units), input_keep_prob=rnn_keep_prob)
             for _ in range(rnn_hidden_layers)]),
        #features,
        attention_output,  # Apply attention output to RNN
        dtype=tf.float32,
        time_major=True,
    )

    rnn_outputs = tf.concat(rnn_outputs, 2)
    print("RNN outputs shape:", rnn_outputs.shape)
    
    print(attention_output.shape)
    # Fully connected layer

    logits = tf.contrib.layers.fully_connected(
        rnn_outputs,
        params['vocabulary_size'] + 1,  # BLANK
        activation_fn=None,
    )

    tf.compat.v1.add_to_collection("logits",logits) # for restoring purposes

    # CTC Loss computation
    seq_len = tf.compat.v1.placeholder(tf.int32, [None], name='seq_lengths')
    targets = tf.compat.v1.sparse_placeholder(dtype=tf.int32, name='target')
    ctc_loss = tf.compat.v1.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=True)
    loss = tf.reduce_mean(ctc_loss)

    # CTC decoding
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits,seq_len,beam_width=50,top_paths=1,merge_repeated=True)

    return input, seq_len, targets, decoded, loss, rnn_keep_prob, attention_weights
