import numpy as np
import tensorflow as tf


def rnn_model_fn(features, labels, mode, params):
  """Model function for RNN."""
  # Input Layer
  input_layer = features["x"]

  cell1 = tf.nn.rnn_cell.LSTMCell(num_units = params['hidden_layers'][0])
  outputs, state = tf.nn.dynamic_rnn(cell=cell1,
                                   inputs=input_layer,
                                   dtype=tf.float64,scope='cell1')
  outputs = tf.nn.relu(outputs)  
    
  
  cell2 = tf.nn.rnn_cell.LSTMCell(num_units = params['hidden_layers'][1])
  outputs, state = tf.nn.dynamic_rnn(cell=cell2,
                                   inputs=outputs,
                                   dtype=tf.float64,scope='cell2')
  outputs = tf.nn.relu(outputs)
  outputs = tf.layers.dropout(inputs=outputs, rate=params['dropout'], training=(mode==tf.estimator.ModeKeys.TRAIN))
    
  outputs = tf.contrib.layers.flatten(outputs) 

  # Logits Layer
  logits = tf.layers.dense(inputs=outputs, units=params['num_classes'])

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  #loss = tf.losses.softmax_cross_entropy(
    #onehot_labels=onehot_labels, logits=logits)  
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def rnn_model_fn_bidirectional(features, labels, mode, params):
  """Model function for RNN."""
  # Input Layer
  input_layer = features["x"]

  cell1 = tf.nn.rnn_cell.LSTMCell(num_units = params['hidden_layers'][0])
  cell2 = tf.nn.rnn_cell.LSTMCell(num_units = params['hidden_layers'][1])
  # outputs, state = tf.nn.dynamic_rnn(cell=cell1,
  #                                  inputs=input_layer,
  #                                  dtype=tf.float64)
  # outputs = tf.nn.relu(outputs)  
    
  # # dropout    
  # outputs = tf.layers.dropout(inputs=outputs, rate=0.5, training=(mode==tf.estimator.ModeKeys.TRAIN))
  
  # cell2 = tf.nn.rnn_cell.LSTMCell(name='cell2', num_units = params['hidden_layers'][1])
  outputs_bi, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell1,cell_bw=cell2,
                                   inputs=input_layer,
                                   dtype=tf.float64)

  outputs = tf.nn.relu(tf.concat(outputs_bi, 2))
  outputs = tf.contrib.layers.flatten(outputs) 
  outputs = tf.layers.dropout(inputs=outputs, rate=0.5, training=(mode==tf.estimator.ModeKeys.TRAIN))

  # Logits Layer
  logits = tf.layers.dense(inputs=outputs, units=params['num_classes'])

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  #loss = tf.losses.softmax_cross_entropy(
    #onehot_labels=onehot_labels, logits=logits)  
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    init = tf.contrib.layers.xavier_initializer()
    
    # Input Layer
    input_layer = tf.cast(features['x'],tf.float32)

    # Convolution Layer 1
    conv1 = tf.layers.conv2d(inputs=input_layer,filters = params['filters'], kernel_size=params['convkernel'],
      strides=(1,1), padding='valid', kernel_initializer=init,activation=tf.nn.relu, kernel_regularizer=regularizer)

    #Flatten 
    fc_in = tf.contrib.layers.flatten(conv1)
    dropout1 = tf.layers.dropout(inputs=fc_in, rate=params['dropout'], training=(mode== tf.estimator.ModeKeys.TRAIN))

    #Fully Connected NN
    fc1 = tf.layers.dense(inputs=dropout1, units=params['hidden_layers'][0], activation=tf.nn.relu, kernel_regularizer=regularizer)
    dropout2 = tf.layers.dropout(inputs=fc1, rate=params['dropout'], training=(mode== tf.estimator.ModeKeys.TRAIN))
    fc2 = tf.layers.dense(inputs=dropout2, units= params['hidden_layers'][1],activation=tf.nn.relu, kernel_regularizer=regularizer)
    dropout3 = tf.layers.dropout(inputs=fc2, rate=params['dropout'], training=(mode== tf.estimator.ModeKeys.TRAIN))
 
    outputs = dropout2
    # Logits Layer
    logits = tf.layers.dense(inputs=outputs, units=params['num_classes'])

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + tf.losses.get_regularization_loss()


    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)