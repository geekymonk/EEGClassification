import os
import shutil
import model_fns
import numpy as np
import tensorflow as tf


def BuildEstimator(ch=25,model=None,subj='all',log_dir=None,learning_rate=0.001,training=True):

  my_config = tf.estimator.RunConfig(save_summary_steps=1)
  eeg_classifier = None

  if os.path.isdir(log_dir) and training:
    shutil.rmtree(log_dir)

  if model=='cnn':
    eeg_classifier = tf.estimator.Estimator(
      model_fn=model_fns.cnn_model_fn, model_dir=log_dir,
      params= {'learning_rate':learning_rate, 'hidden_layers':[50,50], 'dropout':0.3, 'num_classes':4, 
               'filters':128, 'convkernel':(ch,21)}, config=my_config)

  elif model=='rnn':
    eeg_classifier = tf.estimator.Estimator(
      model_fn=model_fns.rnn_model_fn, model_dir=log_dir,
      params= {'learning_rate':learning_rate, 'hidden_layers':[128,128], 'dropout':0.1, 'num_classes':4}, config=my_config)

  elif model=='birnn':
    eeg_classifier = tf.estimator.Estimator(
      model_fn=model_fns.rnn_model_fn_bidirectional, model_dir=log_dir,
      params= {'learning_rate':learning_rate, 'hidden_layers':[128,128], 'num_classes':4}, config=my_config)

  return eeg_classifier







def IndividualEvaluation(eeg_classifier,test_X,test_y):
  eval_results = []
  accuracy = []
  for i in range(len(test_X)):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_X[i]},
      y=test_y[i].astype(int),
      num_epochs=1,
      shuffle=False)
    
    eval_results.append(eeg_classifier.evaluate(input_fn=eval_input_fn))
    accuracy.append(eval_results[-1]['accuracy'])
    
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.vstack(test_X)},
    y=np.hstack(test_y).astype(int),
    num_epochs=1,
    shuffle=False)
  
  avg_eval_results = eeg_classifier.evaluate(input_fn=eval_input_fn)
  avg_accuracy = avg_eval_results['accuracy']
    
  return avg_accuracy,accuracy,eval_results