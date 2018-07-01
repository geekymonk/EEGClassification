import numpy as np
import h5py
import tensorflow as tf


def import_and_split():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in np.arange(9):
        filename = 'project_datasets/A0{}T_slice.mat'.format(i+1)
        A01T = h5py.File(filename, 'r')
        X_temp = np.copy(A01T['image'])
        y_temp = np.copy(A01T['type'])
        y_temp = y_temp[0,0:X_temp.shape[0]:1]
        a = np.argwhere(np.isnan(X_temp))
        c = np.unique(a[:,0])
        X_temp = np.delete(X_temp,c,0)
        y_temp = np.delete(y_temp,c,0)
        y_temp -= 769
        y_temp = y_temp.astype(int)

        indices = np.arange(len(y_temp))
        np.random.shuffle(indices)

        X_test.append(X_temp[indices[:50]].copy())
        y_test.append(y_temp[indices[:50]].copy())

        X_train.append(X_temp[indices[50:]].copy())
        y_train.append(y_temp[indices[50:]].copy())

    return X_train, y_train, X_test, y_test

def partition_train_val(X_train,y_train,val_frac):
    X_val = []
    y_val = []
    for i in range(len(X_train)):
        val_size = round(val_frac*len(y_train[i]))
        indices = np.arange(len(y_train[i]))
        #np.random.shuffle(indices)
        X_val.append(X_train[i][indices[:val_size]].copy())
        y_val.append(y_train[i][indices[:val_size]].copy().astype(int))
        
        X_train[i] = X_train[i][indices[val_size:]]
        y_train[i] = y_train[i][indices[val_size:]].astype(int)
        
    return X_train, y_train, X_val, y_val




def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(5000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

