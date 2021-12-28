import random

import numpy as np
import tensorflow as tf


def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = tf.InteractiveSession(config=config)
    return sess


def set_global_seeds(i):
    """set all seeds"""
    my_seed = i if i is not None else None
    tf.set_random_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)


def save_state(file_name):
    """save trained model"""
    saver = tf.train.Saver()
    sess = get_session()
    saver.save(sess, file_name)


ALREADY_INITIALIZED = set()


def initialize():
    """Initialize all the uninitialized variables in the global scope."""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def swap_flat(arr):
    """
    swap and then flatten axes 0 and 1. """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
