"""
Contains various recommondation implementations
all algorithms return a list of tuple (title,movieid)
"""

from logging import error
import pandas as pd
import numpy as np
import re
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Embedding
from keras.layers import LSTM, Bidirectional,AveragePooling1D
from keras.preprocessing import text, sequence
from keras import backend as K 
from keras.models import Sequential

import pickle
import emoji


def predict_toxic(user_input):
    """"""
    with open('data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    MAX_LEN = 120

    model = tf.keras.models.load_model('data/modelemoji.03-0.26.h5')
    def prepare_input(sentence):
        input = sentence.lower()
        input = [re.sub(r"([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", input)]
        input = tokenizer.texts_to_sequences(input)
        input = sequence.pad_sequences(input, maxlen=MAX_LEN)
        return input
    user_input = prepare_input(user_input)
    pred = model.predict(user_input)
        
    return pred

