import pandas as pd
from sklearn.model_selection import train_test_split
import string
from string import digits
import re
import os
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Input, Dense, Embedding
from keras.models import Model, load_model

from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

import pickle as pkl
import numpy as np

start_target = "sos"
end_target = "eos"
HIDDEN_DIM = 50
type_mode = {
    "0" : "rus-eng.txt-120",
    "1" : "rus-oss.txt-180"
}

def load_model(path):
    json_file = open(f'static/date/{path}-model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_loaded = model_from_json(loaded_model_json)

    model_loaded.load_weights(f'static/date/{path}-model_weight.h5')

    encoder_inputs_inf = model_loaded.input[0]
    encoder_outputs_inf, inf_state_h, inf_state_c = model_loaded.layers[4].output
    encoder_inf_states = [inf_state_h, inf_state_c]
    encoder_model = Model(encoder_inputs_inf, encoder_inf_states)

    decoder_state_h_input = Input(shape=(HIDDEN_DIM,))
    decoder_state_c_input = Input(shape=(HIDDEN_DIM,))
    decoder_state_input = [decoder_state_h_input, decoder_state_c_input]

    decoder_input_inf = model_loaded.input[1]
    decoder_emb_inf = model_loaded.layers[3](decoder_input_inf)
    decoder_lstm_inf = model_loaded.layers[5]
    decoder_output_inf, decoder_state_h_inf, decoder_state_c_inf = decoder_lstm_inf(decoder_emb_inf,
                                                                                    initial_state=decoder_state_input)
    decoder_state_inf = [decoder_state_h_inf, decoder_state_c_inf]
    dense_inf = model_loaded.layers[6]
    decoder_output_final = dense_inf(decoder_output_inf)

    decoder_model = Model([decoder_input_inf] + decoder_state_input, [decoder_output_final] + decoder_state_inf)

    return encoder_model, decoder_model


def load_tokenizer(path):
    with open(f'static/date/{path}-tokenizer_input.pkl', 'rb') as f:
        tokenizer_input = pkl.load(f)
    with open(f'static/date/{path}-tokenizer_target.pkl', 'rb') as f:
        tokenizer_target = pkl.load(f)

    reverse_word_map_input = dict(map(reversed, tokenizer_input.word_index.items()))
    reverse_word_map_target = dict(map(reversed, tokenizer_target.word_index.items()))

    return reverse_word_map_input, reverse_word_map_target, tokenizer_input, tokenizer_target


def decode_seq(input_seq, encoder_model, decoder_model, tokenizer_target, reverse_word_map_target):
    state_values_encoder = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_target.word_index[start_target]
    stop_condition = False
    decoder_sentence = ''

    while not stop_condition:
        sample_word, decoder_h, decoder_c = decoder_model.predict([target_seq] + state_values_encoder)
        sample_word_index = np.argmax(sample_word[0, -1, :])
        decoder_word = reverse_word_map_target[sample_word_index]
        decoder_sentence += ' ' + decoder_word
        if decoder_word == end_target or len(decoder_sentence) > 70:
            stop_condition = True
        target_seq[0, 0] = sample_word_index
        state_values_encoder = [decoder_h, decoder_c]

    return decoder_sentence

def predict(sentance, type):
    path = type_mode[type]
    encoder_model, decoder_model = load_model(path)
    reverse_word_map_input, reverse_word_map_target, tokenizer_input, tokenizer_target = load_tokenizer(path)

    input_seq = tokenizer_input.texts_to_sequences([sentance])
    pad_sequence = pad_sequences(input_seq, maxlen=30, padding='post')
    predicted_target = decode_seq(pad_sequence, encoder_model, decoder_model, tokenizer_target, reverse_word_map_target)
    result = str(predicted_target[:-3]).strip().capitalize()
    return result
