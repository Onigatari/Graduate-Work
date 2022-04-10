import tensorflow as tf
import pickle as pkl
import string

import re, os
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from keras.models import Model, load_model
from keras.layers import LSTM, GRU, Input, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences

start_target = "<sos>"
end_target = "<eos>"

HIDDEN_DIM = 256

type_mode = {
    "0" : "LSTM-[rus-eng]-[Epochs=1]-[LossFunction=categorical_crossentropy]",
    "1" : "rus-oss.txt-180"
}


class Encoder(tf.keras.Model):
    """
        Энкодер
    """

    def __init__(self, vocab_size_input, HIDDEN_DIM):
        super(Encoder, self).__init__()

        self.inputs = Input(shape=(None,), name="encoder_inputs")
        self.embedding = Embedding(vocab_size_input, HIDDEN_DIM, mask_zero=True, name="encoder_embedding")(self.inputs)

        self.encoder = LSTM(HIDDEN_DIM, return_state=True, name="encoder_lstm")
        self.outputs, state_h, state_c = self.encoder(self.embedding)
        self.states = [state_h, state_c]

    @staticmethod
    def getModel(model):
        inputs_inf = model.input[0]
        outputs_inf, inf_state_h, inf_state_c = model.layers[4].output
        inf_states = [inf_state_h, inf_state_c]

        return Model(inputs_inf, inf_states, name='Encoder')


class Decoder(tf.keras.Model):
    """
        Декодер
    """

    def __init__(self, vocab_size_output, HIDDEN_DIM, encoder_states):
        super(Decoder, self).__init__()

        self.inputs = Input(shape=(None,), name="decoder_inputs")
        self.embedding = Embedding(vocab_size_output, HIDDEN_DIM, mask_zero=True, name="decoder_embedding")(self.inputs)

        self.decoder = LSTM(HIDDEN_DIM, return_sequences=True, return_state=True, name="decoder_lstm")
        self.outputs, _, _ = self.decoder(self.embedding, initial_state=encoder_states)
        self.dense = Dense(vocab_size_output, activation='softmax', name="dense_lstm")
        self.outputs = self.dense(self.outputs)

    @staticmethod
    def getModel(model):
        state_h_input = Input(shape=(HIDDEN_DIM,))
        state_c_input = Input(shape=(HIDDEN_DIM,))
        state_input = [state_h_input, state_c_input]

        input_inf = model.input[1]
        emb_inf = model.layers[3](input_inf)
        lstm_inf = model.layers[5]
        output_inf, state_h_inf, state_c_inf = lstm_inf(emb_inf, initial_state=state_input)
        state_inf = [state_h_inf, state_c_inf]
        dense_inf = model.layers[6]
        output_final = dense_inf(output_inf)

        return Model([input_inf] + state_input, [output_final] + state_inf, name='Decoder')

def load_model_and_weights(path):
    json_file = open(f'static/date/{path}.json')
    loaded_model_json = json_file.read()
    json_file.close()
    model_loaded = model_from_json(loaded_model_json)

    model_loaded.load_weights(f'static/date/{path}-[weight].h5')

    return model_loaded

def load_tokenizers(path):
    with open(f'static/date/{path}-[input].pkl', 'rb') as f:
        tokenizer_input = pkl.load(f)
    with open(f'static/date/{path}-[output].pkl', 'rb') as f:
        tokenizer_output = pkl.load(f)

    return tokenizer_input, tokenizer_output

def predict(input_seq, encoder_model, decoder_model, tokenizer_output, reverse_word_map_target):
    state_values_encoder = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_output.word_index[start_target]
    stop_condition = False
    decoder_sentance = ''

    while not stop_condition:
        sample_word, decoder_h, decoder_c = decoder_model.predict([target_seq] + state_values_encoder)
        sample_word_index = np.argmax(sample_word[0, -1, :])
        decoder_word = reverse_word_map_target[sample_word_index]
        decoder_sentance += ' ' + decoder_word
        if (decoder_word == end_target or
                len(decoder_sentance) > 70):
            stop_condition = True
        target_seq[0, 0] = sample_word_index
        state_values_encoder = [decoder_h, decoder_c]
    return decoder_sentance

def translation(sentance, type):
    path = type_mode[type]
    model = load_model_and_weights(path)
    tokenizer_input, tokenizer_output = load_tokenizers(path)

    encoder_model = Encoder.getModel(model)
    decoder_model = Decoder.getModel(model)

    reverse_word_map_target = dict(map(reversed, tokenizer_output.word_index.items()))

    input_seq = tokenizer_input.texts_to_sequences([sentance])
    pad_sequence = pad_sequences(input_seq, maxlen=30, padding='post')
    predicted_target = predict(pad_sequence, encoder_model, decoder_model, tokenizer_output, reverse_word_map_target)

    hypothetic = str(predicted_target[:-5]).strip().capitalize()
    return hypothetic
