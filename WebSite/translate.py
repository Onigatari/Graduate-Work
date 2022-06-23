import tensorflow as tf
import tensorflow_text as tf_text
import model

cur_model = ""
reloaded = None

path_model = {
    '0': 'RUS-OSS',
    '1': 'RUS-ENG-MINI',
}


def load_model(type):
    global cur_model
    global reloaded

    if (cur_model != type):
        cur_model = type
        reloaded = tf.saved_model.load(f'static/model/{path_model[type]}')

def translation(input_sequence, type):
    load_model(type)

    input_text = tf.constant([input_sequence])
    result = reloaded.tf_translate(input_text)

    return result['text'][0].numpy().decode().capitalize()