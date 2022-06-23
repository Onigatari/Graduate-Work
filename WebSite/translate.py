import tensorflow as tf
import tensorflow_text as tf_text

cur_model = ""
reloaded = None

path_model = {
    '0': 'RUS-OSS',
    '1': 'RUS-ENG-MINI',
}


def load_model(model_type):
    global cur_model
    global reloaded

    if cur_model != model_type:
        cur_model = model_type
        reloaded = tf.saved_model.load(f'static/model/{path_model[model_type]}')


def translation(input_sequence, model_type):
    load_model(model_type)

    input_text = tf.constant([input_sequence])
    result = reloaded.tf_translate(input_text)

    return result['text'][0].numpy().decode().capitalize()
