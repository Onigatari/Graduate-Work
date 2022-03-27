from flask import Flask, render_template, request
import translate

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    input_seq = ""
    output_seq = None
    if request.method == 'POST' and request.form['input_sequence']:
        output_seq = translate.predict(request.form['input_sequence'])
        input_seq = request.form['input_sequence']
    return render_template('index.html', output_seq=output_seq, input_seq=input_seq)

if __name__ == '__main__':
    app.run()
