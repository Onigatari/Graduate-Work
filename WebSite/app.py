from flask import Flask, render_template, request
import translate, os

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    input_seq = ""
    output_seq = None
    if request.method == 'POST' and request.form['input_sequence']:
        output_seq = translate.predict(request.form['input_sequence'], request.form['temp_model'])
        input_seq = request.form['input_sequence']
    return render_template('index.html', output_seq=output_seq, input_seq=input_seq)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)