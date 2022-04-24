from flask import Flask, render_template, request
import translate, os

app = Flask(__name__)

path_model = {
    '0': '1. RUS-OSS GRU-[1024]-EMD_DIM[256]-BatchSize[64]-Epochs[15]',
    '1': '2. RUS-OSS GRU-[1024]-EMD_DIM[256]-BatchSize[64]-Epochs[20]',
    '2': '3. RUS-OSS GRU-[1024]-EMD_DIM[128]-BatchSize[64]-Epochs[15]',
    '3': '4. RUS-OSS GRU-[1024]-EMD_DIM[128]-BatchSize[64]-Epochs[20]',
    '4': '5. RUS-OSS GRU-[512]-EMD_DIM[256]-BatchSize[64]-Epochs[15]',
    '5': '6. RUS-OSS GRU-[512]-EMD_DIM[256]-BatchSize[64]-Epochs[20]',
    '6': '7. RUS-OSS GRU-[512]-EMD_DIM[128]-BatchSize[64]-Epochs[15]',
    '7': '8. RUS-OSS GRU-[512]-EMD_DIM[128]-BatchSize[64]-Epochs[20]',
    '8': '9. RUS-OSS GRU-[1024]-EMD_DIM[256]-BatchSize[32]-Epochs[15]',
    '9': '10. RUS-OSS GRU-[1024]-EMD_DIM[256]-BatchSize[32]-Epochs[20]',
    '10': '11. RUS-OSS GRU-[512]-EMD_DIM[128]-BatchSize[32]-Epochs[15]',
    '11': '12. RUS-OSS GRU-[512]-EMD_DIM[128]-BatchSize[32]-Epochs[20]',
    '12': 'RUS-ENG-MINI GRU-[1024]-EMD_DIM[256]-BatchSize[128]-Epochs[5]'
}


@app.route('/', methods=['POST', 'GET'])
def index():
    input_seq = ""
    output_seq = None
    if request.method == 'POST' and request.form['input_sequence']:
        output_seq = translate.translation(request.form['input_sequence'], request.form['temp_model'])
        input_seq = request.form['input_sequence']
    return render_template('index.html', output_seq=output_seq, input_seq=input_seq, path_model=path_model)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)