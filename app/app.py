from flask import Flask, render_template, request
import time, os

from model.tokenizer import *
from model import InferenceModel


model_path = 'model/storage'
attn_model = 'dot'
#``attn_model = 'general'``
#``attn_model = 'concat'``
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
num_word = 18709
backend = os.getenv('INFERENCE_BACKEND') if os.getenv('INFERENCE_BACKEND') else 'pt'

model =  InferenceModel(model_path, attn_model, hidden_size, num_word, encoder_n_layers, decoder_n_layers, backend)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    return get_chat_response(request.form["msg"])

def get_chat_response(text):
    start_time = time.time()
    response = model(text)
    end_time = time.time()
    print(f"processing time: {(end_time-start_time) * 1000} ms")
    
    return response

if __name__ == '__main__':
    app.run(port=9997, host = '0.0.0.0')