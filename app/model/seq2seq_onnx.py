import onnxruntime
import numpy as np
from .utils import to_numpy

import torch 

class EncoderONNX():
    def __init__(self, model_path):
        self.ort_session = onnxruntime.InferenceSession(model_path)

    def __call__(self, input_seq, input_lengths):
        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(input_seq), self.ort_session.get_inputs()[1].name : to_numpy(input_lengths)}
        ort_outs = self.ort_session.run(None, ort_inputs)

        return torch.tensor(ort_outs[0]), torch.tensor(ort_outs[1])

class DecoderONNX():
    def __init__(self, model_path, n_layers = 2):
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.n_layers = n_layers 
        
    def __call__(self, input_step, last_hidden, encoder_outputs):
        ort_inputs = {  self.ort_session.get_inputs()[0].name: to_numpy(input_step), 
                        self.ort_session.get_inputs()[1].name : to_numpy(last_hidden),
                        self.ort_session.get_inputs()[2].name : to_numpy(encoder_outputs),
                    }
        ort_outs = self.ort_session.run(None, ort_inputs)

        return torch.tensor(ort_outs[0]), torch.tensor(ort_outs[1])
        