import pickle
import torch
import torch.nn as nn


from .seq2seq_onnx import EncoderONNX, DecoderONNX
from .seq2seq import EncoderRNN, LuongAttnDecoderRNN
from .greedy_search import GreedySearch

from .tokenizer import indexesFromSentence, normalizeString, SOS_token

class InferenceModel():
    def __init__ (self, model_path, attn_model, hidden_size, num_words, encoder_n_layers, decoder_n_layers, backend = 'onnx'):
        embedding = nn.Embedding(num_words, hidden_size)
        
         # check device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
              
        if backend == 'pt':
            self.encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers)
            self.decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, num_words, decoder_n_layers)
            
            # load models
            self.encoder.load_state_dict(torch.load(f'{model_path}/kaggle-encoder.pt'))
            self.decoder.load_state_dict(torch.load(f'{model_path}/kaggle-decoder.pt'))
            
            # eval mode
            self.encoder.eval()
            self.decoder.eval()
         
            # set up device
            self.encoder.to(self.device)
            self.decoder.to(self.device)
        elif backend == 'onnx':
            self.device = 'cpu'
            self.encoder = EncoderONNX(f'{model_path}/kaggle-encoder.onnx')
            self.decoder = DecoderONNX(f'{model_path}/kaggle-decoder.onnx', decoder_n_layers)
        
        self.searcher = GreedySearch(self.encoder, self.decoder, SOS_token, self.device)
        with open(f'{model_path}/voc_tokenizer.pkl', 'rb') as inp:
            self.voc_tokenizer =  pickle.load(inp)
            
        print('Device: ', self.device) 
           
    def __call__(self, sentence, max_length = 20):
        try:
            # Normalize sentence
            input_sentence = normalizeString(sentence)
            
            # words -> indexes
            indexes_batch = [indexesFromSentence(self.voc_tokenizer, input_sentence)]
            # Create lengths tensor
            lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
            # Transpose dimensions of batch to match models' expectations
            input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
            # Use appropriate device
            input_batch = input_batch.to(self.device)
            lengths = lengths.to("cpu")
            # Decode sentence with searcher
            tokens, scores = self.searcher(input_batch, lengths, max_length)
            # indexes -> words
            output_words = [self.voc_tokenizer.index2word[token.item()] for token in tokens]
            
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            
            output_words = ' '.join(output_words) +'.'
        except:
            output_words = "I don't undertand your message bacause I was trained with small dataset, please send me other text."
            
        return output_words