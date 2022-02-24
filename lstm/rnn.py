import torch

class RNN(torch.nn.Module):

    def __init__(self, options={"input_length":1, "output_length":3, "type":'LSTM'}):
        super().__init__()
        self.options = options
        self.type = self.options['type']
        self.input_length = self.options['input_length']
        self.output_length = self.options['output_length']
        