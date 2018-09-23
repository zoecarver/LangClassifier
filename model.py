import torch
from torch import nn

from tensor import line_to_tensor
from dataset import get_data

import os


categroy_len, _, all_categories, letters_len, _ = get_data()

learning_rate = 0.005
name = 'model.pt'


class RNN(nn.Module):

    @staticmethod
    def model_exists():
        return os.path.exists(name)

    @staticmethod
    def load_model():
        #  model = RNN()
        #  model.load_state_dict(torch.load(name))

        return torch.load(name)

    def __init__(self, input_size=letters_len, output_size=categroy_len, hidden_size=128):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

        self.creterion = nn.NLLLoss()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self): return torch.zeros(1, self.hidden_size)

    def train(self, category_tensor, line_tensor):
        hidden = self.init_hidden()

        self.zero_grad()

        for index in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[index], hidden)

        loss = self.creterion(output, category_tensor)
        loss.backward()

        for param in self.parameters():
            param.data.add_(-learning_rate, param.grad.data)

        return output, loss.item()

    def evaluate(self, line_tensor):
        hidden = self.init_hidden()

        for index in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[index], hidden)

        return output

    def predict(self, input_line, predictions_count=3):
        print('\n> %s' % input_line)

        with torch.no_grad():
            output = self.evaluate(line_to_tensor(input_line))

            topv, topi = output.topk(predictions_count, 1, True)
            predictions = []

            for index in range(predictions_count):
                value = topv[0][index].item()
                category_index = topi[0][index].item()
                print('(%.2f) %s' % (value, all_categories[category_index]))
                predictions.append([value, all_categories[category_index]])

    def save(self):
        torch.save(self, name)
