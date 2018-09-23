import torch

from dataset import get_data
from plot import plot_all_losses
from tensor import line_to_tensor, letter_to_tensor
from model import RNN

import random
import time
import math


# Most of these should be moved to a config file
epochs = 100000
print_every = 5000
plot_every = 1000

hidden_len = 128

categroy_len, categroy_lines, all_categories, letters_len, all_letters = get_data()


def test_letter_to_tensor(): print('test `J`:', letter_to_tensor('J'))


def test_line_to_tensor(): print('[size] test `Jones: ', line_to_tensor('Jones').size())


def category_from_output(output):
    top_len, top_index = output.topk(1)
    category_index = top_index[0].item()

    return all_categories[category_index], category_index


def random_choice(lang):
    return lang[random.randint(0, len(lang) - 1)]


def random_training_example():
    category = random_choice(all_categories)
    # print(category)
    line = random_choice(categroy_lines[category])
    # print(line)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)

    return category, line, category_tensor, line_tensor


def time_from(then):
    now = time.time()
    s = now - then
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)


def print_epoch_guess(epoch, start, loss, line, guess, correct):
    print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epochs / epoch * 100, time_from(start), loss, line, guess, correct))


def train_model():
    current_loss = 0
    all_losses = []

    start = time.time()

    for epoch in range(1, epochs + 1):
        category, line, category_tensor, line_tensor = random_training_example()
        output, loss = rnn.train(category_tensor, line_tensor)
        current_loss += loss

        if epoch % print_every == 0:
            guess, guess_index = category_from_output(output)
            correct = 'Got it' if guess == category else 'Thought it was %s' % category
            print_epoch_guess(epoch, start, loss, line, guess, correct)

        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    rnn.save()


test_letter_to_tensor()
test_line_to_tensor()

if RNN.model_exists():
    print('loading saved model')
    rnn = RNN.load_model()
else:
    print('creating model')
    rnn = RNN()
    print('training model')
    train_model()

while True:
    input_name = raw_input()
    rnn.predict(input_name)

#  plot_all_losses(all_losses)
