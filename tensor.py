import torch

from dataset import get_data


_, _, _, letters_len, all_letters = get_data()


def letter_to_index(letter): return all_letters.find(letter)


def letter_to_tensor(letter):
    tensor = torch.zeros(1, letters_len)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    line_len = len(line)
    tensor = torch.zeros(line_len, 1, letters_len)

    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1

    return tensor
