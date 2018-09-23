from __future__ import unicode_literals, print_function, division

from io import open
import glob
import os

import unicodedata
import string


def get_files(path): return glob.glob(path)


def print_data(): print('Data:', get_files('data/names/*.txt'), '\n')


all_letters = string.ascii_letters + " .,;'"
letters_len = len(all_letters)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) is not 'Mn'
        and c in all_letters
    )


def print_test_unicode(): print('test: ', unicode_to_ascii('test'), '\n')


print_data()
print_test_unicode()

categroy_lines = {}
all_categories = []


def read_lines(file):
    lines = open(file, encoding='utf-8')\
        .read()\
        .strip()\
        .split('\n')
    return [unicode_to_ascii(line) for line in lines]


def get_data():
    for file in get_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(file))[0]
        all_categories.append(category)
        lines = read_lines(file)
        categroy_lines[category] = lines

    categroy_len = len(all_categories)

    return categroy_len, categroy_lines, all_categories, letters_len, all_letters
