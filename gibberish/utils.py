import pickle
import re

import numpy as np
from nltk import ngrams

from gibberish.config import ACCEPTED_CHARS, CHAR_TO_IDX
from gibberish.logger import get_logger

_ALLOWED_LINE_PATTERN = re.compile('[^{}]'.format(ACCEPTED_CHARS))
_logger = get_logger(__name__)


def normalize(line):
    """
    Returns only the subset of chars from `ACCEPTED_CHARS`.
    This helps keep the  model relatively small by ignoring punctuation,
    infrequent symbols, etc.

    :param line:
    :return:
    """
    return _ALLOWED_LINE_PATTERN.sub('', line.lower())


def avg_transition_prob(normalized_line, log_probs_mat):
    """
    Returns the average transition probability in `normalized_line` using `log_probs_mat`.

    :param normalized_line:
    :param log_probs_mat:
    :return:
    """
    log_prob = 0.0
    transition_count = 0

    for transition_count, (char1, char2) in enumerate(ngrams(normalized_line, 2)):
        log_prob += log_probs_mat[CHAR_TO_IDX[char1]][CHAR_TO_IDX[char2]]

    return np.exp(log_prob / max(transition_count, 1))


def read_normalized_lines(file_path=None):
    """
    Reads lines from file or input and normalizes them.

    :param file_path:
    :return:
    """
    if file_path:
        _logger.info('Reading lines from file {}'.format(file_path))
        with open(file_path, encoding='utf8') as fin:
            for line in fin:
                yield normalize(line)
    else:
        _logger.info('Reading lines from input')
        print('Please, write your input:\n')
        while True:
            line = input()
            yield normalize(line)


def save_model(model_path, log_probs_mat, prob_threshold):
    with open(model_path, 'wb') as fout:
        pickle.dump({'mat': log_probs_mat, 'thresh': prob_threshold}, fout)


def load_model(model_path):
    with open(model_path, 'rb') as fin:
        return pickle.load(fin)
