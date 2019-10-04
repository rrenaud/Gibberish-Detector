import argparse
import logging
import pickle
import re
import string

import numpy as np
from nltk import ngrams

_logger = logging.getLogger(__name__)
_ACCEPTED_CHARS = string.ascii_lowercase + ' '
_CHAR_TO_IDX = {char: idx for idx, char in enumerate(_ACCEPTED_CHARS)}
_ALLOWED_LINE_PATTERN = re.compile('[^a-z ]')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='File with phrases', required=True)
    parser.add_argument('-o', '--output', default='model.pkl', help='Output file for string log probabilities')
    parser.add_argument('--bad', default='bad.txt', help='File with non-acceptable phrases')
    parser.add_argument('--good', default='good.txt', help='File with acceptable phrases')

    return parser.parse_args()


def normalize(line):
    """
    Returns only the subset of chars from `_ACCEPTED_CHARS`.
    This helps keep the  model relatively small by ignoring punctuation, 
    infrequent symbols, etc.

    :param line:
    :return:
    """
    return _ALLOWED_LINE_PATTERN.sub('', line.lower())


def avg_transition_prob(line, log_probs_mat):
    """
    Returns the average transition probability in `line` using `log_probs_mat`.

    :param line:
    :param log_probs_mat:
    :return:
    """
    log_prob = 0.0
    transition_count = 0

    for transition_count, (char1, char2) in enumerate(ngrams(line, 2)):
        log_prob += log_probs_mat[_CHAR_TO_IDX[char1]][_CHAR_TO_IDX[char2]]

    return np.exp(log_prob / max(transition_count, 1))


def calculate_log_probs_mat(input_path):
    """
    Calculates log-probabilities of transitions between characters.

    :return:
    """
    k = len(_ACCEPTED_CHARS)
    # Assume we have seen 10 of each character pair.  This acts as a kind of
    # prior or smoothing factor.  This way, if we see a character transition
    # live that we've never observed in the past, we won't assume the entire
    # string has 0 probability.
    counts = np.zeros((k, k), dtype=np.uint32)

    # Count transitions from big text file, taken from http://norvig.com/spell-correct.html
    _logger.info('Calculating transition counts...')
    with open(input_path) as fin:
        for line in fin:
            for char1, char2 in ngrams(line, 2):
                counts[_CHAR_TO_IDX[char1]][_CHAR_TO_IDX[char2]] += 1
    _logger.info('Finished calculating transition counts.')

    # Normalize the counts so that they become log probabilities.  
    # We use log probabilities rather than straight probabilities to avoid
    # numeric underflow issues with long texts.
    # This contains a justification:
    # http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
    log_probs_mat = np.log(counts / counts.sum(axis=1)[:, np.newaxis])

    return log_probs_mat


def calculate_prob_threshold(log_probs_mat, bad_phrases_file_path, good_phrases_file_path):
    """
    Calculates probability threshold for discern between bad and good phrases.

    :param log_probs_mat: 2D np.array corresponding to log-probabilities of transitions between characters
    :param bad_phrases_file_path: path to file with example of bad phrases
    :param good_phrases_file_path: path to file with example of good phrases
    :return:
    """
    _logger.info('Calculating probability threshold.')

    # Find the probability of generating a few arbitrarily chosen good and bad phrases.
    with open(bad_phrases_file_path) as fin:
        bad_probs = [avg_transition_prob(line, log_probs_mat) for line in fin]
    with open(good_phrases_file_path) as fin:
        good_probs = [avg_transition_prob(line, log_probs_mat) for line in fin]

    # Assert that we actually are capable of detecting the junk.
    if min(good_probs) < max(bad_probs):
        error_msg = 'Failed to discern between good and bad phrases.'
        _logger.error(error_msg)
        raise ValueError(error_msg)

    # And pick a threshold halfway between the worst good and best bad inputs.
    prob_threshold = (min(good_probs) + max(bad_probs)) / 2

    return prob_threshold


def save_model(output_path, log_probs_mat, prob_threshold):
    with open(output_path, 'wb') as fout:
        pickle.dump({'mat': log_probs_mat, 'thresh': prob_threshold}, fout)


if __name__ == '__main__':
    args = parse_args()

    log_probs_mat = calculate_log_probs_mat(args.input)
    prob_threshold = calculate_prob_threshold(log_probs_mat, args.bad, args.good)

    save_model(args.output, log_probs_mat, prob_threshold)
