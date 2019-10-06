import argparse
import os

import numpy as np
from nltk import ngrams

from gibberish.config import ACCEPTED_CHARS, CHAR_TO_IDX, DATA_DIR, MODEL_FILE_PATH
from gibberish.logger import get_logger
from gibberish.utils import avg_transition_prob, read_normalized_lines, save_model

_logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='File with phrases', required=True)
    parser.add_argument('-o', '--output', default=MODEL_FILE_PATH, help='Output file for string log probabilities')
    parser.add_argument('--bad', default=os.path.join(DATA_DIR, 'bad.txt'), help='File with non-acceptable phrases')
    parser.add_argument('--good', default=os.path.join(DATA_DIR, 'good.txt'), help='File with acceptable phrases')

    return parser.parse_args()


def calculate_log_probs_mat(input_path):
    """
    Calculates log-probabilities of transitions between characters.

    :return:
    """
    num_chars = len(ACCEPTED_CHARS)
    # Assume we have seen 10 of each character pair.  This acts as a kind of
    # prior or smoothing factor.  This way, if we see a character transition
    # live that we've never observed in the past, we won't assume the entire
    # string has 0 probability.
    counts = np.full((num_chars, num_chars), 10, dtype=np.uint32)

    # Count transitions from big text file, taken from http://norvig.com/spell-correct.html
    _logger.info('Calculating transition counts...')
    for line in read_normalized_lines(input_path):
        for char1, char2 in ngrams(line, 2):
            counts[CHAR_TO_IDX[char1]][CHAR_TO_IDX[char2]] += 1
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
    bad_probs = [avg_transition_prob(line, log_probs_mat) for line in read_normalized_lines(bad_phrases_file_path)]
    good_probs = [avg_transition_prob(line, log_probs_mat) for line in read_normalized_lines(good_phrases_file_path)]

    # Assert that we actually are capable of detecting the junk.
    if min(good_probs) < max(bad_probs):
        error_msg = 'Failed to discern between good and bad phrases.'
        _logger.error(error_msg)
        raise ValueError(error_msg)

    # And pick a threshold halfway between the worst good and best bad inputs.
    prob_threshold = (min(good_probs) + max(bad_probs)) / 2

    return prob_threshold


if __name__ == '__main__':
    args = parse_args()

    log_probs_mat = calculate_log_probs_mat(args.input)
    prob_threshold = calculate_prob_threshold(log_probs_mat, args.bad, args.good)

    save_model(args.output, log_probs_mat, prob_threshold)
