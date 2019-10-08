import argparse

from gibberish.config import MODEL_FILE_PATH
from gibberish.utils import avg_transition_prob, read_normalized_lines, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path', default=MODEL_FILE_PATH, help='File with model')
    parser.add_argument('-i', '--input', help='File with phrases')

    return parser.parse_args()


def is_gibberish(normalized_line, log_probs_mat, prob_threshold):
    """
    Detects whether the line is gibberish or not.

    :param normalized_line:
    :param log_probs_mat:
    :param prob_threshold:
    :return:
    """
    if avg_transition_prob(normalized_line, log_probs_mat) >= prob_threshold:
        return False
    else:
        return True


if __name__ == '__main__':
    args = parse_args()

    model_data = load_model(args.model_path)
    log_probs_mat = model_data['mat']
    prob_threshold = model_data['thresh']

    for line in read_normalized_lines(args.input):
        print('Received line: {}'.format(line))

        if is_gibberish(line, log_probs_mat, prob_threshold):
            print('Gibberish')
        else:
            print('Okay')
