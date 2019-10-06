import argparse

from gibberish.config import MODEL_FILE_PATH
from gibberish.utils import avg_transition_prob, read_normalized_lines, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path', default=MODEL_FILE_PATH, help='File with model')
    parser.add_argument('-i', '--input', help='File with phrases')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_data = load_model(args.model_path)
    log_probs_mat = model_data['mat']
    prob_threshold = model_data['thresh']

    for line in read_normalized_lines(args.input):
        print('Received line: {}'.format(line))

        if avg_transition_prob(line, log_probs_mat) >= prob_threshold:
            print('Okay')
        else:
            print('Gibberish')
