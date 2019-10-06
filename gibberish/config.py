import string

ACCEPTED_CHARS = string.ascii_lowercase + ' '
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ACCEPTED_CHARS)}
MODEL_FILE_PATH = 'data/model.pkl'
