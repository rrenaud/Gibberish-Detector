import os
import string

ACCEPTED_CHARS = string.ascii_lowercase + ' '
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ACCEPTED_CHARS)}

DATA_DIR = 'data'
MODEL_FILE_PATH = os.path.join(DATA_DIR, 'model.pkl')
