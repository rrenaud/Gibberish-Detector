import os
import string

ACCEPTED_CHARS = string.ascii_lowercase + ' '
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ACCEPTED_CHARS)}

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_FILE_PATH = os.path.join(DATA_DIR, 'model.pkl')
