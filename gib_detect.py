#!/usr/bin/python

import pickle
import gib_detect_train

model_data = pickle.load(open('gib_model.pki'))

while True:
    l = raw_input()
    print gib_detect_train.char_ent(l, model_data['mat']) > model_data['thresh']
