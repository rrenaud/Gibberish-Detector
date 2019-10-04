#!/usr/bin/python

import pickle
from src import train

model_data = pickle.load(open('gib_model.pki', 'rb'))

while True:
    l = raw_input()
    model_mat = model_data['mat']
    threshold = model_data['thresh']
    print train.avg_transition_prob(l, model_mat) > threshold
