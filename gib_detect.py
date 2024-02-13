#!/usr/bin/python

import pickle
import gib_detect_train
print(" ")
print("   GIBBBERISH-DETECTOR")
print(" ")
model_data = pickle.load(open('gib_model.pki', 'rb'))
check = 1
while check == 1:
    l = str(input("Enter text: "))
    model_mat = model_data['mat']
    threshold = model_data['thresh']
    if (gib_detect_train.avg_transition_prob(l, model_mat) > threshold) == True:
        print("The text is not gibberish")
    else:
        print("The text is gibberish")
    print(" ")
    check = int(input("Press 1 continue: "))
    print(" ")