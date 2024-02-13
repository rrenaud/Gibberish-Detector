# GIBBERISH-DETECTOR

A program to check if a sentence is gibberish or not using simple markov chain

## 1. Getting Started

Clone the repo:

  ```bash
  git clone https://github.com/rrenaud/Gibberish-Detector.git
  ```
Train the model:

```
python gib_detect_train.py
```
Run the model:

```
python gib_detect.py
```

## 2. How it works

The markov chain first 'trains' or 'studies' a few MB of English text, recording how often characters appear next to each other. Eg, given the text "Rob likes hacking" it sees Ro, ob, o[space], [space]l, ... It just counts these pairs. After it has finished reading through the training data, it normalizes the counts. Then each character has a probability distribution of 27 followup character (26 letters + space) following the given initial.

So then given a string, it measures the probability of generating that string according to the summary by just multiplying out the probabilities of the adjacent pairs of characters in that string. EG, for that "Rob likes hacking" string, it would compute prob['r']['o'] * prob['o']['b'] * prob['b'][' '] ... This probability then measures the amount of 'surprise' assigned to this string according the data the model observed when training. If there is funny business with the input string, it will pass through some pairs with very low counts in the training phase, and hence have low probability/high surprise.

I then look at the amount of surprise per character for a few known good strings, and a few known bad strings, and pick a threshold between the most surprising good string and the least surprising bad string. Then I use that threshold whenever to classify any new piece of text.

Peter Norvig, the director of Research at Google, has this nice talk about "The unreasonable effectiveness of data" here, http://www.youtube.com/watch?v=9vR8Vddf7-s. This insight is really not to try to do something complicated, just write a small program that utilizes a bunch of data and you can do cool things.

![sample](/sample.png)

