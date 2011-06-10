A sample program I wrote to detect gibberish.  It uses a 2 character markov chain.

http://stackoverflow.com/questions/6297991/is-there-any-way-to-detect-strings-like-putjbtghguhjjjanika/6298040#comment-7360747

http://en.wikipedia.org/wiki/Markov_chain

First train the model:

python gib_detect_train.py

Then try it on some sample input

python gib_detect.py

my name is rob and i like to hack
True
is this thing working?
True
i hope so
True
t2 chhsdfitoixcv
False
ytjkacvzw
False
yutthasxcvqer
False
seems okay
True
yay!
True
