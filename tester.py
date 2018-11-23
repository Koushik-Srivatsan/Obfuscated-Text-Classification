import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical

#Import data and labels
t_f = open('xtrain_obfuscated.txt', 'r')
X = [data.strip('\n') for data in t_f]

t_l = open('ytrain.txt','r')
Y_int = [int(target.strip('\n')) for target in t_l]
Y = to_categorical(Y_int)

#Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

#Process the train data - Tokenizing
max_words = 9000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

#Load the model
model = load_model('RNN_trainer_model.h5')

accr = model.evaluate(test_sequences_matrix, Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))