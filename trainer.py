#Trainer module for obfuscated text classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.models import load_model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical

#Import data and labels
t_f = open('xtrain_obfuscated.txt', 'r')
X = [data.strip('\n') for data in t_f]

t_l = open('ytrain.txt','r')
Y_int = [int(target.strip('\n')) for target in t_l]
Y = to_categorical(Y_int)
print(Y)
print(np.shape(Y))
#Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

#Process the train data - Tokenizing
max_words = 9000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

#RNN structure
def RNN():
	inputs = Input(name='inputs',shape=[max_len])
	layer = Embedding(max_words,50,input_length=max_len)(inputs)
	layer = LSTM(64)(layer)
	layer = Dense(256,name='FC1')(layer)
	layer = Activation('relu')(layer)
	layer = Dropout(0.5)(layer)
	layer = Dense(12,name='out_layer')(layer)
	layer = Activation('softmax')(layer)
	model = Model(inputs=inputs,outputs=layer)
	return model

#Compile the model
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

#Fit on training data
model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10, validation_split=0.2)

model.save('RNN_trainer_model.h5')
